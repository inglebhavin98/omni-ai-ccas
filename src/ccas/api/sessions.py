"""Session lifecycle for the workbench and the API.

Each session owns its own ``GraphContext`` -- and therefore its own
``PlaceholderVault``, which must never be shared across callers (ADR-0010). The manager
holds them in memory, so it is single-process by construction; that is the same
constraint the voice worker already lives under.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ccas.config.domain_loader import DomainPackNotFoundError, available_domains, load_domain
from ccas.config.settings import Settings
from ccas.graph.assembly import build_graph
from ccas.graph.checkpoint import memory_checkpointer, thread_config
from ccas.graph.context import GraphContext, build_context
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProvider, LLMProviderError
from ccas.llm.bindings import BindingRegistry, load_bindings
from ccas.llm.factory import build_provider
from ccas.observability.logging import get_logger
from ccas.observability.tracing import current_trace_context
from ccas.schemas.common import Channel, Speaker, VerificationLevel
from ccas.schemas.session import CallerContext, SessionState, Turn

__all__ = ["SessionHandle", "SessionManager", "SessionNotFoundError"]

LOG = get_logger("api.sessions")


class SessionNotFoundError(LookupError):
    """Asked for a session this process does not hold."""


@dataclass(slots=True)
class SessionHandle:
    session_id: str
    domain: str
    ctx: GraphContext
    graph: Any
    config: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> SessionState:
        return SessionState.model_validate(self.state)

    def transcript(self) -> list[dict[str, Any]]:
        return [
            {
                "index": turn.index,
                "speaker": turn.speaker.value,
                "text": turn.content.text,
                "redacted_entities": turn.content.report.entity_counts,
                "at": turn.at.isoformat(),
            }
            for turn in self.snapshot().turns
        ]


class SessionManager:
    """In-memory session registry. One process, one set of vaults."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        bindings: BindingRegistry | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.bindings = bindings or load_bindings(self.settings.models_config)
        self._injected_provider = provider
        self._sessions: dict[str, SessionHandle] = {}
        self._provider_error: str | None = None

    # ------------------------------------------------------------- readiness

    @property
    def domains(self) -> tuple[str, ...]:
        return available_domains(self.settings.domains_dir)

    def provider_for(self, node: str) -> LLMProvider | None:
        """The configured provider, or None with the reason recorded.

        A workbench that silently substituted a scripted responder would show behaviour
        the platform does not have -- the same reason a pending demo stage never
        fabricates output (CLAUDE.md Rule 11).
        """
        if self._injected_provider is not None:
            return self._injected_provider
        try:
            return build_provider(self.bindings.resolve(node), self.settings)
        except LLMProviderError as exc:
            self._provider_error = str(exc)
            return None

    def readiness(self) -> dict[str, Any]:
        provider = self.provider_for("router")
        domains: dict[str, Any] = {}
        for name in self.domains:
            try:
                loaded = load_domain(self.settings.domains_dir, name)
            except (DomainPackNotFoundError, ValueError) as exc:
                domains[name] = {"loads": False, "error": str(exc)}
                continue
            domains[name] = {
                "loads": True,
                "has_taxonomy": loaded.has_taxonomy,
                "tools": len(loaded.pack.tools),
            }
        return {
            "provider_configured": provider is not None,
            "provider_error": self._provider_error,
            "default_provider": self.bindings.default_provider.value,
            "default_domain": self.settings.default_domain,
            "domains": domains,
            "sessions": len(self._sessions),
        }

    # -------------------------------------------------------------- lifecycle

    async def create(
        self,
        domain: str | None = None,
        verification: VerificationLevel = VerificationLevel.NONE,
    ) -> SessionHandle:
        name = domain or self.settings.default_domain
        loaded = load_domain(self.settings.domains_dir, name)
        provider = self.provider_for("task_agent")

        ctx = build_context(
            loaded,
            provider or _UnavailableProvider(),
            self.bindings,
            config_dir=self.settings.config_dir,
        )
        router = (
            IntentRouter(provider, self.bindings.resolve("router"), loaded.taxonomy)
            if provider is not None and loaded.has_taxonomy
            else None
        )
        responder = (
            Responder(provider, self.bindings.resolve("task_agent"))
            if provider is not None
            else None
        )

        session_id = f"wb-{uuid.uuid4().hex[:12]}"
        trace = current_trace_context(correlation_id=session_id)
        graph = build_graph(ctx, router, responder, checkpointer=memory_checkpointer())
        config = thread_config(session_id, trace)

        state = await graph.ainvoke(
            SessionState(
                session_id=session_id,
                trace=trace,
                domain=name,
                channel=Channel.CHAT,
                caller=CallerContext(
                    caller_ref=session_id.ljust(8, "0")[:32], verification=verification
                ),
            ),
            config,
        )
        handle = SessionHandle(
            session_id=session_id,
            domain=name,
            ctx=ctx,
            graph=graph,
            config=config,
            state=state,
        )
        self._sessions[session_id] = handle
        LOG.info(
            "session.created",
            correlation_id=session_id,
            domain=name,
            has_taxonomy=loaded.has_taxonomy,
            provider=provider is not None,
        )
        return handle

    def get(self, session_id: str) -> SessionHandle:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"no session {session_id!r} in this process") from exc

    async def say(self, session_id: str, text: str) -> SessionHandle:
        handle = self.get(session_id)
        content = handle.ctx.redaction.redact(
            text, handle.ctx.redaction.new_allocator(handle.ctx.vault)
        )
        index = int(handle.state.get("turn_index", 0))
        turn = Turn(index=index, speaker=Speaker.CALLER, content=content)
        handle.state = await handle.graph.ainvoke(
            {"turns": [turn], "turn_index": index + 1}, handle.config
        )
        LOG.info(
            "session.turn",
            correlation_id=session_id,
            turn_index=index,
            redacted_entities=content.report.entity_counts,
            escalated=handle.state.get("escalation") is not None,
        )
        return handle

    def drop(self, session_id: str) -> None:
        """Discard a session and, with it, its vault."""
        self._sessions.pop(session_id, None)


class _UnavailableProvider(LLMProvider):
    """Stands in so a context is constructible without a provider.

    Every method raises. The graph is built with ``router=None``/``responder=None`` in
    that case, so nothing calls it -- and if something ever did, it fails loudly rather
    than inventing an answer.
    """

    from ccas.schemas.llm import ProviderName as _ProviderName

    name = _ProviderName.VLLM

    async def complete(self, request: Any) -> Any:
        raise LLMProviderError("no LLM provider is configured")

    def stream(self, request: Any) -> Any:
        raise LLMProviderError("no LLM provider is configured")

    async def structured(self, request: Any) -> Any:
        raise LLMProviderError("no LLM provider is configured")

    async def healthy(self) -> bool:
        return False
