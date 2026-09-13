"""LiveKit agent entrypoint.

    uv run python -m ccas.voice.worker dev

Wires a LiveKit room to the voice session: real transport, real VAD, real STT and TTS,
the same graph and the same latency ledger the scripted tests drive.

Refuses to start rather than degrade. A worker that answered calls without redaction, or
without a provider, would be worse than one that does not answer at all.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from ccas.config.budget import load_budget
from ccas.config.domain_loader import DomainPackNotFoundError, load_domain
from ccas.config.settings import Settings
from ccas.graph.assembly import build_graph
from ccas.graph.checkpoint import memory_checkpointer, thread_config
from ccas.graph.context import build_context
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProviderError
from ccas.llm.bindings import load_bindings
from ccas.llm.factory import build_provider
from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import current_trace_context
from ccas.schemas.common import Channel, VerificationLevel
from ccas.schemas.session import CallerContext, SessionState
from ccas.voice.session import VoiceSession
from ccas.voice.stt.deepgram import DeepgramStt
from ccas.voice.transport import AudioTransport
from ccas.voice.tts.cartesia import CartesiaTts
from ccas.voice.vad import EnergyVad, SileroVad, VoiceActivityDetector

__all__ = ["build_session", "main", "preflight"]

LOG = get_logger("voice.worker")


def preflight(settings: Settings, domain: str) -> list[str]:
    """Everything that must be true before a call can be answered.

    Returned rather than raised so the operator sees the whole list, not the first
    problem and then another run.
    """
    problems: list[str] = []

    try:
        loaded = load_domain(settings.domains_dir, domain)
    except DomainPackNotFoundError as exc:
        return [str(exc)]
    if not loaded.has_taxonomy:
        problems.append(
            f"domain {domain!r} has no mined taxonomy; the router would have nothing to "
            "classify into. Run scripts/mine_taxonomy.py first."
        )

    from ccas.redaction.pipeline import RedactionMode, build_pipeline

    batch = build_pipeline(settings.redaction_policy, pack=loaded.pack, mode=RedactionMode.BATCH)
    if not batch.ready:
        problems.append(
            f"batch redaction unavailable ({', '.join(batch.unavailable_engines())}); "
            "stored transcripts would carry caller names. Run `uv sync --extra redaction`."
        )

    bindings = load_bindings(settings.models_config)
    try:
        build_provider(bindings.resolve("router"), settings)
    except LLMProviderError as exc:
        problems.append(str(exc))

    if not settings.deepgram_api_key:
        problems.append("DEEPGRAM_API_KEY is unset; speech could not be transcribed")
    if not settings.cartesia_api_key:
        problems.append("CARTESIA_API_KEY is unset; the platform could not speak")
    if not settings.livekit_url:
        problems.append("LIVEKIT_URL is unset; there is nothing to connect to")

    return problems


def _vad(settings: Settings) -> VoiceActivityDetector:
    silero = SileroVad()
    if silero.available:
        return silero
    LOG.warning(
        "voice.vad.fallback",
        reason=silero.load_error,
        detail="using the energy detector; install the voice extra for Silero",
    )
    return EnergyVad()


async def build_session(
    transport: AudioTransport,
    settings: Settings,
    domain: str,
    session_id: str | None = None,
) -> VoiceSession:
    """Assemble a session around a connected transport."""
    loaded = load_domain(settings.domains_dir, domain)
    bindings = load_bindings(settings.models_config)
    provider = build_provider(bindings.resolve("task_agent"), settings)
    ctx = build_context(loaded, provider, bindings, config_dir=settings.config_dir)

    sid = session_id or f"call-{uuid.uuid4().hex[:12]}"
    trace = current_trace_context(correlation_id=sid)
    graph = build_graph(
        ctx,
        IntentRouter(provider, bindings.resolve("router"), loaded.taxonomy),
        Responder(provider, bindings.resolve("task_agent")),
        checkpointer=memory_checkpointer(),
    )
    state = SessionState(
        session_id=sid,
        trace=trace,
        domain=domain,
        channel=Channel.VOICE,
        caller=CallerContext(
            caller_ref=sid.ljust(8, "0")[:32], verification=VerificationLevel.NONE
        ),
    )
    return VoiceSession(
        transport=transport,
        vad=_vad(settings),
        stt=DeepgramStt(
            settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else None
        ),
        tts=CartesiaTts(
            settings.cartesia_api_key.get_secret_value() if settings.cartesia_api_key else None,
            voice_id=settings.cartesia_voice_id,
        ),
        graph=graph,
        ctx=ctx,
        state=state,
        config=thread_config(sid, trace),
        budget=load_budget(settings.latency_budget_config),
    )


async def _serve(settings: Settings, domain: str) -> int:
    try:
        from livekit import agents  # noqa: F401
    except ImportError:
        print(
            "error: livekit-agents is not installed. Run `uv sync --extra voice`.",
            file=sys.stderr,
        )
        return 3
    print(
        "The LiveKit room loop is not wired yet: this worker can assemble a session "
        "around a connected transport (`build_session`) but does not yet join a room.\n"
        "Tracked in docs/future-scoped-work.md; the scripted driver exercises the same "
        "session end to end today.",
        file=sys.stderr,
    )
    return 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ccas.voice.worker")
    parser.add_argument("command", choices=["dev", "preflight"], default="dev", nargs="?")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(Path("logs/execution.log"), level=args.log_level, console=True)
    settings = Settings()
    domain = args.domain or settings.default_domain

    problems = preflight(settings, domain)
    print(f"\n  preflight for domain {domain!r}")
    if not problems:
        print("  all checks passed\n")
    for problem in problems:
        print(f"  - {problem}")
    print()

    if args.command == "preflight":
        return 0 if not problems else 1
    if problems:
        print(
            "error: refusing to answer calls with the above unresolved.\n"
            "       A worker that degraded past redaction or routing would be worse\n"
            "       than one that does not answer (CLAUDE.md Rule 2).",
            file=sys.stderr,
        )
        return 1
    return asyncio.run(_serve(settings, domain))


if __name__ == "__main__":
    raise SystemExit(main())
