"""Everything a node is allowed to reach.

Nodes are closures over this object rather than importing their dependencies, so a node
cannot quietly acquire one. It also makes the whole mesh constructible in a test with
stubs, which is what keeps the graph tests free of network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ccas.config.channels import CHANNELS_FILENAME, ChannelProfile, load_channel_profiles
from ccas.config.domain_loader import LoadedDomain
from ccas.llm.base import LLMProvider
from ccas.llm.bindings import BindingRegistry
from ccas.policies.base import PolicyContext
from ccas.policies.engine import PolicyEngine
from ccas.redaction.pipeline import RedactionPipeline, build_pipeline
from ccas.redaction.vault import PlaceholderVault
from ccas.schemas.common import Channel
from ccas.schemas.domain import DomainPack
from ccas.schemas.session import SessionState
from ccas.schemas.taxonomy import IntentNode, IntentTaxonomy
from ccas.tools.executor import ToolExecutor
from ccas.tools.mock_backend import MockToolBackend
from ccas.tools.registry import ToolRegistry, build_registry

__all__ = ["GraphContext", "build_context"]


@dataclass(frozen=True, slots=True)
class GraphContext:
    channel: Channel
    profile: ChannelProfile
    domain: LoadedDomain
    registry: ToolRegistry
    executor: ToolExecutor
    redaction: RedactionPipeline
    policies: PolicyEngine
    bindings: BindingRegistry
    provider: LLMProvider
    vault: PlaceholderVault
    max_no_input: int = 3
    max_no_match: int = 3

    @property
    def pack(self) -> DomainPack:
        return self.domain.pack

    @property
    def taxonomy(self) -> IntentTaxonomy | None:
        return self.domain.taxonomy if self.domain.has_taxonomy else None

    def node_for(self, state: SessionState) -> IntentNode | None:
        """The taxonomy node for the session's current intent, if it resolves."""
        taxonomy = self.taxonomy
        prediction = state.current_intent
        if taxonomy is None or prediction is None or prediction.intent_id is None:
            return None
        try:
            return taxonomy.resolve(prediction.intent_id)
        except KeyError:
            # The router named something outside the taxonomy. Treated as unresolved
            # rather than fatal: the confidence policy will handle it.
            return None

    def policy_context(self, state: SessionState) -> PolicyContext:
        return PolicyContext.for_intent(
            self.pack,
            self.node_for(state),
            max_no_input=self.max_no_input,
            max_no_match=self.max_no_match,
        )


def build_context(
    domain: LoadedDomain,
    provider: LLMProvider,
    bindings: BindingRegistry,
    *,
    channel: Channel = Channel.CHAT,
    config_dir: Path = Path("configs"),
    tool_budget_ms: int = 150,
) -> GraphContext:
    """Assemble a context with the mock tool backend. Real backends are wired at deploy.

    The channel decides how hard to redact, and it is a *profile lookup*, not a branch
    (ADR-0019). Chat defaults because chat is the proving ground now (ADR-0018); voice is
    opt-in and keeps the REALTIME trade its 3 ms slice forces on it.
    """
    pack = domain.pack
    profile = load_channel_profiles(config_dir / CHANNELS_FILENAME).require(channel)
    redaction = build_pipeline(
        config_dir / "redaction_policy.yaml", pack=pack, mode=profile.redaction_mode
    )
    registry = build_registry(pack, config_dir / "tools.yaml")
    vault = PlaceholderVault()
    executor = ToolExecutor(
        registry,
        MockToolBackend.from_pack_dir(domain.root),
        redaction,
        total_budget_ms=tool_budget_ms,
        vault=vault,
    )
    return GraphContext(
        channel=channel,
        profile=profile,
        domain=domain,
        registry=registry,
        executor=executor,
        redaction=redaction,
        policies=PolicyEngine(),
        bindings=bindings,
        provider=provider,
        vault=vault,
    )
