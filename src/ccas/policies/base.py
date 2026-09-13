"""Policy vocabulary.

A policy is a pure function: session state and thresholds in, a verdict out. No I/O, no
LLM, no clock. That is what makes the routing decisions in a voice call reviewable --
every escalation can be traced to a named rule and a number, not to a model's mood.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from ccas.schemas.common import Frozen, RiskTier, Slug, Urgency
from ccas.schemas.domain import ConfidenceThresholds, DomainPack
from ccas.schemas.escalation import EscalationDecision, HandoffReason
from ccas.schemas.session import SessionState
from ccas.schemas.taxonomy import EscalationPolicy, IntentNode

__all__ = [
    "PROCEED",
    "Policy",
    "PolicyAction",
    "PolicyContext",
    "PolicyDecision",
    "proceed",
]


class PolicyAction(StrEnum):
    PROCEED = "proceed"
    """Nothing to say; let the graph continue."""

    CLARIFY = "clarify"
    """Ask one disambiguating question. Bounded by max_clarifications."""

    ESCALATE = "escalate"
    """Hand to a human. Terminal for the bot."""


class PolicyDecision(Frozen):
    action: PolicyAction
    policy: Slug
    """Which rule fired. Becomes ``EscalationDecision.triggered_by``, so a handoff can
    always name the reason rather than implying one."""

    reason: HandoffReason | None = None
    urgency: Urgency = Urgency.NORMAL
    detail: str | None = None

    @property
    def decisive(self) -> bool:
        return self.action is not PolicyAction.PROCEED

    def to_escalation(self, at_turn: int) -> EscalationDecision:
        if self.action is not PolicyAction.ESCALATE or self.reason is None:
            raise ValueError(f"policy {self.policy!r} did not decide to escalate")
        return EscalationDecision(
            reason=self.reason,
            triggered_by=self.policy,
            at_turn=at_turn,
            urgency=self.urgency,
            detail=self.detail,
        )


def proceed(policy: str = "none") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.PROCEED, policy=policy)


PROCEED = proceed()


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Everything a policy is allowed to see, pulled from the pack and the taxonomy.

    Deliberately not the pack itself: a policy that could reach the whole pack would
    grow a dependency on a vertical's shape.
    """

    confidence: ConfidenceThresholds
    escalation: EscalationPolicy
    risk_tier: RiskTier
    max_turns: int
    default_queue: Slug
    max_no_input: int = 3
    max_no_match: int = 3

    @classmethod
    def for_intent(
        cls,
        pack: DomainPack,
        node: IntentNode | None = None,
        max_no_input: int = 3,
        max_no_match: int = 3,
    ) -> PolicyContext:
        """Intent-specific thresholds where the taxonomy has them, pack defaults otherwise."""
        return cls(
            confidence=pack.confidence,
            escalation=node.escalation if node else EscalationPolicy(),
            risk_tier=node.risk_tier if node else RiskTier.LOW,
            max_turns=pack.max_turns,
            default_queue=(
                node.escalation.target_queue
                if node and node.escalation.target_queue
                else pack.default_queue.name
            ),
            max_no_input=max_no_input,
            max_no_match=max_no_match,
        )


Policy = Callable[[SessionState, PolicyContext], PolicyDecision]
