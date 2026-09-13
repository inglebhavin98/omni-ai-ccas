"""Risk-tier policy.

The hardest rule in the mesh, and the first evaluated: an intent the pack marks
``REGULATED`` is one a bot must never attempt, whatever its confidence (CLAUDE.md
Rule 4). Which intents are regulated is pack data -- the core does not know what makes
something regulated, only that it must not proceed.
"""

from __future__ import annotations

from ccas.policies.base import PolicyAction, PolicyContext, PolicyDecision, proceed
from ccas.schemas.common import RiskTier, Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import SessionState

__all__ = ["NAME", "evaluate"]

NAME = "risk"


def evaluate(state: SessionState, ctx: PolicyContext) -> PolicyDecision:
    if ctx.risk_tier is RiskTier.REGULATED or ctx.escalation.auto_escalate:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            policy=NAME,
            reason=HandoffReason.RISK_TIER,
            urgency=Urgency.HIGH,
            detail=(
                f"intent {state.current_intent.intent_id if state.current_intent else '?'} "
                f"is {ctx.risk_tier.value}; automated handling is not permitted"
            ),
        )
    return proceed(NAME)
