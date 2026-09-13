"""An unavailable router is a system error, not an unsupported intent.

Observed live: a free-tier 429 escalated with `reason=unsupported_intent`, which tells a
human reviewer the bot understood the caller and had no way to serve them. It had
understood nothing -- the router never answered. The disposition travels to the CRM and
shapes what the receiving agent is told, so the wrong reason is a lie with consequences.
"""

from __future__ import annotations

from ccas.policies.base import PolicyAction, PolicyContext
from ccas.policies.confidence import evaluate
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import IntentPrediction, SessionState


def _with(session: SessionState, prediction: IntentPrediction) -> SessionState:
    return session.model_copy(update={"current_intent": prediction, "clarification_count": 5})


def test_an_unavailable_router_escalates_as_a_system_error(
    session: SessionState, ctx: PolicyContext
) -> None:
    state = _with(
        session,
        IntentPrediction(intent_id=None, confidence=0.0, source="unavailable", latency_ms=0),
    )
    decision = evaluate(state, ctx)
    assert decision.action is PolicyAction.ESCALATE
    assert decision.reason is HandoffReason.SYSTEM_ERROR
    assert "unavailable" in (decision.detail or "").lower()


def test_a_genuinely_unsupported_intent_still_says_so(
    session: SessionState, ctx: PolicyContext
) -> None:
    """The guard must not relabel every low-confidence escalation as an outage."""
    state = _with(
        session,
        IntentPrediction(intent_id=None, confidence=0.0, source="llm_router", latency_ms=8),
    )
    assert evaluate(state, ctx).reason is HandoffReason.UNSUPPORTED_INTENT
