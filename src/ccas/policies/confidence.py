"""Intent-confidence policy.

Three bands, from the pack's ``ConfidenceThresholds``:

    >= route              dispatch to the intent's agent
    clarify_floor .. route ask one disambiguating question
    < clarify_floor        escalate -- clarifying a guess this weak wastes the caller's turn

Clarification is bounded. A second failed clarification escalates rather than looping,
because a caller who has been asked twice has already decided the bot cannot help.
"""

from __future__ import annotations

from ccas.policies.base import PolicyAction, PolicyContext, PolicyDecision, proceed
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import SessionState

__all__ = ["NAME", "evaluate"]

NAME = "confidence"


def evaluate(state: SessionState, ctx: PolicyContext) -> PolicyDecision:
    prediction = state.current_intent
    if prediction is None:
        return proceed(NAME)

    threshold = max(ctx.confidence.route, ctx.escalation.min_intent_confidence)

    if prediction.resolved and prediction.confidence >= threshold:
        return proceed(NAME)

    exhausted = state.clarification_count >= ctx.escalation.max_clarifications
    too_weak = prediction.confidence < ctx.confidence.clarify_floor

    if too_weak or exhausted:
        # A router that never answered has not failed to understand -- nobody was asked.
        # The distinction reaches the CRM and shapes what the receiving agent is told, so
        # an outage must not be filed as "we understood and cannot help".
        if prediction.source == "unavailable":
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                policy=NAME,
                reason=HandoffReason.SYSTEM_ERROR,
                detail="router unavailable; the utterance was never classified",
            )
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            policy=NAME,
            reason=(HandoffReason.UNSUPPORTED_INTENT if too_weak else HandoffReason.LOW_CONFIDENCE),
            detail=(
                f"confidence {prediction.confidence:.2f} below {threshold:.2f} after "
                f"{state.clarification_count} clarification(s)"
            ),
        )

    return PolicyDecision(
        action=PolicyAction.CLARIFY,
        policy=NAME,
        detail=f"confidence {prediction.confidence:.2f} below {threshold:.2f}",
    )
