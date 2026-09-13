from __future__ import annotations

from dataclasses import replace

import pytest

from ccas.policies import confidence, retry, risk, sentiment
from ccas.policies.base import PolicyAction, PolicyContext, PolicyDecision
from ccas.policies.engine import DEFAULT_POLICIES, PolicyEngine, evaluate
from ccas.schemas.common import RiskTier, Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import IntentPrediction, SentimentSnapshot, SessionState
from ccas.schemas.taxonomy import EscalationPolicy


def predict(value: float, intent: str | None = "billing.dispute.status") -> IntentPrediction:
    return IntentPrediction(intent_id=intent, confidence=value, source="llm_router", latency_ms=40)


def frustration(*values: float) -> list[SentimentSnapshot]:
    return [
        SentimentSnapshot(valence=-v, arousal=v, frustration_index=v, turn_index=i)
        for i, v in enumerate(values)
    ]


# ------------------------------------------------------------------------ risk


def test_a_regulated_intent_escalates_before_anything_else(
    session: SessionState, ctx: PolicyContext
) -> None:
    """CLAUDE.md Rule 4: a bot must never attempt one, whatever its confidence."""
    session.current_intent = predict(0.99)
    decision = risk.evaluate(session, replace(ctx, risk_tier=RiskTier.REGULATED))
    assert decision.action is PolicyAction.ESCALATE
    assert decision.reason is HandoffReason.RISK_TIER
    assert decision.urgency is Urgency.HIGH


def test_an_auto_escalate_intent_escalates_at_any_tier(
    session: SessionState, ctx: PolicyContext
) -> None:
    guarded = replace(ctx, escalation=EscalationPolicy(auto_escalate=True))
    assert risk.evaluate(session, guarded).action is PolicyAction.ESCALATE


@pytest.mark.parametrize("tier", [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH])
def test_unregulated_tiers_proceed(
    session: SessionState, ctx: PolicyContext, tier: RiskTier
) -> None:
    assert risk.evaluate(session, replace(ctx, risk_tier=tier)).action is PolicyAction.PROCEED


# ------------------------------------------------------------------ confidence


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.95, PolicyAction.PROCEED),
        (0.82, PolicyAction.PROCEED),  # exactly at route
        (0.81, PolicyAction.CLARIFY),
        (0.45, PolicyAction.CLARIFY),  # exactly at clarify_floor
        (0.44, PolicyAction.ESCALATE),
        (0.10, PolicyAction.ESCALATE),
    ],
)
def test_confidence_bands(
    session: SessionState, ctx: PolicyContext, value: float, expected: PolicyAction
) -> None:
    session.current_intent = predict(value)
    assert confidence.evaluate(session, ctx).action is expected


def test_a_second_failed_clarification_escalates(session: SessionState, ctx: PolicyContext) -> None:
    """A caller asked twice has already decided the bot cannot help."""
    session.current_intent = predict(0.70)
    session.clarification_count = 2
    decision = confidence.evaluate(session, ctx)
    assert decision.action is PolicyAction.ESCALATE
    assert decision.reason is HandoffReason.LOW_CONFIDENCE


def test_a_very_weak_guess_escalates_rather_than_clarifying(
    session: SessionState, ctx: PolicyContext
) -> None:
    session.current_intent = predict(0.05)
    assert confidence.evaluate(session, ctx).reason is HandoffReason.UNSUPPORTED_INTENT


def test_an_intent_specific_threshold_overrides_the_pack(
    session: SessionState, ctx: PolicyContext
) -> None:
    """A high-stakes intent may demand more certainty than the pack default."""
    session.current_intent = predict(0.85)
    assert confidence.evaluate(session, ctx).action is PolicyAction.PROCEED
    strict = replace(ctx, escalation=EscalationPolicy(min_intent_confidence=0.95))
    assert confidence.evaluate(session, strict).action is PolicyAction.CLARIFY


def test_no_prediction_yet_abstains(session: SessionState, ctx: PolicyContext) -> None:
    assert confidence.evaluate(session, ctx).action is PolicyAction.PROCEED


def test_an_unresolved_prediction_does_not_proceed(
    session: SessionState, ctx: PolicyContext
) -> None:
    session.current_intent = predict(0.99, intent=None)
    assert confidence.evaluate(session, ctx).action is not PolicyAction.PROCEED


# ------------------------------------------------------------------- sentiment


def test_no_readings_abstains(session: SessionState, ctx: PolicyContext) -> None:
    assert sentiment.evaluate(session, ctx).action is PolicyAction.PROCEED


def test_crossing_the_threshold_escalates(session: SessionState, ctx: PolicyContext) -> None:
    session.sentiment_trail = frustration(0.7)
    decision = sentiment.evaluate(session, ctx)
    assert decision.action is PolicyAction.ESCALATE
    assert decision.reason is HandoffReason.NEGATIVE_SENTIMENT
    assert decision.urgency is Urgency.HIGH


def test_a_calm_caller_proceeds(session: SessionState, ctx: PolicyContext) -> None:
    session.sentiment_trail = frustration(0.1, 0.1, 0.1)
    assert sentiment.evaluate(session, ctx).action is PolicyAction.PROCEED


def test_a_rising_trend_escalates_before_the_threshold(
    session: SessionState, ctx: PolicyContext
) -> None:
    """Waiting for a fixed line costs a turn the caller did not want to spend."""
    session.sentiment_trail = frustration(0.36, 0.42, 0.55)
    decision = sentiment.evaluate(session, ctx)
    assert decision.action is PolicyAction.ESCALATE
    assert decision.urgency is Urgency.NORMAL


def test_a_rising_but_still_low_trend_proceeds(session: SessionState, ctx: PolicyContext) -> None:
    """Rising from nothing is a conversation warming up, not a caller losing patience."""
    session.sentiment_trail = frustration(0.05, 0.10, 0.20)
    assert sentiment.evaluate(session, ctx).action is PolicyAction.PROCEED


def test_two_readings_are_not_a_trend(session: SessionState, ctx: PolicyContext) -> None:
    session.sentiment_trail = frustration(0.40, 0.60)
    assert sentiment.evaluate(session, ctx).action is PolicyAction.PROCEED


def test_a_dip_breaks_the_trend(session: SessionState, ctx: PolicyContext) -> None:
    session.sentiment_trail = frustration(0.50, 0.40, 0.60)
    assert sentiment.evaluate(session, ctx).action is PolicyAction.PROCEED


# ----------------------------------------------------------------------- retry


def test_the_turn_ceiling_escalates(session: SessionState, ctx: PolicyContext) -> None:
    session.turn_index = ctx.max_turns
    decision = retry.evaluate(session, ctx)
    assert decision.action is PolicyAction.ESCALATE
    assert decision.reason is HandoffReason.MAX_TURNS


def test_repeated_tool_failures_escalate(session: SessionState, ctx: PolicyContext) -> None:
    session.tool_failure_count = 2
    decision = retry.evaluate(session, ctx)
    assert decision.reason is HandoffReason.TOOL_FAILURE
    assert decision.urgency is Urgency.HIGH


def test_silence_escalates(session: SessionState, ctx: PolicyContext) -> None:
    session.no_input_count = 3
    assert retry.evaluate(session, ctx).action is PolicyAction.ESCALATE


def test_repeated_no_match_escalates(session: SessionState, ctx: PolicyContext) -> None:
    session.no_match_count = 3
    assert retry.evaluate(session, ctx).reason is HandoffReason.UNSUPPORTED_INTENT


def test_a_fresh_session_proceeds(session: SessionState, ctx: PolicyContext) -> None:
    assert retry.evaluate(session, ctx).action is PolicyAction.PROCEED


# ---------------------------------------------------------------------- engine


def test_the_first_decisive_verdict_wins(session: SessionState, ctx: PolicyContext) -> None:
    """Risk outranks everything: a regulated intent is not clarified, it is handed over."""
    session.current_intent = predict(0.50)  # would clarify
    session.sentiment_trail = frustration(0.9)  # would escalate on sentiment
    decision = evaluate(session, replace(ctx, risk_tier=RiskTier.REGULATED))
    assert decision.policy == "risk"


def test_sentiment_outranks_confidence(session: SessionState, ctx: PolicyContext) -> None:
    """The worst turn in the system is asking a furious caller to rephrase."""
    session.current_intent = predict(0.50)
    session.sentiment_trail = frustration(0.9)
    assert evaluate(session, ctx).policy == "sentiment"


def test_exhaustion_outranks_sentiment(session: SessionState, ctx: PolicyContext) -> None:
    session.turn_index = ctx.max_turns
    session.sentiment_trail = frustration(0.9)
    assert evaluate(session, ctx).policy == "retry"


def test_a_healthy_session_proceeds(session: SessionState, ctx: PolicyContext) -> None:
    session.current_intent = predict(0.95)
    session.sentiment_trail = frustration(0.1)
    assert evaluate(session, ctx).action is PolicyAction.PROCEED


def test_the_engine_reports_its_order() -> None:
    assert PolicyEngine().names == ("risk", "retry", "sentiment", "confidence")


def test_every_default_policy_is_pure(session: SessionState, ctx: PolicyContext) -> None:
    """A policy that mutated state would make the order unreviewable."""
    before = session.model_dump_json()
    for policy in DEFAULT_POLICIES:
        policy(session, ctx)
    assert session.model_dump_json() == before


def test_evaluate_all_returns_every_verdict(session: SessionState, ctx: PolicyContext) -> None:
    verdicts = PolicyEngine().evaluate_all(session, ctx)
    assert len(verdicts) == len(DEFAULT_POLICIES)


def test_a_decision_converts_to_an_escalation(session: SessionState, ctx: PolicyContext) -> None:
    session.turn_index = ctx.max_turns
    decision = evaluate(session, ctx)
    escalation = decision.to_escalation(at_turn=session.turn_index)
    assert escalation.triggered_by == "retry"
    assert escalation.reason is HandoffReason.MAX_TURNS


def test_a_non_escalating_decision_cannot_become_one() -> None:
    with pytest.raises(ValueError, match="did not decide to escalate"):
        PolicyDecision(action=PolicyAction.CLARIFY, policy="confidence").to_escalation(1)


def test_context_is_built_from_pack_and_node(retail_pack) -> None:
    from ccas.schemas.taxonomy import AutomationScore, IntentNode, VolumeStats

    node = IntentNode(
        intent_id="billing",
        level=1,
        label="B",
        description="d",
        risk_tier=RiskTier.HIGH,
        escalation=EscalationPolicy(target_queue="billing"),
        volume=VolumeStats(utterance_count=1, call_count=1, share_of_total=0.1),
        automation=AutomationScore(
            feasibility=0.5, complexity=0.5, confidence=0.5, rationale="r", volume_share=0.1
        ),
    )
    ctx = PolicyContext.for_intent(retail_pack, node)
    assert ctx.risk_tier is RiskTier.HIGH
    assert ctx.default_queue == "billing"
    assert ctx.max_turns == retail_pack.max_turns


def test_context_falls_back_to_pack_defaults(retail_pack) -> None:
    ctx = PolicyContext.for_intent(retail_pack, None)
    assert ctx.risk_tier is RiskTier.LOW
    assert ctx.default_queue == retail_pack.default_queue.name
