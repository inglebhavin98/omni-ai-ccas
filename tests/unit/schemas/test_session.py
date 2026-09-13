from __future__ import annotations

import typing
from operator import add

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    EscalationDecision,
    HandoffReason,
    IntentPrediction,
    LatencyLedger,
    SentimentSnapshot,
    SessionState,
    SlotValue,
    Speaker,
    Turn,
)
from tests.factories import redacted


def turn(index: int) -> Turn:
    return Turn(index=index, speaker=Speaker.CALLER, content=redacted(f"t{index}"))


@pytest.mark.parametrize("field", ["turns", "intent_history", "sentiment_trail", "tool_records"])
def test_append_only_fields_carry_an_add_reducer(field: str) -> None:
    """LangGraph merges concurrent node writes via these reducers -- without one, a
    parallel branch would clobber the other's appends."""
    annotation = SessionState.model_fields[field].metadata
    assert add in annotation, f"{field} is missing its `add` reducer"


def test_reducer_annotation_is_visible_to_typing() -> None:
    hints = typing.get_type_hints(SessionState, include_extras=True)
    assert typing.get_args(hints["turns"])[1] is add


def test_session_starts_empty(session: SessionState) -> None:
    assert session.turns == []
    assert session.sentiment is None
    assert not session.escalated
    assert not session.terminal


def test_sentiment_returns_the_latest_snapshot(session: SessionState) -> None:
    session.sentiment_trail = [
        SentimentSnapshot(valence=0.1, arousal=0.2, frustration_index=0.1, turn_index=0),
        SentimentSnapshot(valence=-0.6, arousal=0.8, frustration_index=0.9, turn_index=1),
    ]
    latest = session.sentiment
    assert latest is not None
    assert latest.frustration_index == 0.9


def test_missing_required_slots_treats_invalid_as_missing(session: SessionState) -> None:
    session.slots = {
        "a": SlotValue(name="a", raw=redacted("x"), valid=True, captured_via="speech"),
        "b": SlotValue(name="b", raw=redacted("y"), valid=False, captured_via="dtmf"),
    }
    assert session.missing_required_slots(("a", "b", "c")) == ("b", "c")


def test_escalated_flag_follows_the_decision(session: SessionState) -> None:
    session.escalation = EscalationDecision(
        reason=HandoffReason.LOW_CONFIDENCE, triggered_by="confidence", at_turn=3
    )
    assert session.escalated


def test_session_rejects_unknown_fields(session: SessionState) -> None:
    with pytest.raises(ValidationError):
        SessionState(
            session_id="s",
            trace=session.trace,
            domain="retail",
            caller=session.caller,
            surprise=1,  # type: ignore[call-arg]
        )


def test_intent_prediction_reports_resolution() -> None:
    assert IntentPrediction(
        intent_id="billing", confidence=0.9, source="llm_router", latency_ms=40
    ).resolved
    assert not IntentPrediction(confidence=0.1, source="fallback", latency_ms=40).resolved


def test_latency_ledger_sums_every_stage() -> None:
    ledger = LatencyLedger(
        vad_ms=100,
        stt_ms=180,
        redact_us=3000,
        router_ms=90,
        tool_ms=150,
        llm_ttft_ms=180,
        tts_ttfb_ms=120,
    )
    assert ledger.total_rtt_ms == 823
    assert ledger.breached
    assert ledger.headroom_ms == -23


def test_latency_ledger_is_within_budget_when_stages_fit() -> None:
    ledger = LatencyLedger(vad_ms=90, stt_ms=150, router_ms=60, llm_ttft_ms=150, tts_ttfb_ms=100)
    assert ledger.total_rtt_ms == 550
    assert not ledger.breached
    assert ledger.headroom_ms == 250


def test_latency_ledger_exposes_named_stages() -> None:
    assert set(LatencyLedger().stages()) == {
        "vad_ms",
        "stt_ms",
        "redact_ms",
        "router_ms",
        "tool_ms",
        "llm_ttft_ms",
        "tts_ttfb_ms",
    }


def test_ledger_is_mutable_but_validated() -> None:
    ledger = LatencyLedger()
    ledger.stt_ms = 200
    assert ledger.stt_ms == 200
    with pytest.raises(ValidationError):
        ledger.stt_ms = -1


def test_turns_accept_append(session: SessionState) -> None:
    session.turns = [*session.turns, turn(0)]
    assert len(session.turns) == 1
