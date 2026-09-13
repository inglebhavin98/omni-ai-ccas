from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    HandoffContext,
    HandoffReason,
    RedactionStatus,
    SlotValue,
    Speaker,
    ToolPayload,
    ToolRecord,
    ToolResult,
    ToolStatus,
    TraceContext,
    Turn,
    Urgency,
    VerificationLevel,
    VerifiedIdentity,
    utcnow,
)
from tests.factories import make_report, redacted


def handoff(trace: TraceContext, **kw: object) -> HandoffContext:
    base: dict[str, object] = {
        "handoff_id": "h-1",
        "session_id": "sess-1",
        "trace": trace,
        "domain": "retail",
        "reason": HandoffReason.LOW_CONFIDENCE,
        "urgency": Urgency.NORMAL,
        "target_queue": "tier-1",
        "intent_confidence": 0.4,
        "summary": redacted("Caller asked about [ACCOUNT_REF_1]."),
    }
    base.update(kw)
    return HandoffContext(**base)  # type: ignore[arg-type]


def test_handoff_refuses_an_unredacted_summary(trace: TraceContext) -> None:
    with pytest.raises(ValidationError, match="summary redaction status"):
        handoff(trace, summary=redacted("SSN 123-45-6789", RedactionStatus.UNVERIFIED))


def test_handoff_refuses_an_unredacted_transcript_turn(trace: TraceContext) -> None:
    leaky = Turn(index=0, speaker=Speaker.CALLER, content=redacted("raw", RedactionStatus.DIRTY))
    with pytest.raises(ValidationError, match="transcript turn 0 is unredacted"):
        handoff(trace, transcript=(leaky,))


def test_handoff_refuses_an_unredacted_slot(trace: TraceContext) -> None:
    slot = SlotValue(
        name="reference",
        raw=redacted("4111111111111111", RedactionStatus.DIRTY),
        captured_via="dtmf",
    )
    with pytest.raises(ValidationError, match="slot 'reference' is unredacted"):
        handoff(trace, collected_slots={"reference": slot})


def test_handoff_refuses_an_unredacted_tool_result(trace: TraceContext) -> None:
    record = ToolRecord(
        payload=ToolPayload(
            tool_call_id="tc-1",
            tool_name="get_record_status",
            session_id="sess-1",
            timeout_ms=1000,
            trace=trace,
        ),
        result=ToolResult(
            tool_call_id="tc-1",
            status=ToolStatus.OK,
            elapsed_ms=10,
            redaction=make_report(RedactionStatus.UNVERIFIED),
        ),
    )
    with pytest.raises(ValidationError, match="is unredacted"):
        handoff(trace, tool_trace=(record,))


def test_intent_path_must_descend_l1_to_l3(trace: TraceContext) -> None:
    with pytest.raises(ValidationError, match="must descend"):
        handoff(trace, intent_path=("billing.refund", "billing"))


def test_intent_path_accepts_a_well_formed_descent(trace: TraceContext) -> None:
    ctx = handoff(trace, intent_path=("billing", "billing.refund", "billing.refund.status"))
    assert ctx.leaf_intent == "billing.refund.status"


def test_leaf_intent_is_none_without_a_path(trace: TraceContext) -> None:
    assert handoff(trace).leaf_intent is None


def test_verified_identity_requires_a_real_level() -> None:
    with pytest.raises(ValidationError, match="requires a level above NONE"):
        VerifiedIdentity(
            caller_ref="a" * 32,
            level=VerificationLevel.NONE,
            method="otp",
            verified_at=utcnow(),
        )


def test_verified_identity_accepts_a_verified_caller(trace: TraceContext) -> None:
    identity = VerifiedIdentity(
        caller_ref="a" * 32,
        level=VerificationLevel.STRONG,
        method="otp",
        verified_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
    )
    assert handoff(trace, identity=identity).identity is identity


def test_audio_travels_by_reference_not_inline(trace: TraceContext) -> None:
    ctx = handoff(trace, audio_recording_ref="s3://recordings/sess-1.wav")
    assert ctx.audio_recording_ref is not None
    assert not hasattr(ctx, "audio_bytes")
