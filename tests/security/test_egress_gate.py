"""CLAUDE.md Rule 2: no unredacted text can reach an external boundary.

These tests assert the *type-level* gate rather than any engine behaviour -- the point
is that an unredacted egress payload must be impossible to construct at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    CallLog,
    Channel,
    DatasetSource,
    HandoffContext,
    HandoffReason,
    LLMRequest,
    Message,
    ModelBinding,
    ProviderName,
    RedactionStatus,
    Speaker,
    TraceContext,
    Urgency,
    Utterance,
)
from tests.factories import make_report, redacted

NOT_CLEAN = [RedactionStatus.DIRTY, RedactionStatus.UNVERIFIED, RedactionStatus.BYPASSED]

pytestmark = pytest.mark.security


@pytest.mark.parametrize("status", NOT_CLEAN)
def test_call_log_cannot_hold_unredacted_text(status: RedactionStatus) -> None:
    with pytest.raises(ValidationError):
        CallLog(
            call_id=CallLog.make_call_id(DatasetSource.SYNTHETIC, "r"),
            source=DatasetSource.SYNTHETIC,
            source_record_id="r",
            channel=Channel.VOICE,
            utterances=(Utterance(index=0, speaker=Speaker.CALLER, content=redacted("x", status)),),
            redaction=make_report(),
        )


@pytest.mark.parametrize("status", NOT_CLEAN)
def test_handoff_cannot_hold_unredacted_text(status: RedactionStatus, trace: TraceContext) -> None:
    with pytest.raises(ValidationError):
        HandoffContext(
            handoff_id="h",
            session_id="s",
            trace=trace,
            domain="retail",
            reason=HandoffReason.POLICY,
            urgency=Urgency.NORMAL,
            target_queue="tier-1",
            intent_confidence=0.5,
            summary=redacted("x", status),
        )


@pytest.mark.parametrize("status", NOT_CLEAN)
def test_llm_request_cannot_hold_unredacted_text(status: RedactionStatus) -> None:
    with pytest.raises(ValidationError):
        LLMRequest(
            binding=ModelBinding(node="router", provider=ProviderName.VLLM, model="llama-3.3-70b"),
            messages=(Message(role="user", content=redacted("x", status)),),
        )


@pytest.mark.parametrize("status", NOT_CLEAN)
def test_require_egress_refuses_every_non_clean_status(status: RedactionStatus) -> None:
    with pytest.raises(PermissionError):
        redacted("x", status).require_egress()


def test_an_unavailable_engine_fails_closed() -> None:
    """Never fail open: a missing ONNX model must block egress, not wave it through."""
    report = redacted("x", RedactionStatus.UNVERIFIED).report
    assert not report.egress_permitted


def test_reports_never_carry_matched_text() -> None:
    """A report must be safe to log; the thing it describes is not."""
    from ccas.schemas import RedactionReport

    fields = set(RedactionReport.model_fields)
    assert "text" not in fields
    assert "matched" not in fields
    assert "original" not in fields
