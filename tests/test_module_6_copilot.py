"""Module gate: M6a — the copilot handoff.

Granular tests live in ``tests/unit/copilot/``; the escalation journeys that produce a
handoff in the first place are in ``tests/integration/test_graph_e2e.py``. This asserts
the module is usable as a whole and that the guarantee it exists for actually holds:
**nothing reaches a vendor that has not cleared redaction.**

M6b (judge, Ragas/DeepEval) is a separate half and is not covered here. Its contracts
already exist in ``schemas/eval.py``; its runtime does not.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.copilot.crm.base import CrmAdapter, CrmRecord, attached_data
from ccas.copilot.crm.mock import MockCrmAdapter
from ccas.schemas import (
    HandoffContext,
    HandoffReason,
    RedactionStatus,
    Speaker,
    TraceContext,
    Turn,
    Urgency,
    VerificationLevel,
    VerifiedIdentity,
    utcnow,
)
from tests.factories import redacted


def handoff(trace: TraceContext, **kw: object) -> HandoffContext:
    base: dict[str, object] = {
        "handoff_id": "h-1",
        "session_id": "sess-1",
        "trace": trace,
        "domain": "retail",
        "reason": HandoffReason.TOOL_FAILURE,
        "urgency": Urgency.HIGH,
        "target_queue": "billing",
        "required_skills": ("billing", "tier-2"),
        "intent_confidence": 0.91,
        "intent_path": ("billing", "billing.refund"),
        "summary": redacted("Caller could not complete a return on [ORDER_REF_1]."),
        "transcript": (
            Turn(index=0, speaker=Speaker.CALLER, content=redacted("my card [PAYMENT_CARD_1]")),
        ),
        "cti_attributes": {"triggered_by": redacted("tool_failure")},
    }
    base.update(kw)
    return HandoffContext(**base)  # type: ignore[arg-type]


def test_the_mock_adapter_satisfies_the_adapter_protocol() -> None:
    """A pack may bring its own adapter, so the protocol is the real contract."""
    assert isinstance(MockCrmAdapter(), CrmAdapter)


async def test_a_handoff_reaches_the_crm_as_a_record(trace: TraceContext) -> None:
    record = await MockCrmAdapter().push(handoff(trace))
    assert isinstance(record, CrmRecord)
    assert record.handoff_id == "h-1"
    assert record.url is not None


async def test_the_desktop_gets_what_it_needs_to_route(trace: TraceContext) -> None:
    """The point of a handoff is that the caller does not repeat themselves."""
    data = (await MockCrmAdapter().push(handoff(trace))).attributes
    assert data["target_queue"] == "billing"
    assert data["required_skills"] == "billing,tier-2"
    assert data["intent_path"] == "billing > billing.refund"
    assert data["reason"] == "tool_failure"
    assert data["urgency"] == "high"
    assert data["correlation_id"] == trace.correlation_id


def test_attached_data_is_flat_strings_only(trace: TraceContext) -> None:
    """CTI attached data is a flat map on the wire. Anything else silently truncates or
    stringifies into whatever repr the vendor's client happens to produce."""
    for key, value in attached_data(handoff(trace)).items():
        assert isinstance(key, str) and isinstance(value, str)


def test_the_transcript_is_not_stapled_to_every_cti_event(trace: TraceContext) -> None:
    """Attached data is retained by the vendor for the life of the interaction."""
    data = attached_data(handoff(trace))
    assert not any("my card" in value for value in data.values())
    assert "summary" not in data


# --------------------------------------------------------------- the actual guarantee


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", redacted("call me on 555-0100", RedactionStatus.DIRTY)),
        ("summary", redacted("engine down", RedactionStatus.UNVERIFIED)),
        ("summary", redacted("test override", RedactionStatus.BYPASSED)),
    ],
)
def test_an_unclean_payload_cannot_be_constructed(
    trace: TraceContext, field: str, value: object
) -> None:
    """Every non-CLEAN status is forbidden, not just the obviously broken one.
    UNVERIFIED matters most: the engine being unavailable must never fail open."""
    with pytest.raises(ValidationError):
        handoff(trace, **{field: value})


async def test_an_unclean_payload_is_refused_at_the_wire_too(trace: TraceContext) -> None:
    """Belt and braces. The constructor is the gate, but the adapter is the last code
    that runs while the data is still ours, so it does not trust its input."""
    leaked = handoff(trace).model_copy(
        update={"summary": redacted("555-0100", RedactionStatus.UNVERIFIED)}
    )
    with pytest.raises(PermissionError, match="egress blocked"):
        await MockCrmAdapter().push(leaked)


async def test_a_verified_identity_travels_hashed_and_redacted(trace: TraceContext) -> None:
    """caller_ref is a hash a human resolves against the CRM, not an identifier, and the
    verified attributes are caller-derived so they carry reports like any other text."""
    identity = VerifiedIdentity(
        caller_ref="b" * 32,
        level=VerificationLevel.STRONG,
        method="otp",
        attributes={"postcode": redacted("[POSTCODE_1]")},
        verified_at=utcnow(),
    )
    data = (await MockCrmAdapter().push(handoff(trace, identity=identity))).attributes
    assert data["caller_ref"] == "b" * 32
    assert data["verification_level"] == "strong"
    assert data["identity.postcode"] == "[POSTCODE_1]"


async def test_a_retry_does_not_open_a_second_case(trace: TraceContext) -> None:
    adapter = MockCrmAdapter()
    ctx = handoff(trace)
    assert (await adapter.push(ctx)).record_id == (await adapter.push(ctx)).record_id
    assert len(adapter.records) == 1


async def test_audio_travels_as_a_reference_never_inline(trace: TraceContext) -> None:
    data = attached_data(handoff(trace, audio_recording_ref="s3://recordings/sess-1.wav"))
    assert data["audio_recording_ref"].startswith("s3://")
