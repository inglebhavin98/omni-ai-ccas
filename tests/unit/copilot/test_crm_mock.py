"""The mock CRM adapter -- the last hop before caller data leaves for a vendor.

`docs/future-scoped-work.md` 7.1 has claimed since Phase 4 that this file "proves the
contract". It did not exist. These are the properties it has to prove.
"""

from __future__ import annotations

import pytest

from ccas.copilot.crm.mock import MockCrmAdapter
from ccas.schemas import (
    HandoffContext,
    HandoffReason,
    RedactionStatus,
    TraceContext,
    Urgency,
)
from tests.factories import redacted


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
        "intent_path": ("billing", "billing.refund"),
        "cti_attributes": {"reason": redacted("low_confidence")},
    }
    base.update(kw)
    return HandoffContext(**base)  # type: ignore[arg-type]


async def test_a_push_returns_a_record_the_agent_can_be_pointed_at(trace: TraceContext) -> None:
    record = await MockCrmAdapter().push(handoff(trace))
    assert record.handoff_id == "h-1"
    assert record.system == "mock"
    assert record.record_id


async def test_attached_data_leaves_as_plain_strings(trace: TraceContext) -> None:
    """CTI attached data is a flat key/value map on the wire. The conversion out of
    ``RedactedText`` is the boundary, so it goes through ``require_egress`` rather than
    ``.text`` -- that is the whole reason the field is typed."""
    record = await MockCrmAdapter().push(handoff(trace))
    assert record.attributes["reason"] == "low_confidence"
    assert all(isinstance(v, str) for v in record.attributes.values())


async def test_the_routing_facts_reach_the_desktop(trace: TraceContext) -> None:
    """A desktop that cannot see the queue or the intent path has to ask the caller to
    repeat themselves, which is the thing a handoff exists to prevent."""
    record = await MockCrmAdapter().push(handoff(trace))
    assert record.attributes["target_queue"] == "tier-1"
    assert record.attributes["intent_path"] == "billing > billing.refund"
    assert record.attributes["reason"] == "low_confidence"


async def test_pushing_the_same_handoff_twice_does_not_open_two_cases(
    trace: TraceContext,
) -> None:
    """A retry after a timeout is the expected case, not the exotic one."""
    adapter = MockCrmAdapter()
    ctx = handoff(trace)
    first = await adapter.push(ctx)
    second = await adapter.push(ctx)
    assert first.record_id == second.record_id
    assert len(adapter.records) == 1


async def test_a_summary_that_lost_its_clearance_is_refused(trace: TraceContext) -> None:
    """`HandoffContext` cannot be built unredacted, so this can only happen if someone
    constructs one another way. The adapter still refuses rather than trusting its input:
    it is the last code that runs before the data is somebody else's."""
    ctx = handoff(trace)
    # model_copy(update=...) skips validation, which is the only way to get an unclean
    # payload past the constructor -- i.e. exactly the bypass the adapter must survive.
    leaked = ctx.model_copy(
        update={"summary": redacted("call me on 555-0100", RedactionStatus.DIRTY)}
    )
    with pytest.raises(PermissionError, match="egress blocked"):
        await MockCrmAdapter().push(leaked)


async def test_the_transcript_does_not_travel_in_attached_data(trace: TraceContext) -> None:
    """Attached data is retained by the vendor indefinitely. The transcript belongs in the
    handoff record the agent opens, not stapled to every CTI event."""
    record = await MockCrmAdapter().push(handoff(trace))
    assert not any("summary" in key or "transcript" in key for key in record.attributes)
