"""An in-memory CRM, so the handoff path can be exercised without a vendor account.

This is the adapter `docs/future-scoped-work.md` 7.1 means when it says the contract is
proven without a real integration: it holds the same ``CrmAdapter`` shape a Salesforce or
ServiceNow adapter would, so the thing under test is the *boundary*, not the vendor.

It is not a stub. It applies the same egress gate, is idempotent on ``handoff_id`` the way
the protocol requires, and keeps what it was given so a test or the demo console can
assert on what would have left the process.
"""

from __future__ import annotations

from ccas.copilot.crm.base import CrmRecord, attached_data
from ccas.observability.logging import get_logger
from ccas.schemas.common import Slug
from ccas.schemas.handoff import HandoffContext

__all__ = ["MockCrmAdapter"]

LOG = get_logger("copilot.crm.mock")


class MockCrmAdapter:
    """Records pushes in memory. Never leaves the process, by construction."""

    name: Slug = "mock"

    def __init__(self, system: Slug = "mock") -> None:
        self.name = system
        self._records: dict[str, CrmRecord] = {}

    @property
    def records(self) -> tuple[CrmRecord, ...]:
        """Insertion-ordered, for assertions and the demo console."""
        return tuple(self._records.values())

    async def push(self, handoff: HandoffContext) -> CrmRecord:
        existing = self._records.get(handoff.handoff_id)
        if existing is not None:
            # Idempotent, and quietly so: a retry is not an anomaly worth a warning.
            return existing

        # Before the record exists, so a refusal leaves nothing half-created.
        data = attached_data(handoff)

        record = CrmRecord(
            system=self.name,
            record_id=f"{self.name}-{handoff.handoff_id}",
            handoff_id=handoff.handoff_id,
            attributes=data,
            url=f"https://crm.invalid/cases/{handoff.handoff_id}",
        )
        self._records[handoff.handoff_id] = record

        LOG.info(
            "copilot.crm.push",
            correlation_id=handoff.trace.correlation_id,
            system=self.name,
            handoff_id=handoff.handoff_id,
            reason=handoff.reason.value,
            target_queue=handoff.target_queue,
            # Counts and shapes only -- never a value (Rule 11).
            attribute_count=len(data),
            transcript_turns=len(handoff.transcript),
            tool_calls=len(handoff.tool_trace),
            verified=handoff.identity is not None,
        )
        return record
