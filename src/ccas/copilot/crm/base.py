"""The CRM adapter boundary -- where a ``HandoffContext`` stops being ours.

Everything upstream of here is protected by types: ``RedactedText`` cannot be built
without a report, and ``HandoffContext`` refuses construction if any text-bearing field
is not egress-permitted. None of that survives the wire. A CRM takes flat strings, so
this is the layer that unwraps them, and unwrapping is exactly where a gate belongs.

Hence ``attached_data``: one function, used by every adapter, that converts through
``require_egress()`` rather than reading ``.text``. The difference is that one raises and
the other does not (Rule 2).

Adapters are ``Protocol``-shaped so a pack can supply its own without this package
importing a vendor SDK. Real adapters are customer-specific and deliberately out of
scope -- `docs/future-scoped-work.md` 7.1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from ccas.schemas.common import Frozen, Slug
from ccas.schemas.handoff import HandoffContext

__all__ = ["CrmAdapter", "CrmError", "CrmRecord", "attached_data"]


class CrmError(RuntimeError):
    """The CRM refused or could not be reached.

    Distinct from a redaction refusal, which is a ``PermissionError`` and is never
    retryable -- retrying a leak just leaks again.
    """


class CrmRecord(Frozen):
    """What the CRM gave back: enough to point a human at the case, and nothing more."""

    system: Slug
    record_id: str = Field(min_length=1, max_length=128)
    handoff_id: str = Field(min_length=1, max_length=128)
    attributes: dict[str, str] = Field(default_factory=dict)
    """Already unwrapped and already cleared. Plain ``str`` because the wire is plain."""

    url: str | None = Field(default=None, max_length=512)


def attached_data(handoff: HandoffContext) -> dict[str, str]:
    """Flatten a handoff into CTI attached data.

    Two rules decide what belongs here. It must be **routing or identification** -- what a
    desktop needs to put the call in front of the right person -- and it must be small,
    because attached data is retained by the vendor for as long as they keep the
    interaction. The summary and the transcript are deliberately absent: they belong in
    the handoff record an agent opens, not stapled to every CTI event.

    Every value crosses via ``require_egress()``, so a payload that lost its clearance
    raises here instead of arriving at a vendor.
    """
    data: dict[str, str] = {
        "handoff_id": handoff.handoff_id,
        "session_id": handoff.session_id,
        "correlation_id": handoff.trace.correlation_id,
        "domain": handoff.domain,
        "reason": handoff.reason.value,
        "urgency": handoff.urgency.value,
        "target_queue": handoff.target_queue,
        "intent_confidence": f"{handoff.intent_confidence:.2f}",
    }
    if handoff.intent_path:
        data["intent_path"] = " > ".join(handoff.intent_path)
    if handoff.required_skills:
        data["required_skills"] = ",".join(handoff.required_skills)
    if handoff.audio_recording_ref:
        data["audio_recording_ref"] = handoff.audio_recording_ref

    # The pack's own attributes last: a pack may add, and may also correct a default.
    for key, value in handoff.cti_attributes.items():
        data[key] = value.require_egress()

    if handoff.identity is not None:
        data["verification_level"] = handoff.identity.level.value
        data["caller_ref"] = handoff.identity.caller_ref
        for key, value in handoff.identity.attributes.items():
            data[f"identity.{key}"] = value.require_egress()

    # Reading the summary proves the payload still holds its clearance, even though the
    # text itself does not travel. A handoff that cannot be summarised must not be pushed.
    handoff.summary.require_egress()
    return data


@runtime_checkable
class CrmAdapter(Protocol):
    """One method, because one is all a handoff needs."""

    name: Slug

    async def push(self, handoff: HandoffContext) -> CrmRecord:
        """Create or update the case for this handoff.

        Must be idempotent on ``handoff.handoff_id``: a retry after a timeout is the
        expected case, and a second case for one call is worse than a failed push.
        """
        ...
