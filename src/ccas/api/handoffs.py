"""Retention for ``HandoffContext`` payloads (the agent-desktop surface).

The escalate node builds a handoff and returns; nothing kept a reference, so there was
nothing for ``GET /v1/handoffs/{id}`` to serve. This store keeps the most recent
handoffs of this process in memory -- bounded, because a workbench that grows without
limit is a leak with a UI.

The payloads themselves are already egress-clean by construction: ``HandoffContext``
refuses to build around anything unredacted, so storing one cannot create a leak. It can
create *retention*, which is why the store is bounded and lives only in the session
manager's process, like the sessions themselves.
"""

from __future__ import annotations

from ccas.schemas.handoff import HandoffContext

__all__ = ["HandoffStore", "handoff_id_for"]


def handoff_id_for(session_id: str) -> str:
    """The deterministic id the escalate node assigns. The desktop can derive it, but
    serving it explicitly beats making a client guess a naming convention."""
    return f"{session_id}-handoff"


class HandoffStore:
    """``handoff_id -> HandoffContext`` for this process, oldest dropped first."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("a store that holds nothing holds nothing")
        self._capacity = capacity
        self._items: dict[str, HandoffContext] = {}

    def put(self, handoff: HandoffContext) -> None:
        if len(self._items) >= self._capacity and handoff.handoff_id not in self._items:
            oldest = next(iter(self._items))
            del self._items[oldest]
        self._items[handoff.handoff_id] = handoff

    def get(self, handoff_id: str) -> HandoffContext | None:
        return self._items.get(handoff_id)

    def __len__(self) -> int:
        return len(self._items)
