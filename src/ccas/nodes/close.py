"""Close: end the session with a disposition.

Every terminal path writes a ``SessionOutcome``. A call that ends without one is a call
nobody can report on, which is how the QA gap in the systems this replaces got so wide.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn, speak
from ccas.schemas.session import SessionOutcome, SessionState

__all__ = ["CLOSING_MESSAGE", "NAME", "make_node"]

NAME = "close"

CLOSING_MESSAGE = "Glad I could help. Thanks for calling."


def make_node(ctx: GraphContext) -> NodeFn:
    async def close(state: SessionState) -> dict[str, Any]:
        already_spoken = state.terminal
        update: dict[str, Any] = {
            "outcome": state.outcome
            or SessionOutcome(
                status="contained",
                resolved_intent=(state.current_intent.intent_id if state.current_intent else None),
                turn_count=state.turn_index,
                duration_ms=0,
                disposition_code="resolved",
            ),
            "terminal": True,
        }
        if not already_spoken:
            update.update(speak(ctx, state, CLOSING_MESSAGE))
        return update

    return close
