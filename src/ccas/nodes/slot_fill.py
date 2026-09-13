"""Slot filling: collect what the intent needs before any tool runs.

Slots are declared by the taxonomy, not by this module, so a new vertical adds slots as
data. One slot per turn: asking for three things at once produces an answer to one of
them and a caller who has to be asked again.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn, latest_caller_text, speak
from ccas.schemas.session import SessionState, SlotValue
from ccas.schemas.taxonomy import SlotSpec

__all__ = ["NAME", "make_node", "missing_slots", "next_slot"]

NAME = "slot_fill"


def missing_slots(ctx: GraphContext, state: SessionState) -> tuple[SlotSpec, ...]:
    node = ctx.node_for(state)
    if node is None:
        return ()
    outstanding = state.missing_required_slots(tuple(s.name for s in node.required_slots))
    return tuple(s for s in node.required_slots if s.name in outstanding)


def next_slot(ctx: GraphContext, state: SessionState) -> SlotSpec | None:
    pending = missing_slots(ctx, state)
    return pending[0] if pending else None


def make_node(ctx: GraphContext) -> NodeFn:
    async def slot_fill(state: SessionState) -> dict[str, Any]:
        captured = _capture_pending(ctx, state)

        # Re-read what is outstanding *after* capturing, so a filled slot is not asked
        # for again in the same turn.
        working = state.model_copy(update={"slots": {**state.slots, **captured}})
        spec = next_slot(ctx, working)
        if spec is None:
            return {"slots": working.slots, "pending_slot": None}

        attempts = state.slots[spec.name].attempts if spec.name in state.slots else 0
        if attempts >= spec.max_attempts:
            # Give up on this slot rather than loop. The retry policy escalates.
            return {
                "slots": working.slots,
                "pending_slot": None,
                "no_match_count": state.no_match_count + 1,
            }

        prompt = spec.reprompt if attempts and spec.reprompt else spec.elicitation_prompt
        return {
            **speak(ctx, state, prompt),
            "slots": working.slots,
            "pending_slot": spec.name,
        }

    return slot_fill


def _capture_pending(ctx: GraphContext, state: SessionState) -> dict[str, SlotValue]:
    """Treat this turn's caller utterance as the answer to the outstanding question."""
    if state.pending_slot is None:
        return {}
    node = ctx.node_for(state)
    spec = next((s for s in (node.slots if node else ()) if s.name == state.pending_slot), None)
    answer = latest_caller_text(state)
    if spec is None or answer is None:
        return {}

    previous = state.slots.get(spec.name)
    valid = _valid(spec, answer.text)
    return {
        spec.name: SlotValue(
            name=spec.name,
            raw=answer,
            parsed=answer.text if valid else None,
            valid=valid,
            attempts=(previous.attempts + 1) if previous else 1,
            captured_via="speech",
        )
    }


def _valid(spec: SlotSpec, text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if spec.enum_values is not None:
        return value.casefold() in {v.casefold() for v in spec.enum_values}
    if spec.validation_regex is not None:
        import re

        return re.search(spec.validation_regex, value) is not None
    # A redacted placeholder means the caller *did* supply the identifier; the value is
    # gone but the fact that it was given is what the slot records.
    return True
