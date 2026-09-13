"""Grounded response.

Two independent checks, because either alone is insufficient. The model reports whether
it could answer from the evidence -- and a model can assert grounding it lacks.
So the node also checks whether any tool result was actually usable: a turn with no
usable result cannot be grounded, whatever the model says.

When either check fails the node does not speak. It records an escalation and lets the
graph hand over, so the caller hears one handoff message rather than an apology followed
by a handoff (CLAUDE.md Rule 4).
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.graph.responder import Responder
from ccas.nodes.base import NodeFn, latest_caller_text, speak
from ccas.schemas.common import Urgency
from ccas.schemas.escalation import EscalationDecision, HandoffReason
from ccas.schemas.session import SessionState
from ccas.schemas.tools import ToolRecord

__all__ = ["NAME", "make_node", "usable_records"]

NAME = "respond"


def usable_records(state: SessionState) -> tuple[ToolRecord, ...]:
    return tuple(r for r in state.tool_records if r.result.ok and r.result.safe_for_model)


def make_node(ctx: GraphContext, responder: Responder | None) -> NodeFn:
    async def respond(state: SessionState) -> dict[str, Any]:
        utterance = latest_caller_text(state)
        node = ctx.node_for(state)
        expects_tools = bool(node and node.required_tools)

        if responder is None or utterance is None:
            return _hand_over(state, "no responder is configured for this deployment")

        if expects_tools and not usable_records(state):
            return _hand_over(state, "no tool returned usable data for this request")

        intent_id = state.current_intent.intent_id if state.current_intent else None
        reply = await responder.reply(utterance, intent_id, tuple(state.tool_records))

        ledger = state.latency.model_copy(update={"llm_ttft_ms": reply.latency_ms})

        if not reply.grounded or not reply.reply.strip():
            return {
                **_hand_over(
                    state,
                    "the response could not be grounded in tool results"
                    + (f" (missing: {', '.join(reply.missing)})" if reply.missing else ""),
                ),
                "latency": ledger,
            }

        return {**speak(ctx, state, reply.reply), "latency": ledger}

    return respond


def _hand_over(state: SessionState, detail: str) -> dict[str, Any]:
    """Say nothing here; the escalate node owns the one message the caller hears."""
    return {
        "escalation": EscalationDecision(
            reason=HandoffReason.SYSTEM_ERROR,
            triggered_by="grounding",
            at_turn=state.turn_index,
            urgency=Urgency.NORMAL,
            detail=detail,
        )
    }
