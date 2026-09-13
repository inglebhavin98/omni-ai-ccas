"""Tool execution.

Builds one payload per tool the intent declares and runs them together. All guarantees
live in ``ToolExecutor``; this node's job is to decide *whether* to call, to fold the
results into state, and to notice when a refusal is actually a request for a stronger
verification level rather than a failure.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn
from ccas.schemas.common import JsonValue
from ccas.schemas.session import SessionState
from ccas.schemas.tools import ToolRecord, ToolStatus

__all__ = ["NAME", "make_node", "needs_step_up"]

NAME = "tool_exec"


def needs_step_up(records: tuple[ToolRecord, ...]) -> bool:
    """A DENIED result is recoverable: ask for a stronger factor and try again."""
    return any(
        r.result.status is ToolStatus.DENIED and r.result.error_code == "not_authorized"
        for r in records
    )


def make_node(ctx: GraphContext) -> NodeFn:
    async def tool_exec(state: SessionState) -> dict[str, Any]:
        node = ctx.node_for(state)
        if node is None or not node.required_tools:
            return {}

        payloads = [
            ctx.executor.build_payload(
                spec.name,
                session_id=state.session_id,
                trace=state.trace,
                arguments=_arguments_for(spec.name, state),
                intent_id=node.intent_id,
                verification=state.caller.verification,
                call_index=state.turn_index * 10 + index,
            )
            for index, spec in enumerate(ctx.registry.for_intent(node))
        ]
        records = await ctx.executor.execute_many(payloads)
        failures = sum(1 for r in records if not r.result.ok)
        spent_ms = sum(r.result.elapsed_ms for r in records)

        return {
            "tool_records": list(records),
            "tool_failure_count": state.tool_failure_count + failures,
            "latency": state.latency.model_copy(
                update={"tool_ms": state.latency.tool_ms + spent_ms}
            ),
        }

    return tool_exec


def _arguments_for(tool_name: str, state: SessionState) -> dict[str, JsonValue]:
    """Map captured slots onto a tool's arguments by name.

    Deliberately literal. A model constructing arguments freehand is the failure mode
    Rule 4 exists to prevent -- it chooses *which* tool, never what to send it.
    """
    return {
        name: slot.parsed if slot.parsed is not None else slot.raw.text
        for name, slot in state.slots.items()
        if slot.valid
    }
