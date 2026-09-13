"""Graph assembly.

One invocation handles one caller turn. The graph ends when it has said something and is
waiting for a reply, or when the call is over -- which is what makes the turn count, the
clarification ceiling and the latency ledger meaningful.

Every edge is a pure function of state. No node decides its own successor, so "can this
reach a human?" and "can this loop?" are answerable by reading the routing table rather
than by sampling behaviour (CLAUDE.md Rule 4).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ccas.graph.context import GraphContext
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.nodes import clarify, close, escalate, greet, identify, respond, route, slot_fill
from ccas.nodes import tool_exec as tool_exec_node
from ccas.nodes.base import NodeFn, latest_caller_text
from ccas.policies.base import PolicyAction
from ccas.schemas.session import SessionState

__all__ = ["NODE_NAMES", "build_graph", "select_entry"]

NODE_NAMES: tuple[str, ...] = (
    greet.NAME,
    identify.NAME,
    route.NAME,
    clarify.NAME,
    slot_fill.NAME,
    tool_exec_node.NAME,
    respond.NAME,
    escalate.NAME,
    close.NAME,
)


def select_entry(state: SessionState) -> str:
    """Where this turn begins."""
    if state.terminal:
        return close.NAME
    if state.turn_index == 0:
        return greet.NAME
    if state.pending_slot:
        # The caller is answering a question we asked; capture it before re-routing.
        return slot_fill.NAME
    return route.NAME


def _after_greet(state: SessionState) -> str:
    # A caller transferred mid-conversation may already have spoken. One who has not
    # gets the greeting and a chance to answer, rather than "sorry, I didn't catch that".
    return identify.NAME if latest_caller_text(state) is not None else END


def _make_after_identify(ctx: GraphContext) -> Any:
    def after_identify(state: SessionState) -> str:
        needed = identify.required_level(ctx, state)
        if state.caller.verification.satisfies(needed):
            return route.NAME
        # We just asked for a factor. Ending here is what breaks the
        # tool_exec -> identify -> route -> tool_exec cycle.
        return END

    return after_identify


def _make_after_route(ctx: GraphContext) -> Any:
    def after_route(state: SessionState) -> str:
        verdict = ctx.policies.evaluate(state, ctx.policy_context(state))
        if verdict.action is PolicyAction.ESCALATE:
            return escalate.NAME
        if verdict.action is PolicyAction.CLARIFY:
            return clarify.NAME

        node = ctx.node_for(state)
        if node is None:
            return escalate.NAME
        if slot_fill.next_slot(ctx, state) is not None:
            return slot_fill.NAME
        return tool_exec_node.NAME if node.required_tools else respond.NAME

    return after_route


def _make_after_slot_fill(ctx: GraphContext) -> Any:
    def after_slot_fill(state: SessionState) -> str:
        if state.pending_slot:
            return END  # asked for a value; the caller answers next turn
        verdict = ctx.policies.evaluate(state, ctx.policy_context(state))
        if verdict.action is PolicyAction.ESCALATE:
            return escalate.NAME
        node = ctx.node_for(state)
        if node is None:
            return escalate.NAME
        return tool_exec_node.NAME if node.required_tools else respond.NAME

    return after_slot_fill


def _make_after_tool_exec(ctx: GraphContext) -> Any:
    def after_tool_exec(state: SessionState) -> str:
        if tool_exec_node.needs_step_up(tuple(state.tool_records)):
            # A refusal for want of verification is recoverable, not a failure.
            return identify.NAME
        verdict = ctx.policies.evaluate(state, ctx.policy_context(state))
        return escalate.NAME if verdict.action is PolicyAction.ESCALATE else respond.NAME

    return after_tool_exec


def _add_node(graph: StateGraph[Any, Any, Any, Any], name: str, node: NodeFn) -> None:
    """Register a node.

    LangGraph's ``_Node`` protocol does not admit an async callable returning a
    partial-update dict, although that is the documented node contract and what every
    node here is. One suppression in a shim beats nine at the call sites.
    """
    graph.add_node(name, node)  # type: ignore[call-overload]


def _after_respond(state: SessionState) -> str:
    # respond stays silent when it could not ground an answer, so the caller hears one
    # handoff message instead of an apology followed by a handoff.
    return escalate.NAME if state.escalation is not None else END


def build_graph(
    ctx: GraphContext,
    router: IntentRouter | None = None,
    responder: Responder | None = None,
    checkpointer: Any = None,
) -> Any:
    """Compile the mesh. ``router``/``responder`` are optional so the graph is
    constructible -- and its shape testable -- with no provider configured."""
    graph = StateGraph(SessionState)

    _add_node(graph, greet.NAME, greet.make_node(ctx))
    _add_node(graph, identify.NAME, identify.make_node(ctx))
    _add_node(graph, route.NAME, route.make_node(ctx, router))
    _add_node(graph, clarify.NAME, clarify.make_node(ctx))
    _add_node(graph, slot_fill.NAME, slot_fill.make_node(ctx))
    _add_node(graph, tool_exec_node.NAME, tool_exec_node.make_node(ctx))
    _add_node(graph, respond.NAME, respond.make_node(ctx, responder))
    _add_node(graph, escalate.NAME, escalate.make_node(ctx))
    _add_node(graph, close.NAME, close.make_node(ctx))

    graph.add_conditional_edges(
        START,
        select_entry,
        {
            close.NAME: close.NAME,
            greet.NAME: greet.NAME,
            slot_fill.NAME: slot_fill.NAME,
            route.NAME: route.NAME,
        },
    )
    graph.add_conditional_edges(greet.NAME, _after_greet, {identify.NAME: identify.NAME, END: END})
    graph.add_conditional_edges(
        identify.NAME, _make_after_identify(ctx), {route.NAME: route.NAME, END: END}
    )
    graph.add_conditional_edges(
        route.NAME,
        _make_after_route(ctx),
        {
            escalate.NAME: escalate.NAME,
            clarify.NAME: clarify.NAME,
            slot_fill.NAME: slot_fill.NAME,
            tool_exec_node.NAME: tool_exec_node.NAME,
            respond.NAME: respond.NAME,
        },
    )
    graph.add_conditional_edges(
        slot_fill.NAME,
        _make_after_slot_fill(ctx),
        {
            END: END,
            escalate.NAME: escalate.NAME,
            tool_exec_node.NAME: tool_exec_node.NAME,
            respond.NAME: respond.NAME,
        },
    )
    graph.add_conditional_edges(
        tool_exec_node.NAME,
        _make_after_tool_exec(ctx),
        {
            identify.NAME: identify.NAME,
            escalate.NAME: escalate.NAME,
            respond.NAME: respond.NAME,
        },
    )
    graph.add_conditional_edges(
        respond.NAME, _after_respond, {escalate.NAME: escalate.NAME, END: END}
    )
    graph.add_edge(clarify.NAME, END)
    graph.add_edge(escalate.NAME, END)
    graph.add_edge(close.NAME, END)

    return graph.compile(checkpointer=checkpointer)
