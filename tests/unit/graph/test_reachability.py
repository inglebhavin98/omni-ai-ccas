"""Graph shape assertions (CLAUDE.md Rule 4).

These are the tests an open-ended agent loop cannot support. "Can this reach a human?"
and "can this loop forever?" are answered here by reading the routing table, not by
sampling behaviour.
"""

from __future__ import annotations

from collections import deque

import pytest

from ccas.graph.assembly import NODE_NAMES, build_graph, select_entry
from ccas.graph.context import GraphContext
from ccas.nodes import escalate
from ccas.schemas.session import SessionState

START = "__start__"
END = "__end__"


@pytest.fixture
def drawn(ctx: GraphContext):
    return build_graph(ctx).get_graph()


def _edges(drawn) -> list[tuple[str, str]]:
    return [(e.source, e.target) for e in drawn.edges]


def _successors(drawn) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for source, target in _edges(drawn):
        out.setdefault(source, set()).add(target)
    return out


def _reachable(drawn) -> set[str]:
    successors = _successors(drawn)
    seen = {START}
    queue = deque([START])
    while queue:
        for target in successors.get(queue.popleft(), ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def test_every_declared_node_is_in_the_graph(drawn) -> None:
    assert set(NODE_NAMES) <= set(drawn.nodes)


@pytest.mark.parametrize("node", NODE_NAMES)
def test_every_node_is_reachable_from_the_entry(drawn, node: str) -> None:
    """An unreachable node is dead code that looks like a capability."""
    assert node in _reachable(drawn)


@pytest.mark.parametrize("node", NODE_NAMES)
def test_every_node_can_terminate(drawn, node: str) -> None:
    """From anywhere, END must be reachable -- there is no state the caller gets stuck in."""
    successors = _successors(drawn)
    seen = {node}
    queue = deque([node])
    while queue:
        for target in successors.get(queue.popleft(), ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    assert END in seen


def _turn_loop_successors(drawn) -> dict[str, set[str]]:
    """The graph as a *call* rather than a single turn.

    One invocation handles one turn, so reaching END means "we said something and are
    waiting for a reply". The call continues: the next invocation re-enters at the entry
    point. Modelling that edge is what makes escalation reachability a statement about
    the conversation rather than about one node run.
    """
    successors = _successors(drawn)
    successors.setdefault(END, set()).add(START)
    return successors


def _closure(successors: dict[str, set[str]], origin: str) -> set[str]:
    seen = {origin}
    queue = deque([origin])
    while queue:
        for target in successors.get(queue.popleft(), ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


@pytest.mark.parametrize("node", NODE_NAMES)
def test_a_human_is_reachable_from_every_node(drawn, node: str) -> None:
    """Rule 4: every graph must have a reachable escalation path."""
    if node in {escalate.NAME, "close"}:
        pytest.skip("terminal by design")
    assert escalate.NAME in _closure(_turn_loop_successors(drawn), node), (
        f"{node} cannot reach a human"
    )


def test_clarify_ends_the_turn_rather_than_looping(drawn) -> None:
    """Within one invocation clarify is terminal -- the caller must get to answer.

    Its bound is the confidence policy, not the graph: a second failed clarification
    escalates. Asserted end to end in tests/integration/test_graph_e2e.py.
    """
    assert _successors(drawn)["clarify"] == {END}


def test_terminal_nodes_go_only_to_end(drawn) -> None:
    successors = _successors(drawn)
    for node in (escalate.NAME, "close", "clarify"):
        assert successors[node] == {END}


def test_the_step_up_cycle_is_broken(drawn) -> None:
    """tool_exec -> identify exists for step-up. identify must be able to end the turn,
    or that becomes tool_exec -> identify -> route -> tool_exec forever."""
    successors = _successors(drawn)
    assert "identify" in successors["tool_exec"]
    assert END in successors["identify"]


def test_no_node_routes_to_itself(drawn) -> None:
    assert all(source != target for source, target in _edges(drawn))


# ----------------------------------------------------------------- entry point


def test_a_fresh_session_is_greeted(session: SessionState) -> None:
    assert select_entry(session) == "greet"


def test_a_terminal_session_closes(session: SessionState) -> None:
    session.terminal = True
    assert select_entry(session) == "close"


def test_a_pending_slot_captures_before_routing(session: SessionState) -> None:
    """The caller is answering a question we asked; re-routing first would lose it."""
    session.turn_index = 3
    session.pending_slot = "order_reference"
    assert select_entry(session) == "slot_fill"


def test_an_ongoing_session_routes(session: SessionState) -> None:
    session.turn_index = 3
    assert select_entry(session) == "route"


def test_terminal_outranks_a_pending_slot(session: SessionState) -> None:
    session.turn_index = 3
    session.pending_slot = "order_reference"
    session.terminal = True
    assert select_entry(session) == "close"
