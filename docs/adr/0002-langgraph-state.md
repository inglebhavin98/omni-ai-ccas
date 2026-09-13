# ADR 0002 — Deterministic StateGraph, not an agent loop

Status: accepted · 2026-09-12

## Context

Voice has no undo. A caller cannot scroll back to check what was said, cannot re-read a
wrong answer, and cannot abandon a half-finished action cheaply. Every turn is committed
the moment it is spoken.

Open-ended agent loops — ReAct, `AgentExecutor`, auto-chaining frameworks — are
attractive because they generalise without new wiring. They also offer **no static
guarantee about what the system can do on a given turn**. "What tools can this reach?"
and "can this loop forever?" become empirical questions answered by sampling.

## Decision

Orchestration is a LangGraph `StateGraph` with explicit nodes, explicit conditional
edges, and explicit terminal states, over a single `SessionState`.

- The model chooses *which* declared tool to call. It never constructs a URL, a query,
  or an unregistered call; tools come from the domain pack registry.
- Every graph must have a reachable escalation path and a turn ceiling.
- Answers are grounded in tool results. Missing data escalates; it never gets improvised.
- Append-only state fields (`turns`, `intent_history`, `sentiment_trail`, `tool_records`)
  carry `add` reducers so concurrent nodes merge instead of clobbering.

## Consequences

- Graph shape becomes testable. `tests/integration/test_graph_e2e.py` asserts every node
  is reachable and every path terminates — an assertion an agent loop cannot support.
- Adding a capability means adding a node and an edge, which surfaces in review rather
  than emerging from a prompt change.
- Genuinely open-ended requests are out of scope by construction; they escalate to a
  human. For a care platform that is the correct failure mode.
- Cost: more upfront wiring per intent, and a real risk of graph sprawl as verticals
  accumulate. `agents/factory.py` builds one subgraph per L1 intent from the pack to keep
  that growth declarative rather than hand-written.
