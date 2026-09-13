# `graph/` — M4: deterministic orchestration

LangGraph `StateGraph` only: explicit nodes, explicit conditional edges, explicit
terminal states. **No `AgentExecutor`, no open-ended ReAct loop, no auto-chaining**
([ADR-0002](../../../docs/adr/0002-langgraph-state.md)).

## Files

| File | Holds |
|---|---|
| `assembly.py` | Builds the graph from a pack — nodes, edges, terminals |
| `router.py` | Conditional-edge logic. Where control goes and why |
| `responder.py` | Response assembly at the graph boundary |
| `context.py` | Per-turn context construction |
| `checkpoint.py` | State persistence across turns |

## Why a state machine and not an agent

An agent loop is easier to write and impossible to certify. A caller-facing system needs
to answer "what can this do?" with a finite list, and "what happened on that call?" with
a trajectory. Both require the transitions to be declared rather than discovered at
runtime.

`SessionState` is the only mutable model in the platform, and it lives here.

The graph must have a reachable escalation path from every node and a turn ceiling.

```bash
uv run python -m cli.demo pipeline "where is my delivery"
uv run pytest tests/unit/graph -q
```
