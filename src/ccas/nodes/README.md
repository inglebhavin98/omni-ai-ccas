# `nodes/` — M4: one node per turn behaviour

Each file is a single explicit behaviour in the graph. Nodes read `SessionState`, call
policies and tools, and return a state delta. They do not decide where control goes next
— that is `graph/`.

## Files

| File | Behaviour |
|---|---|
| `greet.py` | Opening turn |
| `identify.py` | Caller verification to the level the risk tier demands |
| `route.py` | Utterance → intent, against the loaded taxonomy |
| `clarify.py` | Confidence below threshold — ask, do not guess |
| `slot_fill.py` | Gather pack-declared slots, including DTMF capture |
| `tool_exec.py` | Dispatch through `ToolExecutor` |
| `respond.py` | Grounded answer from tool results or retrieval |
| `escalate.py` | Hand to a queue with a `HandoffContext` |
| `close.py` | Terminal state |
| `base.py` | Shared node contract |

## Two invariants

**Never improvise a factual claim about the caller's account.** Answers are grounded in
tool results or retrieval. When the data is missing, say so and escalate.

**Every node has a reachable escalation path**, and the graph has a turn ceiling. An
unbounded loop is a defect, not a tuning problem.

A new node needs a reachability test and at least one failure-path test (Rule 7).

```bash
uv run pytest tests/unit/nodes tests/unit/graph -q
```
