# data/raw/natcs — the dialogue corpus

**NatCS spoken natural conversations.** Real multi-turn dialogue with turn timing.

## Permitted roles

| Role | Why this corpus |
|---|---|
| `transcript_ingestion` | Turn boundaries and timings map onto `Utterance.start_ms`/`end_ms`. |
| `dialogue_context_benchmark` | The only corpus with genuine cross-turn reference. |
| `state_graph_benchmark` | Real trajectories to replay through the `StateGraph`. |

## Denied roles

`intent_mining` — too small and too narrow; clusters would overfit its topic coverage
and contaminate a taxonomy meant to reflect real traffic.

`nlu_ground_truth`, `judge_evaluation` — intent labels are absent or inconsistent.

`redaction_corpus` — useful as a smoke test, but too small to establish a p99.

## Where it is used

- **Module 1** (`ingestion/adapters/natcs.py`) — dialogues → `CallLog` with timings.
- **Module 4** (`tests/integration/test_graph_e2e.py`) — replay a real trajectory and
  assert the graph transitions node-for-node and retains context across turns.
- **Module 5** (`tests/latency/`) — realistic turn-taking cadence for barge-in and
  end-of-speech tuning.

## Needed by

**Phase 4–5.** Nothing before then depends on it, so it can wait until the agentic mesh
needs real trajectories to replay.

## Layout, as verified against the live download

```
data/raw/natcs/
├── dialogues.jsonl          # hydrated by scripts/fetch_datasets.py
└── dstc11_one_intent.csv    # the published file, unmodified
```

```bash
uv run python scripts/fetch_datasets.py --corpus natcs
```

Source: `splevine/dstc11-intent` — the intent-labelled release of the DSTC11 track that
used NatCS. **It is narrower than this file originally assumed, in three ways that
matter:**

| Assumed here | Actually published |
|---|---|
| speaker-tagged turns | **caller turns only** — all 4,205 rows are `speaker_role: Customer`; there is no agent side |
| turn timing → `Utterance.start_ms`/`end_ms` | **no timings at all** |
| multi-turn dialogue | 2,747 dialogues, but 1,659 are single-turn; only 61 have >= 4 turns |

Industries are finance, banking and insurance — not retail or healthcare.

Nothing in the fetcher invents an agent reply or a timestamp to close those gaps.
`dialogue_id` is preserved verbatim; an `id` field namespaces it by industry, because the
insurance subset numbers its dialogues from `0` while the others are prefixed.

### Consequence for the declared roles

`dialogue_context_benchmark` still holds in a reduced form — a caller's trajectory across
turns is real context. **`state_graph_benchmark` does not**: replaying a trajectory
node-for-node needs the agent turns that drive the transitions, and they are not in this
release. `DATASET_ROLES` has deliberately *not* been edited to match — narrowing a role is
a decision with its own friction (Rule 9, ADR-0004), not a detail to fix in passing. Pick
one before Phase 4 leans on it: source the full NatCS release, or narrow the role.
