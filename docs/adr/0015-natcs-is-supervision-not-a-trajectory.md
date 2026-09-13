# ADR-0015 — NatCS is supervision, not a trajectory

- **Status:** accepted
- **Date:** 2026-09-13
- **Amends:** [0004](0004-dataset-role-separation.md)

## Context

ADR-0004 fixed each corpus to the roles it can support and put the rule in code
(`DATASET_ROLES`) rather than a README, so a misuse fails at the point of the mistake.
The principle holds. Two of the three facts it rests on turned out to be wrong about
NatCS, and I had flagged the first one rather than quietly editing the ADR.

Measured against `data/raw/natcs/dialogues.jsonl` (2,747 dialogues, 4,205 turns):

| ADR-0004 said | The corpus says |
|---|---|
| unlabelled, so never ground truth | **every turn carries an intent label** — 82 distinct intents across insurance (854 dialogues), banking (857) and finance (1,036) |
| "real turn-taking", so the context and state-graph benchmark | **8 of 1,088 multi-turn dialogues have consecutive turn ids.** The release publishes *labelled turns*, not conversations. `insurance_000_001` is followed by `insurance_000_047` |
| — | no slot annotations anywhere |

The roles were exactly inverted. NatCS was admitted for the one thing it cannot do and
forbidden from the one thing it is.

The second row is the serious one. A state-graph benchmark built on sampled turns would
have run, produced a transition-accuracy percentage, and meant nothing — the failure mode
ADR-0004 exists to prevent, reached by trusting a dataset description instead of the
dataset.

## Decision

`DATASET_ROLES[NATCS]` becomes `{TRANSCRIPT_INGESTION, NLU_GROUND_TRUTH, JUDGE_EVALUATION}`.

Removed: `DIALOGUE_CONTEXT_BENCHMARK`, `STATE_GRAPH_BENCHMARK`. **No corpus on disk holds
either role now.** Both stay declared in `DatasetRole` with a docstring saying they are
unsourced, and a test asserts no real corpus carries them — an honest empty set is worth
more than a role assigned to data that cannot support it.

Not granted: `SLOT_EXTRACTION_GROUND_TRUTH` (no parameter annotations) and `INTENT_MINING`
(labelled — clustering it would rediscover its own label set, exactly the Bitext
reasoning).

**Bitext and NatCS now share `NLU_GROUND_TRUTH`, and that overlap is deliberate.** Bitext
is templated: `question about cancelling order {{Order Number}}` with machine-generated
typo variants. NatCS is what people actually said: *"been noticing multiple debit
transactions on my account and I'm starting to think I'm being ripped off."* A router
graded only on templates scores better than it is. Two graders, one templated and one
natural, is a stronger bar than either alone.

The old invariant — "the three corpora have disjoint primary roles" — was therefore
retired, because it is not the property worth defending. The replacement is narrower and
actually load-bearing: **no corpus is admitted for both mining and supervision.**

## Alternatives considered

- **Edit ADR-0004 in place.** Forbidden by Rule 10, and rightly: the reasoning that led to
  a wrong role assignment is the useful part of the record.
- **Keep the benchmark roles on NatCS and filter to the 8 consecutive dialogues.** Eight
  dialogues is not a benchmark.
- **Reconstruct trajectories by interpolating the missing turns.** Inventing conversation
  to grade a conversation engine.
- **Leave NatCS out of supervision to preserve disjointness.** Preserving a tidy invariant
  by discarding the only natural-speech labels available, in the vertical this project
  just built a pack for.

## Consequences

- **Phase 6 loses a planned benchmark.** Multi-turn context retention and state-graph
  transition accuracy have no data source. `docs/future-scoped-work.md` 9.4 carries this;
  it was previously written as "narrow the role or source the full release", and the first
  half is now done.
- **The router gains a harder grader**, and one covering insurance — the vertical the
  AIxBlock corpus turned out to be (ADR-0013). NatCS's 22 insurance intents are a
  ready-made check on whether a mined taxonomy covers what callers actually ask.
- `CLAUDE.md` Rule 9's table is updated. Its "never use for" column for NatCS now reads
  *intent mining (labelled), slot extraction (unannotated), trajectory replay (turns are
  sampled)*.
- Nothing consumed the removed roles yet, so there is no migration.
