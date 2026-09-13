# ADR 0004 — Corpora have fixed, enforced roles

Status: accepted · 2026-09-12

## Context

Three public corpora back the platform, and they are superficially similar enough to
pool: all three are customer-support conversations in English.

They are not equivalent. AIxBlock is unlabelled scripts at volume. Bitext is labelled,
single-turn, already sanitised. NatCS is real spoken dialogue with timing, and small.

The failure mode is specific and quiet. If the taxonomy is mined from **Bitext**, HDBSCAN
rediscovers Bitext's own label set; the taxonomy then scores near-perfectly against
Bitext, and the number means nothing — it measures self-consistency, not whether the
miner can find structure in real traffic. Nothing crashes. The dashboard looks good.

## Decision

Each corpus is admitted only for roles it can actually support, declared in
`src/ccas/ingestion/datasets.py` and enforced by `require_role()`, which raises
`DatasetRoleError` at the top of any stage that misuses one.

| Corpus | Permitted | Notably denied |
|---|---|---|
| `aixblock` | transcript ingestion, redaction corpus, **intent mining** | all ground-truth roles (unlabelled) |
| `bitext` | **NLU ground truth**, slot-extraction ground truth, judge evaluation | **intent mining** (circular) |
| `natcs` | transcript ingestion, **dialogue context benchmark**, state-graph benchmark | intent mining (too narrow), ground truth (unlabelled) |

The three primary roles are disjoint by construction, and a test asserts it — if two
corpora shared one, the separation would be decorative.

`synthetic`, `generic_csv` and `live_capture` are unrestricted: they carry no
corpus-level bias.

## Consequences

- Mining runs on AIxBlock; Bitext then *grades* the result. External validity is a real
  check rather than a restatement.
- A misuse fails at the point of the mistake with a message naming the corpora that
  would have worked, instead of surfacing as a suspiciously good metric.
- Adding a role to a corpus is a deliberate edit to `DATASET_ROLES` with a test to
  update — the friction is the point.
- Cost: three adapters instead of one loader, and three fixture sets.
