# `evals/` — M6: measuring the platform

Offline scoring. Ragas, DeepEval and the async judge runtime land with the rest of M6b;
what ships today is parity, router accuracy, and the two instruments below.

## Files

| File | Holds |
|---|---|
| `parity.py` | Cross-variant comparison, verdicts, throttle handling |
| `router_accuracy.py` | Held-out scoring for the router: exact and category accuracy, per-intent scores, confusions, p95. Owns `holdout_cases`, so two harnesses grade the *same* rows, and `measured_rows`, so they agree on which rows count |
| `confidence.py` | What a confidence cutoff actually buys — coverage, accuracy above it, and whether deferred rows still had the right L1 |
| `judge_rubric.py` | `JudgeDimension` as an ordered rubric a non-writing model can answer |

## A threshold is measured, not inherited

The pack routes at `intent_confidence >= 0.82`. That number belongs to **one rubric asked
of one model** and does not travel — a different model reports a different statistic, and
a decision model reports the *shape* of a distribution rather than the winning option's
probability at all.

`sweep()` splits a graded run at each candidate cutoff. `recommend()` returns the
**lowest** cutoff clearing an accuracy floor — every point above what the floor needs is
coverage given away — and returns `None` rather than name one backed by fewer than five
rows. Both router harnesses print the same table, so two models can be compared by eye.

The deferred slice is scored twice, at the leaf and at the L1 category, because *unsure*
is not *no idea*: a row the router will not commit to may still have the right category,
and that queue is cheaper than a human.

## A judge dimension is a rubric

Each `JudgeDimension` is an ordered set of level descriptions, low end first, so a model
that cannot write prose can still answer it. The rationale is then authored **in code**
from the level chosen — stronger than a generated one, because it is the rubric's own
words and cannot describe a level the model did not pick.

`pii_leakage` has no rubric and never will. Judging it means sending pre-redaction text
to a vendor, which Rule 2 forbids; a post-redaction judge could only confirm that clean
text is clean, which local Presidio already establishes more strongly. A test pins the
absence so nobody adds one as an oversight.

**Neither instrument has yet produced a usable number for the judge.** See
`docs/future-scoped-work.md` 6.12 — separation held on twelve cases and did not survive
thirty.

## Unmeasured is not agreement

The rule that makes parity numbers mean something
([ADR-0014](../../../docs/adr/0014-parity-verdicts-and-throttling.md)):

- a case nobody served — a 429, an exhausted quota — is **unmeasured**, and leaves the
  denominator
- a case that **failed** is divergent
- a run with too few measurements **skips**, with the count in the reason. It never passes
- **zero measured cases is 0% agreement, not 100%**

That last one is the trap. An empty result set trivially satisfies "no divergences
found", which is how a parity gate silently stops testing anything.

Any change to a prompt, tool schema, or graph node must pass
`tests/evals/test_provider_parity.py`. If two variants diverge beyond threshold, fix it
or record the divergence in an ADR — do not quietly pin to one.

Grading uses Bitext and NatCS gold labels. Never grade a taxonomy against the corpus it
was mined from.

```bash
make evals
```
