# ADR-0014 — A throttled parity case is unmeasured, not divergent

- **Status:** accepted
- **Date:** 2026-09-13
- **Extends:** [0012](0012-openrouter-free-models.md), [0003](0003-hybrid-llm-provider.md)

## Context

Rule 6 says no node depends on a single model. Until today that rule had a test named in
`CLAUDE.md` and no test on disk, so it had never executed. Writing it surfaced a question
the rule does not answer.

`tests/evals/test_provider_parity.py` puts the same eight utterances to both router
variants and compares the intent each returns. On its first live run it reported 25%
agreement and six divergences. None of them were disagreements:

```
openrouter returned 429 for 'nex-agi/nex-n2.5-pro:free' after 3 attempt(s)
  Rate limit exceeded: free-models-per-day  (X-RateLimit-Limit: 50, Remaining: 0)
```

The free tier allows 50 model requests per day. A parity run costs 16. The gate was
reporting a quota as a model defect — which is the failure mode that trains a team to
ignore a gate.

The opposite reflex is worse. Treating any unanswered case as "skip, carry on" means a
provider outage reads as green, and the one test that exists to catch a silent
single-model dependency becomes the thing that hides it.

## Decision

Parity outcomes have three verdicts, not two.

| Outcome | Meaning | Effect on the report |
|---|---|---|
| a decision | the variant answered | counted; agreement compared |
| an error | the variant failed — timeout, 5xx, malformed output | counted as a **divergence** |
| unavailable | the request was never served — 429, quota, cold model | **dropped from the denominator**, counted in `skipped` |

Three rules keep the third verdict from becoming a loophole:

1. `ParityReport.agreement_rate` over zero measured cases is **0.0, not 1.0**. An empty
   run is not a perfect run.
2. `meets(threshold, min_cases=N)` requires `N` *measured* cases. The router gate demands
   five of eight.
3. `inconclusive(min_cases=N)` is true only when the shortfall is accompanied by
   `skipped > 0`. The test skips on that, with the count in the reason. A shortfall with
   nothing throttled means a broken harness and still fails.

`ProviderRateLimitedError` is a new subclass of `LLMProviderError`, raised by
`OpenRouterProvider` on 429 after retries are exhausted. Classifying at the provider
means the harness matches an exception type rather than sniffing an error string.

## Alternatives considered

- **Count a 429 as a divergence** (the behaviour observed). Honest about there being no
  measurement, dishonest about why, and it makes the gate red on a quiet Sunday.
- **Retry until served.** Retries already run three deep; the daily cap is not a transient.
  Waiting 15 hours inside a test is not a test.
- **Skip the whole test when credentials are missing or exhausted.** This is what the
  test does for *missing credentials*, where nothing can be inferred. For *exhausted*
  credentials the distinction still matters: a run that measured six of eight cases should
  be graded on those six, not discarded.
- **Pay for credits so the limit never binds.** Reasonable for CI, and orthogonal —
  the verdict model has to be right either way, because any shared quota eventually binds.

## Consequences

- **Rule 6 is currently unverified for `router`.** The gate skips with
  `parity unmeasured: 8/8 cases were rate limited`. That is a true statement of what is
  known, and it is louder than a pass. It must run green before Module 6 is called done.
- `make evals` must be run when the quota has room — 16 requests against a 50/day cap.
  Raising the cap costs $10 in credits and lifts it to 1,000/day.
- The pacing constant (`PACING_S = 3.0`) exists for the *per-minute* limit and does
  nothing for the per-day one. It is not a fix, only politeness.
- `ProviderRateLimitedError` is now available on the call path too. Nothing catches it
  there yet; a turn that hits a quota should escalate rather than retry into the same
  wall, and that is listed in `docs/future-scoped-work.md`.
