# ADR-0021 — A timeout is unmeasured only if the model answered elsewhere

- **Status:** accepted
- **Date:** 2026-09-15
- **Amends:** [0014](0014-parity-verdicts-and-throttling.md)
- **Context from:** [0012](0012-openrouter-free-models.md), [0018](0018-freeze-voice-pivot-to-chat.md)

## Context

ADR-0014 gave a parity outcome three verdicts and put each failure mode in exactly one:

| Outcome | Meaning | Effect |
|---|---|---|
| an error | the variant failed — **timeout**, 5xx, malformed output | counted as a **divergence** |
| unavailable | never served — 429, quota, **cold model** | **dropped from the denominator** |

That table is internally ambiguous, and nothing had exercised the ambiguity until now.
**A cold model is listed as unavailable, but a cold model is observed as a timeout.** The
two rows describe the same event. ADR-0014 was written from a run in which every failure
was a clean 429, so the overlap never had to be resolved.

The first live router evaluation landed exactly on it (`make eval-router`, 2026-09-14,
retail, 40 held-out Bitext rows, `nex-agi/nex-n2.5-pro:free`):

```
21/40 exact (52.5%), category 52.5%, 0 unmeasured, p95 83277 ms
```

Read naively that is a router getting half its intents wrong. It is not:

- **All 19 failures are `expected -> <none>`**, across 15 distinct confusion entries.
  Not one row was routed to a wrong intent.
- **The failures were timeouts.** Provable from wall clock without spending a call: the
  per-row ceiling is 3 attempts x 30 s + 2.25 s backoff = **92.25 s**, and the run took at
  least 2100 s. Had the 19 failures been instant parse failures, the 21 successes would
  each have to average 2100/21 = **100 s, above the 92.25 s ceiling** — impossible.
- The 83 s p95 is over successes only. Against a 30 s timeout that means those rows landed
  on retry attempt 2 or 3. The free tier was the bottleneck, not the prompt.

So the headline conflated **capacity** with **comprehension** — the precise failure
ADR-0014 exists to prevent, arriving through the one door ADR-0014 left open. Accuracy on
rows the model actually answered was 21/21.

The opposite reflex remains worse, and ADR-0014 was right to fear it. Dropping every
timed-out row unconditionally means a binding that answers *nothing* reports "nothing to
see" rather than failing, and Rule 6's one defence against a silent single-model
dependency becomes the thing that conceals it.

A second measurement bounds the problem. With quota available, `make evals` ran green for
the first time on 2026-09-15 — 33/33, no skips, agreement at or above the 0.75 threshold
over at least five measured cases. Timeouts are therefore an **occasional capacity event**,
not a standing property of the free tier, and a blanket reversal of 0014 would be a large
change bought by a narrow problem — made immediately after the gate started passing, which
is when loosening it deserves the most suspicion.

## Decision

A 429 and a timeout are not the same kind of evidence, and are no longer treated alike.

- **A 429 is self-describing.** The provider stated it never served the request. It is
  unmeasured on its own, with no corroboration. Unchanged from ADR-0014.
- **A timeout is ambiguous on its own.** It is unmeasured **only if the same model
  answered at least one other row in the same run**, and a divergence otherwise.

The same binding answering elsewhere in the run is the evidence that the clock ran out
rather than the model. Absent any answer, the run cannot separate a queue from a model
that cannot serve, and ADR-0014's verdict stands: it fails.

Mechanically, the classification is now in two places on purpose. `outcome_for_exception`
records *what happened* to one row (`unavailable` for a 429, `timed_out` for an exhausted
ladder); `score_router` decides *what it means*, because only the scorer sees the whole
run. A per-row classifier cannot make this call, which is why ADR-0014 could not have
made it.

`ProviderTimeoutError` is a new subclass of `LLMProviderError`, raised by
`OpenRouterProvider` when the retry ladder is exhausted and by `VLLMProvider` on any
`httpx.TimeoutException`. As with `ProviderRateLimitedError`, classifying at the provider
lets a harness match an exception type instead of sniffing an error string.

### Scope

This amends the verdict model for **`ccas.evals.router_accuracy`**. The parity gate
(`ccas.evals.parity`, `tests/evals/test_provider_parity.py`) is **deliberately unchanged**:
`ProviderTimeoutError` subclasses `LLMProviderError` and not `ProviderRateLimitedError`, so
a timeout there is still a divergence exactly as ADR-0014 specified.

That leaves the two harnesses disagreeing about a timeout, which is a real cost and is
accepted knowingly. The reason is evidence, not convenience: the router evaluation is 40
sequential rows and demonstrably provokes the capacity failure, while parity is 16 paced
calls and has not. Aligning them belongs with the next parity run that actually times out —
recorded in `docs/future-scoped-work.md` rather than guessed at here.

## Alternatives considered

- **Leave 0014 alone.** Defensible; the seam is narrow. Rejected because the one live
  router number the project has is contaminated by it, and anyone reading 52.5% without the
  wall-clock arithmetic draws the wrong conclusion about the core.
- **Blanket reversal: every timeout is unmeasured, in both harnesses.** The obvious fix,
  and what the session first recommended. Rejected once parity ran green: it is a large
  change bought by one run, it reopens ADR-0014's dead-model loophole, and it makes a gate
  easier to pass immediately after it started passing.
- **Raise `timeout_ms` or `max_attempts` until timeouts stop.** Treats the symptom, hides
  the capacity signal entirely, and lengthens an already ~50-minute run. The budget is a
  contract (Rule 3); a measurement problem is not a reason to move it.
- **Require N answers rather than one** before a timeout counts as unserved. More robust
  against a model that answers once by luck, but N is a knob with no evidence behind it.
  One answer is the weakest claim that does the job; raise it when a run justifies it.

## Consequences

- The 52.5% figure from 2026-09-14 is **superseded and must not be quoted**. Under this
  rule the same outcomes read as 21 measured, 19 unmeasured, 100% exact — on a denominator
  small enough that it is a smoke test, not a result. The run needs redoing on a provider
  that can answer inside 30 s before any accuracy claim is made.
- `make eval-router` now prints failures by cause, and `failures_by_cause` is in the JSON
  report. The absence of that breakdown is why the original diagnosis had to be
  reconstructed from wall-clock arithmetic.
- A timeout's verdict is no longer inferable from the exception alone, so the progress line
  marks a row `-` only for a 429; a timed-out row shows `x` until the run is scored.
- **`make evals` leaves no durable record.** The gate emits `parity.router` via `LOG.info`,
  but nothing calls `configure_logging` in a pytest run, so the event never reaches
  `logs/execution.log`. The first green parity run was therefore observed only as an exit
  code. Listed in `docs/future-scoped-work.md`.
- Rule 6 is verified for `router` as of 2026-09-15.
