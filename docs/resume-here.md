# Resume here — 2026-09-15 (evening)

A point-in-time note, not a maintained document. `docs/future-scoped-work.md` is the
living list; this says what to do **first** and what is blocked on what. Delete or replace
it once the next session has picked the work up.

## State

39 commits on `main`. **The working tree is dirty and everything in it is finished** — the
decision that was blocking it has been made (ADR-0021). It is ready to commit; it was left
uncommitted only because nobody asked for a commit.

Green as of this note: `ruff format --check`, `ruff check`, `uv run mypy` (133 files),
`pytest` **939 passed / 2 skipped**, `make voice` **95 passed**.

## Do this first

**1. Revoke the exposed key.** Outstanding since 2026-09-13 and still not done. An invalid
key returns 401; this one returns 429, so it authenticates.

    openrouter.ai/keys  ->  delete sk-or-v1-0b707...  ->  put the new key in .env

The daily quota is per *account*, so a new key does not reset it. (Neither session could
verify which key `.env` holds — reading it is blocked by the harness.)

**2. Commit the tree.** `make check` is green. Branch per convention
(`phase<N>/<module>-<slug>`), Conventional Commits. Rule 10 docs are already written and
are part of the same change.

**3. Quota.** ~16 of 50 free calls spent on 2026-09-15 (the parity gate). The
2026-09-14 router run spent ~40. Resets 00:00 UTC.

## What happened this session

### The Rule 6 parity gate ran green for the first time

`make evals` — exit 0, **33/33, zero skips**. ADR-0014 said *"Rule 6 is currently
unverified for `router`… It must run green before Module 6 is called done."* Discharged.
Passing means ≥75% agreement over ≥5 measured cases across both router variants.

**The exact agreement number was not captured**, and that is a real gap, now
`future-scoped-work.md` 6.7. The gate emits `parity.router` through `LOG.info`, but nothing
calls `configure_logging` in a pytest run, so it went to captured stdout and vanished on
pass. `logs/execution.log` has `eval.router` twice and no `parity.router` ever. The first
green run of the project's most important gate was observed only as an exit code.
Re-running to capture it costs 16 calls.

### The router evaluation was diagnosed, and its headline retired

`make eval-router` on 2026-09-14 returned `21/40 exact (52.5%), 0 unmeasured, p95 83277 ms`.
That number is **superseded — do not quote it.**

- All 19 failures were `expected -> <none>`. **Nothing was misrouted.**
- The failures were **timeouts**, proved from wall clock: per-row ceiling is 92.25 s
  (3 x 30 s + 2.25 s backoff), the run took ≥2100 s, so instant parse failures would force
  the 21 successes to average 100 s each — above the ceiling, impossible.
- Under ADR-0021 the same outcomes read **21 measured, 19 unmeasured, 100% exact**. That
  is a smoke test on a denominator of 21, not a result.
- The nine intents scoring zero were the rows that timed out. **Nothing is known about
  them either way** — do not read that column as bad routing.

Redoing it properly is `future-scoped-work.md` 6.8, blocked on a provider that answers
inside 30 s.

### ADR-0021 — a timeout is read against the run

ADR-0014's verdict table was **internally ambiguous**: it listed *cold model* as unavailable
and *timeout* as a divergence, but a cold model is observed as a timeout. The live run
landed on the seam.

Decided (option 3 of three, chosen over a blanket reversal):

- **A 429 is self-describing** — unmeasured on its own. Unchanged from ADR-0014.
- **A timeout is ambiguous** — unmeasured only if the same model answered another row in
  the same run; divergent otherwise, so a wholly dead binding still fails.

The classification is split on purpose: `outcome_for_exception` records *what happened* to
a row, `score_router` decides *what it means*, because only the scorer sees the whole run.

What changed the recommendation mid-session: the parity gate going green showed timeouts
are an occasional capacity event, not systematic. A blanket reversal would have been a
large change bought by one run — and would have loosened a gate immediately after it
started passing. The reasoning is in the ADR; the counter-arguments are in its Alternatives
section.

**Deliberate inconsistency:** `parity` still scores a timeout as a divergence per ADR-0014.
`ProviderTimeoutError` subclasses `LLMProviderError`, not `ProviderRateLimitedError`, so the
parity path is untouched. The two harnesses now disagree about a timeout. That is accepted
knowingly and recorded as `future-scoped-work.md` 6.6, to be resolved by the first parity
run that actually times out rather than by guessing now.

## What is in the dirty tree

```
 docs/adr/0021-a-timeout-is-read-against-the-run.md   (new)
 docs/adr/README.md            0014 -> "amended by 0021"; 0021 indexed
 docs/future-scoped-work.md    6.6, 6.7, 6.8, 6.9
 docs/skills.md                unmeasured-vs-divergent convention
 docs/resume-here.md           this file
 scripts/eval_router.py        failures_by_cause; progress marks
 src/ccas/evals/router_accuracy.py   outcome_for_exception; run-aware scoring; latency fix
 src/ccas/llm/base.py          ProviderTimeoutError
 src/ccas/llm/openrouter_provider.py  raises it when the ladder is exhausted
 src/ccas/llm/vllm_provider.py        converts httpx.TimeoutException
 tests/evals/test_router_accuracy.py         +8
 tests/unit/llm/test_openrouter_provider.py  +1
 tests/unit/llm/test_vllm_provider.py        +1
```

Beyond the ADR, three fixes worth knowing about:

- **The latency sample was polluted.** `score_router` appended `latency_ms` for every
  measured row, and a failed row carries 0 ms — 19 zeros in a 40-value p95. Rank 38 of 40
  still landed in real data so 83,277 ms was genuine, but the statistic was structurally
  wrong and a smaller failure count would have shifted the rank into the zeros.
- **`failures_by_cause`** in the report and on the progress line. Its absence is why the
  timeout diagnosis had to be reconstructed from wall-clock arithmetic instead of read off
  the report.
- **`vllm_provider.py` let `httpx.ReadTimeout` escape raw.** The exact failure
  `test_a_read_timeout_never_escapes_as_an_httpx_error` prevents on the OpenRouter side;
  the OpenRouter docstring says why — *"a raw httpx error propagating out of a graph node
  takes the whole call down"*. Now converted, with the mirrored test.

## Then, in rough order of value

| what | why | blocked on |
|---|---|---|
| **M6 copilot** | the only module with no gate test (Rule 11). `src/ccas/copilot/` is two empty `__init__.py` files, 6 lines total. `HandoffContext` is already frozen and strict — validators reject an unredacted summary, transcript turn, slot or tool result, plus an L1->L2->L3 path check. Missing: summariser -> disposition -> handoff builder -> mock CRM adapter, a gate test, a demo stage and structured logs | nothing |
| **Capture the parity numbers** (6.7) | the Rule 6 gate proves something and records nothing | 16 calls, or a conftest fixture |
| **Redo the router eval** (6.8) | the only accuracy number the project has rests on 21 rows | a provider that answers inside 30 s |
| **A real browser test** (9.21) | a reviewer clicks the UI before reading an ADR | an ADR — Playwright is outside the locked stack (Rule 5) |
| **`docker-compose.yml` never started** (9.20) | valid YAML, written on a machine without Docker | a machine with Docker |

Note: `future-scoped-work.md` 7.1 claims `copilot/crm/mock.py` "proves the contract". That
file does not exist. The claim is aspirational, not a record.

`src/cli/stages.py` has one `_pending` stage (`_stage_intent`, line 182) and it is an
honest conditional — it fires only when the pack has no mined taxonomy and prints the
command to mine it. An earlier note said there were two; there is one.

## Known-unknown worth stating plainly

Coverage on the mined insurance corpus tops out at **19.7%** and **the cause is not
established**. Three explanations were tested and refuted — call direction, campaign
variety, lexical repetition — and one lexical result was retracted after the measure turned
out to carry a vocabulary-size artefact. `docs/future-scoped-work.md` 9.17 has the full
record. Do not re-propose the outbound-sales explanation; it is refuted with numbers.

## What not to touch

`src/ccas/voice/` is frozen (ADR-0018) and must stay green under `make voice`. The tripwire
at `tests/test_module_5_voice_frozen.py` runs by default and has already caught the core
drifting away from it once.

The folder structure was reviewed and deliberately left alone. `domains/` stays at the repo
root because packs are data, not code — moving them under `src/` would break the
domain-literal gate that keeps the core honest.
