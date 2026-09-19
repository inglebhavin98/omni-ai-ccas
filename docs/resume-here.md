# Resume here — 2026-09-19

A point-in-time note, not a maintained document. `docs/future-scoped-work.md` is the
living list. Delete or replace it once the next session has picked the work up.

## State

Branch `phase6/evals-timeout-verdict`, **6 commits ahead of `main`, pushed**, tree clean.
No PR opened yet.

    https://github.com/inglebhavin98/omni-ai-ccas/pull/new/phase6/evals-timeout-verdict

Green: `ruff format --check`, `ruff check`, `uv run mypy` (135 files), `pytest`
**961 passed / 2 skipped**, `make voice` **95 passed**.

## Do this first

**Revoke the exposed key.** Outstanding since 2026-09-13 across three sessions. An invalid
key returns 401; this one returns 429, so it authenticates.

    openrouter.ai/keys  ->  delete sk-or-v1-0b707...  ->  put the new key in .env

The daily quota is per *account*, so a new key does not reset it. No session has been able
to verify which key `.env` holds — reading it is blocked by the harness.

## What this branch did

**The Rule 6 parity gate ran green for the first time.** `make evals`, 33/33, zero skips.
ADR-0014 said Rule 6 was unverified for `router` and had to run green before Module 6 could
be called done. Discharged. The exact agreement number was **not captured** — see 6.7.

**ADR-0021** — a timeout is unmeasured only if the same model answered another row in the
same run. ADR-0014's table was internally ambiguous (*cold model* listed as unavailable,
*timeout* as a divergence, and a cold model is observed as a timeout). Parity is
deliberately left on the old rule; the two harnesses disagree knowingly (6.6).

**The 52.5% router figure is retired — do not quote it.** All 19 failures were
`expected -> <none>`; nothing was misrouted. They were timeouts, provable from wall clock:
per-row ceiling 92.25 s, run ≥2100 s, so instant parse failures would force the 21
successes to average 100 s each, above the ceiling. Under ADR-0021 it reads 21 measured /
19 unmeasured / 100% exact — a smoke test on 21 rows. The nine intents scoring zero were
the rows that timed out; **nothing is known about them either way**. Redo is 6.8.

**Three Rule 2 holes closed on the handoff** (schema 1.0 → 1.1, migration note in
tech-spec §1.7). `cti_attributes`, `NextBestAction.action`/`.rationale` and
`VerifiedIdentity.attributes` were plain `str` on a model whose docstring promises every
text-bearing field carries a report. The third was a **live leak**: verified attributes are
caller-derived, and `escalate.py` copied them into a vendor-bound payload unredacted.

**M6a now passes Rule 11** — gate test, demo stage and structured logs all present.
`copilot/crm/` has the adapter boundary and `MockCrmAdapter`; `future-scoped-work.md` 7.1
had claimed since Phase 4 that this file proved the contract, and it did not exist.

Verified end to end: `cli.demo pipeline` with a PAN and an email redacts to
`[PAYMENT_CARD_1]` / `[EMAIL_1]`, and the 11 attached-data pairs reaching the mock CRM
contain neither. **6/8 stages live**; the two pending ones name their phase.

## What is left

| what | why | blocked on |
|---|---|---|
| **Open the PR** | branch is pushed, nothing reviewed | nothing |
| **M6b** — judge, Ragas/DeepEval | the other half of Module 6, entirely unbuilt. Contracts already exist in `schemas/eval.py` (`JudgeDimension`, `JudgeScore`, `JudgeVerdict`) | nothing |
| **Agent-desktop surface** | `GET /v1/handoffs/{id}` and `WS /v1/ws/copilot/{session_id}`, tech-spec §3.2a, still "planned" | nothing |
| **Capture the parity numbers** (6.7) | the Rule 6 gate proves something and records nothing — no `configure_logging` in a pytest run | 16 calls, or a conftest fixture |
| **Redo the router eval** (6.8) | the only accuracy number rests on 21 rows | a provider answering inside 30 s |
| **Align the two harnesses** (6.6) | they disagree about a timeout | the first parity run that actually times out |
| **A real browser test** (9.21) | a reviewer clicks the UI before reading an ADR | an ADR — Playwright is outside the locked stack |
| **`docker-compose.yml` never started** (9.20) | valid YAML, written on a machine without Docker | a machine with Docker |

`src/cli/stages.py` has one `_pending` stage (`_stage_intent`) and it is an honest
conditional — it fires only when the pack has no mined taxonomy and prints the command to
mine it.

## Known-unknown worth stating plainly

Coverage on the mined insurance corpus tops out at **19.7%** and **the cause is not
established**. Three explanations were tested and refuted — call direction, campaign
variety, lexical repetition — and one lexical result was retracted after the measure turned
out to carry a vocabulary-size artefact. `docs/future-scoped-work.md` 9.17 has the record.
Do not re-propose the outbound-sales explanation; it is refuted with numbers.

## What not to touch

`src/ccas/voice/` is frozen (ADR-0018) and must stay green under `make voice`. The tripwire
at `tests/test_module_5_voice_frozen.py` runs by default and has already caught the core
drifting away from it once.

`domains/` stays at the repo root because packs are data, not code — moving them under
`src/` would break the domain-literal gate that keeps the core honest.
