# Resume here — 2026-09-27

A point-in-time note, not a maintained document. `docs/future-scoped-work.md` is the
living list. Delete or replace it once the next session has picked the work up.

## State

Branch `spike/typesafe-jev-router`, uncommitted working tree (see the commit list below).
Green: `ruff format --check`, `ruff check`, `mypy --strict` (143 files), pytest
**1039 passed / 3 skipped**, `make browser` **3 passed**.

## Do this first

**Revoke the exposed key.** Outstanding since 2026-09-13 across four sessions. An invalid
key returns 401; this one returns 429, so it authenticates.

    openrouter.ai/keys  ->  delete sk-or-v1-...6638  ->  put the new key in .env

The daily quota is per *account*, so a new key does not reset it.

## What this session did

**M6b judge built and run live** (`src/ccas/evals/judge.py`, gate
`tests/test_module_6b_judge.py`, demo stage in `src/cli/stages_llm.py`): `SessionJudge`
over the locked `LLMProvider` contract — two variants per Rule 6, quota/unavailable =
`None` unmeasured, malformed = raise. The judge node was **rebound off
`ling-3.0-flash-fin`** (it burns its whole token allowance reasoning inline before any
JSON — truncated twice) to `nemotron-3-super-120b-a12b` primary +
`dots-3-note-preview` alt, both `structured_mode: response_format`, `max_tokens: 8192`.

**First live suite run** (`scripts/run_evals.py`, 2026-09-27): 17/30 measured, 13
unmeasured (429s). **The 6.12 rubric gap reproduces live:** faithfulness 0.00 vs 0.00
(+0.00) — the judge cannot tell a grounded refusal from a hallucination under the current
rubric; task_success separates (+0.67); policy_adherence does not. The faithfulness rubric
rewrite is the blocking experiment (6.12 has the numbers).

**Agent-desktop surface shipped** (tech-spec §3.2a, was "planned"): `GET
/v1/handoffs/{id}`, `GET /v1/sessions/{id}/handoff`, `WS /v1/ws/copilot/{session_id}`.
Terminal+escalated sessions retain their handoff; the socket closes 4000 on an unknown
session.

**Browser tests landed (ADR-0022, closes 9.21)**: Playwright 1.63 dev-only via npm, three
specs under `make browser` against the empty-key server. **First run caught a real
defect**: `send()`'s `finally` re-enabled the Send button after `render()` disabled it for
a terminal session — a dead button made live again. Fixed.

**Empty-key guard** (`factory.key_is_set`): `OPENROUTER_API_KEY=` read as configured, so
readiness claimed a provider that did not exist and every call failed with a 401 that
looked like a broken key. Unit-tested.

**End-to-end verified** (`scripts/e2e_run.py`, live): session → turns → escalation →
handoff fetch → copilot WS → judge, 3 LLM calls.

## What is left

| what | why | blocked on |
|---|---|---|
| **Review and merge PR #1** | open since 2026-09-19 | a reviewer |
| **Rewrite the faithfulness rubric** (6.12) | the judge cannot separate grounded refusals from hallucinations; live numbers confirm the spike | the rewrite + a second label reader; a re-run costs 30 quota |
| **Widen the router eval** (6.8) / chat cutoff (6.11) | 26 measured rows is ~1 per intent | quota — 13/30 judge cases went unserved this session; the cap bites fast |
| **Parity gate re-run** (6.7) | 16 calls; plumbing fixed, numbers wanted | quota headroom |
| **Docker compose verify** (9.20) | never started | a machine with Docker |

## Known-unknown worth stating plainly

The stub router (`tests/graph_stub.py`) matches keywords against the test-fixture
taxonomy, not `domains/retail/taxonomy.json` — through the API, every spoken turn
escalates under the stub. Pre-existing, understood, not a defect introduced this session.

## What not to touch

`src/ccas/voice/` is frozen (ADR-0018) and must stay green under `make voice`.

`domains/` stays at the repo root because packs are data, not code.
