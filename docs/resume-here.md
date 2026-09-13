# Resume here — 2026-09-13

A point-in-time note, not a maintained document. `docs/future-scoped-work.md` is the
living list; this says what to do **first** and what is blocked on what. Delete or replace
it once the next session has picked the work up.

## State

38 commits, pushed to `github.com/inglebhavin98/omni-ai-ccas`, working tree clean.
`make check` exits 0: 929 passed, 3 skipped on the default selection, 95 more under
`make voice`. ruff and `mypy --strict` clean.

The chat channel runs end to end. A clone needs one key and `make workbench`.

## Do this first

**1. Revoke the exposed key.** It is still live. An invalid key returns 401; this one
returns 429, which means it authenticates. A new key was created but the old one was never
deleted, and `.env` still uses it.

    openrouter.ai/keys  ->  delete sk-or-v1-0b707...  ->  put the new key in .env

The daily quota is per *account*, so a new key does not reset it.

**2. Run the router evaluation.** The free-tier quota resets at **00:00 UTC**. This is the
number that decides whether the core works — until it runs, "the chat channel works" means
"it routes without crashing".

    make eval-router          # 40 held-out rows, one LLM call each

Reports exact accuracy, category accuracy, per-intent scores, the most frequent confusions
and p95 latency. A throttled row is reported unmeasured rather than wrong, so a partial run
is still honest.

**3. Run the Rule 6 parity gate**, which has never once run green.

    make evals                # 16 calls: 8 cases x 2 router variants

Budget both against the 50/day free-tier cap: the two together are ~56 calls, so they will
not both fit in one day without credits.

## Known-unknown worth stating plainly

Coverage on the mined insurance corpus tops out at **19.7%** and **the cause is not
established**. Three explanations were tested and refuted — call direction, campaign
variety, lexical repetition — and one lexical result was retracted after the measure turned
out to carry a vocabulary-size artefact. `docs/future-scoped-work.md` 9.17 has the full
record. Do not re-propose the outbound-sales explanation; it is refuted with numbers.

## Then, in rough order of value

| what | why | blocked on |
|---|---|---|
| **M6 copilot** | only module with no gate test (Rule 11): CRM handoff, disposition, summary | nothing |
| **A real browser test** (9.21) | a reviewer clicks the UI before reading an ADR; route and DOM contracts are covered, rendering is not | an ADR — Playwright is outside the locked stack (Rule 5) |
| **Two `_pending` demo stages** | `src/cli/stages.py`; Rule 11 requires a real call or an honest pending | the modules they demo |
| **`docker-compose.yml` never started** (9.20) | valid YAML, written on a machine without Docker | a machine with Docker |

## What not to touch

`src/ccas/voice/` is frozen (ADR-0018) and must stay green under `make voice`. The tripwire
at `tests/test_module_5_voice_frozen.py` runs by default and has already caught the core
drifting away from it once.

The folder structure was reviewed and deliberately left alone. `domains/` stays at the repo
root because packs are data, not code — moving them under `src/` would break the
domain-literal gate that keeps the core honest.
