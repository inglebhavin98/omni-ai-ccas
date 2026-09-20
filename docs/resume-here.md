# Resume here — 2026-09-20

A point-in-time note, not a maintained document. `docs/future-scoped-work.md` is the
living list. Delete or replace it once the next session has picked the work up.

## State

`main` is at `0587747` (PR #1 merged) and green: **963 passed / 2 skipped**,
`make voice` **95**, `ruff` and `mypy --strict` clean.

**Three branches, two of them unreviewed work that is finished and waiting:**

| branch | PR | what | state |
|---|---|---|---|
| `fix/console-default-domain` | **#2 open** | console opened on a pack with no taxonomy; favicon 404 | green, 966 tests, ready to merge |
| `spike/typesafe-jev-router` | none — evidence, not a change to merge | TypeSafe `jev` evaluation | green, 976 tests |
| `docs/resume-2026-09-20` | this note | — | — |

## Do this first

**1. Revoke the exposed OpenRouter key.** Outstanding since 2026-09-13, across five
sessions. An invalid key returns 401; this one returns 429, so it authenticates.

    openrouter.ai/keys  ->  delete sk-or-v1-0b707...  ->  put the new key in .env

The daily quota is per *account*, so a new key does not reset it. No session has been able
to verify which key `.env` holds — reading it is blocked by the harness, deliberately.

**2. Merge or review PR #2.** Small, green, and it fixes the first thing anyone opening the
console hits.

**3. `.env` now also holds `TYPESAFE_API_KEY`** (added 2026-09-19). Nothing on any call
path uses it; only `scripts/spike_jev_router.py` reads it.

## Quota state

OpenRouter free tier was exhausted on 2026-09-19 (~46 of 50) and resets 00:00 UTC. This is
why **no successful routed conversation has yet been driven through the UI** — everything
either side of the router is verified, the router step itself returns
`ProviderRateLimitedError` and escalates honestly.

## What the numbers say

**The router works.** 30 held-out Bitext rows, `openrouter_alt`
(`inclusionai/ling-3.0-flash-fin:free`), 2026-09-19:

    23/26 exact (88.5%), category 100.0%, 4 unmeasured, p95 3496 ms

Every prediction landed in the right L1 category; all three exact misses are
near-neighbours inside it. The 4 unmeasured were 429s once the cap bit.

**Rule 6 parity is verified with numbers:** 8 cases, agreement **1.0**, 0 divergent,
0 unmeasured. The two variants differ fivefold in latency (12,525 ms vs 2,347 ms p95)
while agreeing on everything.

**The old 52.5% figure is retired — do not quote it.** It was capacity, not comprehension:
every failure was a timeout, nothing was misrouted. ADR-0021 has the arithmetic.

**Still thin: width.** 26 measured rows across 27 intents is about one each, so the
headline is sound and per-intent detail is not (6.8).

## TypeSafe `jev` — what was learned

Branch `spike/typesafe-jev-router`. **Nothing imports it and nothing is bound to it in
`configs/models.yaml`.** Reports in `data/interim/jev_*.json`.

**It is not an LLM.** A "System One" model: *"Code handles deterministic work and owns the
control flow. The model appears only where the system needs programmable common sense."*
It cannot generate text, cannot act autonomously, and returns a probability distribution
over options you define — it cannot invent a value outside the schema. Philosophically
close to Rule 4, which already forbids open-ended agent loops.

Measured on the same 30 rows, same `score_router`:

| | flat (one Choice over 27 leaves) | hierarchical (one Choice per level) |
|---|---|---|
| exact | **90.0 / 93.3 / 93.3%** (three runs) | 83.3% |
| category | 96.7% (all three) | 86.7% |
| p95 | **401–452 ms** | 863 ms |
| API calls | 30 | 57 |

Against the chat baseline's 88.5% / 100% / 3,496 ms. **Latency is the finding — roughly
8× — and accuracy is a wash.** Note jev's category accuracy is slightly *worse*: it made
the only cross-category error in either run (`feedback.complaint -> refund.get_refund`).
Cost is negligible: **$0.0013 per 30-row run**, $0.042/M input tokens, output free.
Limits: 64k context (32k for state), 1,200 req/min.

**The documented shape lost, and the reason is our taxonomy.** The docs prescribe
hierarchical descent for a taxonomy; greedy descent cannot recover an early mistake, and
**this taxonomy's L1 categories are not mutually exclusive** — `delivery` vs `shipping`,
and `order.cancel_order` beside a whole `cancel` category. Three of five errors are exactly
that, including `delivery.delivery_options -> shipping.set_up_shipping_address` and its
mirror. A flat question never commits to a category, so it sidesteps an ambiguity the
hierarchy forces it to resolve first. The docs recommend hierarchy for "thousands of
options"; 27 is not that. **This is worth knowing independently of jev.**

Beam search is the documented fix for greedy and was deliberately not pursued: flat already
wins on accuracy, latency *and* call count, so beam must beat 93.3% at 430 ms while
spending more of both.

### Three things that would bite on adoption

1. **The 0.82 route threshold cannot be carried over.** jev's confidence is a statistic
   over the distribution's *shape*, explicitly not the top probability, and jagged edge 8
   says *"don't transfer thresholds between question formats."* Every policy gating on
   `intent_confidence` needs re-deriving. Not a config swap.
2. **It does not treat state as hostile** (jagged edge 6). Caller utterances are untrusted;
   redaction handles PII and does nothing about injection.
3. **It can never serve `respond`** — *"not trained to generate text."* Router and judge
   only, so the graph would run two vendors.

### Ruled out, not merely untested

**jev cannot be the PII leak detector.** That would mean sending pre-redaction text to a
vendor, which Rule 2 forbids outright. The sandbox blocked the experiment and was right to.
It could only ever confirm *already-redacted* text is clean, which duplicates local
Presidio. Local inference stays.

### Untested and worth testing

The **M6b judge dimensions** as Score — `faithfulness`, `task_success`,
`policy_adherence`. Arguably a better fit than the router: M6b is unbuilt, and there is no
800 ms budget there. `pii_leakage` is excluded by the point above.

**No ADR written.** Adoption is a locked-stack change (Rule 5) and Rule 6 would still
demand a second variant naming a distinct model — hard when the primitive is proprietary.

## What is left

| what | blocked on |
|---|---|
| **Merge PR #2** | a reviewer |
| **M6b** — judge, Ragas/DeepEval; contracts exist in `schemas/eval.py`, runtime does not | nothing |
| **Agent-desktop surface** — `GET /v1/handoffs/{id}`, `WS /v1/ws/copilot/{session_id}` | nothing |
| **A routed conversation through the UI** | OpenRouter quota, or credits, or an Anthropic key |
| **jev judge experiment + threshold study** | nothing — key is in `.env` |
| **Widen the router eval** (6.8) | credits, or two days of free quota |
| **Align the two eval harnesses** (6.6) | the first parity run that actually times out |
| **A real browser test** (9.21) | an ADR — Playwright is outside the locked stack |
| **`docker-compose.yml` never started** (9.20) | a machine with Docker |
| **Label the insurance taxonomy** (9.12) | LLM quota. Would make a second pack routable |

Only `retail` has a mined taxonomy. Healthcare and insurance load and expose tools but
escalate every turn, which is why the console default mattered.

## Verified end to end (2026-09-19, real Chrome)

`cli.demo pipeline` shows **6/8 stages live**; the two pending ones name their phase and
fabricate nothing. Driving the console in Chrome: redaction preview, new session, a typed
turn, all four panels, the tool prober with 9 tools — and **the unredacted PAN never
reaches the DOM**, asserted programmatically. The missing piece is the router step.

## Known-unknown worth stating plainly

Coverage on the mined insurance corpus tops out at **19.7%** and **the cause is not
established**. Three explanations were tested and refuted — call direction, campaign
variety, lexical repetition — and a fourth apparent finding was retracted when its measure
turned out to carry a vocabulary-size artefact. `future-scoped-work.md` 9.17 has the
record. Do not re-propose the outbound-sales explanation; it is refuted with numbers.

## What not to touch

`src/ccas/voice/` is frozen (ADR-0018) and must stay green under `make voice`. The tripwire
at `tests/test_module_5_voice_frozen.py` runs by default and has already caught the core
drifting away from it once.

`domains/` stays at the repo root because packs are data, not code — moving them under
`src/` would break the domain-literal gate that keeps the core honest.
