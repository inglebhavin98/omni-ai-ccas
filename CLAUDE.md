# CLAUDE.md

Governance rulebook for `omni-ai-ccas`. These rules are binding. When a request conflicts
with a rule here, say so and stop — do not silently comply.

## Project

`omni-ai-ccas` — a domain-agnostic, AI-native Customer Care Experience Platform built to
fully replace legacy IVR/CCaaS stacks (Genesys, Cisco). The same core serves retail,
healthcare, banking, telecom, utilities, and SaaS by loading a different domain pack.

## Setup

```bash
uv sync --all-extras          # install (uv is the ONLY package manager)
cp .env.example .env          # set OPENROUTER_API_KEY; the rest have working defaults
make workbench                # chat console at http://127.0.0.1:8000
```

**The chat channel needs nothing else** — no Docker, no vector store, no vendor
credentials. It runs against the mock tool backend with one LLM key.

```bash
docker compose up -d                  # qdrant · redis · otel-collector, when you want them
docker compose --profile voice up -d  # adds livekit-server (frozen channel, ADR-0018)
```

Every `make` target runs `uv run --frozen`. Without `--frozen`, uv re-resolves the
`en-core-web-sm` direct URL on every invocation, and a slow fetch kills the command
*before* it runs anything — which looks like a hang, not a network error.

## Common commands

```bash
make check        # ruff format --check && ruff check && mypy --strict && pytest   <- must be green to commit
make test         # pytest -q
make test-fast    # pytest -m "not slow and not integration"
make latency      # pytest tests/latency -q          (budget gate)
make security     # pytest tests/security -q         (leakage + domain-leak gate)
make voice        # the frozen voice channel (ADR-0018); excluded from `make test`
make evals        # provider parity (Rule 6); Ragas/DeepEval land with Module 6
uv run scripts/ingest.py --source bitext --domain retail
uv run scripts/mine_taxonomy.py --domain retail
uv run python -m ccas.voice.worker preflight  # what blocks answering a call
uv run python -m ccas.voice.worker dev      # LiveKit agent
uv run python scripts/bench_latency.py      # per-stage latency vs budget
uv run python -m cli.demo pipeline "..."    # manual review console
uv run python -m cli.demo gate              # watch the Rule 2 egress gate refuse
make workbench                              # interactive console, 127.0.0.1:8000
```

The workbench exposes session state. It binds to loopback and is a development
surface, never a production endpoint.

## Architecture

Six modules over one frozen contract layer. `src/ccas/schemas/` has no intra-project
imports; everything depends on it and nothing depends on anything above it.

| Module | Package | Responsibility |
|---|---|---|
| M1 | `ingestion/` | source adapters -> redactor -> `CallLog` |
| M2 | `redaction/` | regex + Presidio(ONNX) + leak detector (library, not a stage) |
| M3 | `mining/` | embed -> HDBSCAN -> LLM label -> `IntentTaxonomy` |
| M4 | `tools/`, `policies/`, `nodes/`, `graph/` | tool registry + executor, pure policies, LangGraph `StateGraph` over `SessionState` |
| M5 | `voice/` | LiveKit WebRTC · Silero VAD · STT · TTS · barge-in · DTMF |
| M6 | `copilot/`, `evals/` | CTI handoff, CRM summary, Ragas/DeepEval, async judge |
| — | `api/` | workbench API + console (development surface, loopback only) |

Cross-cutting: `llm/` (provider abstraction), `config/` (settings + domain packs),
`observability/` (OTel, JSON logs, audit). `src/cli/` is the manual-review console —
an operator affordance; the platform must never import it.

Full detail: `docs/design-doc.md` (flows), `docs/tech-spec.md` (shapes and budgets),
`docs/skills.md` (conventions and agent capabilities), `docs/adr/` (why).

## Rule 1 — Domain-agnostic core (hard)

- No domain vocabulary in `src/ccas/`. Banned as literals: `claim`, `policy`, `order`,
  `refund`, `member_id`, `patient`, `diagnosis`, `sku`, `premium`, `deductible`, and
  anything like them. `tests/security/test_no_domain_literals.py` enforces this.
- Intents are `Slug` strings. Slots, tools, queues, prompts, thresholds, and redaction
  patterns are **data** loaded from `domains/<pack>/`.
- Never add a domain branch (`if domain == "healthcare"`). Add a pack field instead.
- A feature is complete only when it works for both `retail` and `healthcare` packs.

## Rule 2 — Zero-leakage PII/PHI (hard)

- **No unredacted caller text, audio, or tool output reaches a model, a log, a stored
  record, or an external vendor** — LLM provider, TTS vendor, CRM, log sink,
  telemetry, or crash report. No exceptions for "just debugging".
- A **registered tool backend is none of those**: it is an internal system of record
  that already holds the caller's data, and reaching it is the point of the call.
  `ToolExecutor` restores placeholders from the session `PlaceholderVault`
  immediately before dispatch, and nowhere else. The vault is never a field on a
  Pydantic model — see `docs/adr/0010-placeholder-vault.md`.
- Only `RedactedText` may be passed to `LLMProvider`. `CallLog` and `HandoffContext`
  reject construction unless `RedactionReport.status is CLEAN`. Do not weaken these
  validators; do not add a bypass flag outside `tests/`.
- If the redaction engine is unavailable the status is `UNVERIFIED` and egress is
  **forbidden** — fail the turn and escalate. Never fail open.
- Never log raw utterances, DTMF digit runs, `caller_ref` pre-hash, or span match text.
  Log entity *types* and counts only.
- Redact by *context*, not only by shape. A keypad answer to a PII slot is replaced
  wholesale (`redact_value`) — six bare digits match no pattern, and the caller was
  asked for a reference and gave one.
- Never commit real customer data. `tests/fixtures/` is synthetic or public-dataset only.
- Raw audio lives in encrypted object storage referenced by URI; it never enters a
  Pydantic model, a log line, or a prompt.

## Rule 3 — Latency budget (hard)

- End-to-end audio round trip: **<= 800 ms p95**. Per-stage budgets in
  `configs/latency_budget.yaml` are the contract; `tests/latency/` fails the build on
  breach. Raising a budget is an ADR, not a commit.
- Stream everywhere: STT partials, LLM token stream piped directly into TTS. Never
  buffer a full LLM response before speaking.
- Every external call carries an explicit timeout. No unbounded awaits.
- Blocking I/O or CPU work in an async path is a defect. Offload to a thread/process.
- Prompt-cache the stable prefix (system + tool defs + pack prompts); keep volatile
  content after the last cache breakpoint. Verify with `usage.cache_read_input_tokens`.

## Rule 4 — Deterministic orchestration (hard)

- LangGraph `StateGraph` only: explicit nodes, explicit conditional edges, explicit
  terminal states. No `AgentExecutor`, no open-ended ReAct loop, no auto-chaining.
- The LLM chooses **which** declared tool to call; it never constructs a URL, a query,
  or an unregistered call. All tools come from the pack registry.
- Every graph must have a reachable escalation path and a turn ceiling. An unbounded
  loop is a defect.
- Answers are grounded in tool results or retrieval. When data is missing, say so and
  escalate — never improvise a factual claim about the caller's account.

## Rule 5 — Tech stack (locked; changing any row needs an ADR)

| Concern | Locked choice |
|---|---|
| Language / runtime | Python 3.12+, `uv` (never pip/poetry/conda) |
| Contracts | Pydantic v2 (`Frozen` base; `SessionState` is the only mutable model) |
| Embeddings | sentence-transformers (`bge-large-en-v1.5` default) |
| Clustering | HDBSCAN primary, BERTopic alternate |
| PII | Microsoft Presidio (ONNX, local) + regex NER — **local inference only** |
| Orchestration | LangGraph `StateGraph` |
| LLM | `LLMProvider` protocol. **OpenRouter** is the configured provider (free models); Anthropic and vLLM bindings remain available. `structured_mode` is declared per model — most free models ignore `response_format` (ADR-0012). **Free models serve the offline and chat paths only; the voice call path needs a low-latency provider** |
| Voice | **FROZEN — ADR-0018.** LiveKit Agents (WebRTC), Silero VAD, Deepgram Nova-3 STT, Cartesia Sonic TTS, local Whisper/Piper fallbacks. Code and tests remain and must keep passing under `make voice`; voice is not a live constraint on core work, and its 800 ms budget no longer gates model selection |
| Evals | Ragas + DeepEval offline; async LLM-as-judge on a 5% production sample |
| Observability | OpenTelemetry; hash-chained immutable audit log |
| Vector / cache | Qdrant; Redis for semantic cache |
| Lint / types / test | ruff, mypy `--strict`, pytest (+ Hypothesis for redaction) |

Hard-coding a model ID outside `configs/models.yaml` is a defect.

## Rule 6 — No node depends on a single model

Every node in `configs/models.yaml` declares **at least two variants naming distinct
models**, and `load_bindings` refuses anything less. A variant is usually a provider;
when every model sits behind one gateway it is a second model, which is the more useful
comparison there anyway.

Any change to a prompt, tool schema, or graph node must pass
`tests/evals/test_provider_parity.py` (`make evals`). If two variants diverge beyond
threshold, fix it or record the divergence in an ADR — do not quietly pin to one.

A case nobody served — a 429, a quota — is **unmeasured**, not divergent, and leaves the
denominator. A case that *failed* is divergent. A run with too few measurements skips
with the count in the reason; it never passes (ADR-0014). Zero measured cases is 0%
agreement, not 100%.

`structured_mode` is declared per model, not per provider. Most free models ignore
`response_format` entirely; getting this wrong fails silently and far away (ADR-0012).

## Rule 7 — TDD (hard)

- Test first. One micro-PR = one source file + its test. Red -> green -> refactor.
- No network in unit tests. External calls use recorded cassettes or fakes.
- New PII pattern => a `tests/fixtures/pii_corpus/` entry **and** a Hypothesis property.
- New graph node => reachability test + at least one failure-path test.
- New latency-touching code => a `tests/latency/` assertion.
- Coverage floor 85% on `src/ccas/`; 100% on `schemas/` and `redaction/`.
- Never weaken or skip a test to make a build pass. Fix the code or escalate.

## Rule 8 — Coding standards

- `from __future__ import annotations`; full type hints; mypy `--strict` clean.
- No bare `except`. No silent `pass` on exception. No `# type: ignore` without a reason.
- Async-first for anything on the call path. `httpx.AsyncClient`, pooled and reused.
- Structured logging only (`structlog`), never `print`. Every log line carries
  `correlation_id`.
- Pure functions in `mining/`, `redaction/`, `policies/` — I/O at the edges. A policy
  receives state and thresholds, never the pack itself.
- Config via `pydantic-settings`; no `os.environ` reads scattered through modules.
- Module <= 400 lines, function <= 50. Split rather than nest.
- Comments explain *why*. Match surrounding style; no decorative banners.
- Public API changes to `schemas/` require a `schema_version` bump + migration note.

## Rule 9 — Dataset roles are fixed (hard)

The three corpora are not interchangeable and must never be pooled:

| `data/raw/` | Use it for | Never use it for |
|---|---|---|
| `aixblock` | raw transcript ingestion, PII redaction corpus, **unsupervised HDBSCAN intent mining** | any ground-truth or accuracy metric (it is unlabelled) |
| `bitext` | **ground-truth NLU intents**, tool-parameter extraction targets, offline LLM-as-a-Judge evaluation | **intent mining** — clustering a labelled set rediscovers its own labels and proves nothing |
| `natcs` | **ground-truth NLU intents in natural phrasing** (82 intents, human speech — the harder grader next to Bitext's templates), LLM-as-a-Judge evaluation, transcript ingestion | intent mining (labelled), tool-parameter extraction (unannotated), **trajectory replay — its turns are sampled, not consecutive** (ADR-0015) |

- Gate every corpus-consuming stage with `require_role(source, role)` from
  `ccas.ingestion.datasets`. It raises `DatasetRoleError` at the point of the mistake.
- Adding a role to a corpus means editing `DATASET_ROLES` and its test. That friction is
  deliberate — see `docs/adr/0004-dataset-role-separation.md`, amended by
  `docs/adr/0015-natcs-is-supervision-not-a-trajectory.md`.
- **Verify a corpus before trusting its description.** Both role assignments that have
  been wrong so far were wrong because the dataset card said one thing and the bytes said
  another (ADR-0013, ADR-0015). Profile it first.
- `DIALOGUE_CONTEXT_BENCHMARK` and `STATE_GRAPH_BENCHMARK` are declared and **unsourced**.
  No corpus on disk supports them. Do not grant one either role to unblock a benchmark.
- Raw corpora are inputs to Module 2, never to a model. Only `data/interim/` is safe
  downstream, because a `CallLog` cannot be built from unredacted text.

## Rule 10 — Documentation is maintained, not written once

These files are part of the definition of done:

| File | Update when |
|---|---|
| `docs/skills.md` | a rule, convention, tool or agent capability changes |
| `docs/tech-spec.md` | a contract, API shape or latency budget changes |
| `docs/design-doc.md` | a data flow or subsystem boundary changes |
| `docs/adr/NNNN-*.md` | **any** technology choice or major trade-off — write it in the same change |
| `docs/future-scoped-work.md` | anything is deliberately deferred — record it in the commit that defers it |

ADRs are numbered sequentially and never renumbered once merged; a reversal supersedes
by reference rather than editing history. Do not write an ADR for a refactor or a bug
fix, or for a choice with no live alternative.

## Rule 11 — Every module ships visibility (hard)

A module is not complete until all three exist:

1. **A gate test** at `tests/test_module_<N>_<name>.py` asserting the module is usable as
   a whole. Granular tests still live in `tests/unit/` mirroring `src/ccas/`. The gate
   must be green before the module is called done.
2. **A demo stage** in `src/cli/stages.py`, flipped from `_pending(...)` to a real call,
   so `uv run python -m cli.demo pipeline "<transcript>"` shows the module working under
   human review. **A pending stage names the phase that delivers it and never fabricates
   output** — a demo that invented a redaction it had not performed would be worse than
   no demo.
3. **Structured JSON logs** to `logs/execution.log` via `ccas.observability.logging`,
   with `correlation_id` on every event. Log entity types and counts, never content;
   `_forbid_raw_content` raises on `utterance`, `transcript`, `raw_text`, `digits`,
   `ani`, `slot_value` and friends.

## Conventions

- Intent ids: dotted lowercase slugs, depth = level — `billing.refund.status`.
- Tool names: `verb_noun` snake_case — `get_order_status`. Shared tools live in
  `configs/tools.yaml` with `domain: core`; a pack adds its own and may not shadow one.
- Slots: snake_case, pack-declared. Placeholders: `[ENTITY_TYPE_N]`, stable per session.
- Files: snake_case; classes: PascalCase; one primary export per module.
- Branches: `phase<N>/<module>-<slug>`. Commits: Conventional Commits.
- ADRs in `docs/adr/NNNN-title.md` for every locked-row change or budget change.

## Do not

- Do not call an external LLM/TTS/STT with unredacted text.
- Do not add a domain `if`/`elif` in `src/ccas/`.
- Do not introduce an unconstrained agent loop.
- Do not raise a latency budget to make a test pass.
- Do not commit real customer data, recordings, or credentials.
- Do not add a dependency outside the locked stack without an ADR.
- Do not mine a taxonomy from a labelled corpus, or grade one against the corpus it
  was mined from.
- Do not log an utterance, a slot value, a DTMF run, or a pre-hash caller reference.
- Do not mark a module done without its gate test, demo stage and structured logs.
- Do not let a demo stage fabricate output for a module that is not built.
- Do not return tool output a model can read without passing it through the executor's
  redaction step; `pii_output_fields` alone is not sufficient.
- Do not put the placeholder vault, or any detokenized value, on a Pydantic model.
- Do not declare an intent whose tools require arguments its slots do not cover;
  `load_taxonomy` rejects it, and the call would otherwise die on a schema violation.
