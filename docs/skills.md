# Project Skills, Rules & Agent Capabilities

How to work in this repository: the binding rules, the tools available at each layer,
and what each agent in the mesh is actually allowed to do.

`CLAUDE.md` is the authoritative rulebook — this document explains and operationalises
it. Where the two disagree, `CLAUDE.md` wins.

---

## 1. The five rules that shape every change

| # | Rule | Enforced by |
|---|---|---|
| 1 | No vertical vocabulary in `src/ccas/` | `tests/security/test_no_domain_literals.py` |
| 2 | No unredacted text crosses a boundary | Contract validators + `tests/security/test_egress_gate.py` + the log processor |
| 3 | 800 ms p95 audio round trip | `configs/latency_budget.yaml` + `tests/latency/` |
| 4 | Deterministic `StateGraph` only | Graph reachability tests in `tests/integration/` |
| 6 | No node depends on a single model | `load_bindings` refusal + `tests/evals/test_provider_parity.py` |

None of these are advisory. Each fails a build.

### What a violation looks like in practice

```
$ make security
FAILED tests/security/test_no_domain_literals.py::test_core_contains_no_domain_vocabulary[refund]
  domain term 'refund' leaked into the core (CLAUDE.md Rule 1).
  Add a DomainPack field instead of a branch.
  src/ccas/schemas/taxonomy.py:130: """...e.g. ``billing.refund.status``."""
```

The fix is always to move the vertical concept into `domains/<pack>/pack.yaml`. Adding
the file to `ALLOWED_PATHS` is a last resort that needs a reason in review.

---

## 2. Working conventions

### Micro-PR loop

One source file plus its test, red → green → refactor. `make check` must be green before
a commit: `ruff format --check`, `ruff check`, `mypy --strict`, `pytest`.

```bash
make check        # the gate
make test-fast    # unit only, no integration
make security     # Rules 1 & 2
make latency      # Rule 3
make evals        # Rule 6 parity gate (Ragas/DeepEval suites land with Module 6)
```

### Test layout

Two layers, deliberately:

| Path | Purpose |
|---|---|
| `tests/test_module_X.py` | **Module gate.** Is this module usable as a whole? One per module; a module is incomplete until its gate is green. |
| `tests/unit/<mirrors src>/` | Granular unit tests, one file per source file. |
| `tests/integration/` | Cross-module flows (ingest→taxonomy, graph e2e, voice loop). |
| `tests/latency/` | Budget assertions. |
| `tests/security/` | Leakage and domain-agnosticism invariants. |
| `tests/evals/` | Judge calibration and variant parity. `test_parity_report.py` exercises the harness offline — feed it known-divergent outcomes and check it says so. `test_provider_parity.py` is the Rule 6 gate: a structural half that runs everywhere, and an `integration` half that calls both variants for real. |

Gate files, in order:

```
tests/test_module_0_foundation.py      schemas · config · llm · observability   [green]
tests/test_module_1_ingestion.py       adapters → CallLog                       [green]
tests/test_module_2_redaction.py       regex · presidio · leak detector         [green]
tests/test_module_3_mining.py          embed → cluster → taxonomy               [green]
tests/test_module_4_orchestration.py   StateGraph · tools · policies            [green]
tests/test_module_5_voice.py           LiveKit · VAD · STT/TTS · barge-in       [green]
tests/test_module_6_copilot_evals.py   handoff · summary · judge · parity       (phase 6)
```

### Testing rules

- **No network in unit tests.** External calls use `httpx.MockTransport`, recorded
  cassettes, or fakes. The vLLM binding is tested over an in-process transport; the
  Anthropic binding's request shaping is a pure function (`build_kwargs`) tested directly.
- A new PII pattern needs a `tests/fixtures/pii_corpus/` entry **and** a Hypothesis
  property.
- A new graph node needs a reachability test and a failure-path test.
- Anything touching the call path needs a `tests/latency/` assertion.
- Coverage floor 85%; 100% on `schemas/` and `redaction/`.
- **Never weaken a test to make a build pass.** Fix the code or escalate.

### Documentation maintenance

These files are maintained, not written once:

| File | Update when |
|---|---|
| `docs/tech-spec.md` | A contract, API shape or budget changes |
| `docs/design-doc.md` | A data flow or subsystem boundary changes |
| `docs/adr/NNNN-*.md` | Any technology choice or major trade-off |
| `docs/future-scoped-work.md` | Anything is deliberately deferred |
| `docs/skills.md` | A rule, convention or agent capability changes |

---

## 3. Tooling by layer

### Contracts — `src/ccas/schemas/`

Pydantic v2. `Frozen` base (immutable, `extra="forbid"`); `SessionState` and
`LatencyLedger` are the only mutable models. No intra-project imports — everything
depends on `schemas`, so a dependency here would be a cycle (asserted in the module gate).

`DropsComputedFields` reconciles `@computed_field` with `extra="forbid"` so models
round-trip through parquet and the wire.

### Config — `src/ccas/config/`

`Settings` (pydantic-settings) is the **only** place environment is read. `budget.py`
loads the latency contract. `domain_loader.py` loads packs and refuses to default on an
unknown one.

### LLM — `src/ccas/llm/`

| Module | Role |
|---|---|
| `base.py` | `LLMProvider` ABC: `complete`, `stream`, `structured`, `healthy` |
| `anthropic_provider.py` | Anthropic binding; `build_kwargs` is the pure shaper |
| `vllm_provider.py` | OpenAI-compatible binding; `build_payload` is the pure shaper |
| `capabilities.py` | Per-model parameter gating — the API *rejects* unsupported params |
| `bindings.py` | `models.yaml` → `ModelBinding`; refuses single-provider nodes |

Model ids live in `configs/models.yaml`. Hard-coding one elsewhere is a defect.

### Observability — `src/ccas/observability/`

`tracing.current_trace_context()` snapshots the active span into a serializable
`TraceContext` that rides on every boundary-crossing payload.

`logging.configure_logging()` writes JSON lines to `logs/execution.log`. The
`_forbid_raw_content` processor **raises** on any event carrying `utterance`,
`transcript`, `raw_text`, `digits`, `ani`, `slot_value` and friends. Log entity types and
counts, never content.

```python
log.info("redaction.done", entity_counts={"person": 2, "phone": 1}, elapsed_us=840)  # ok
log.info("turn.received", utterance=text)                                            # raises
```

### Redaction — `src/ccas/redaction/`

Two modes, because the difference was measured (ADR-0007):

| Mode | Engines | p99 | Where |
|---|---|---|---|
| `REALTIME` | regex only | **111 µs** | the live call path, per utterance |
| `BATCH` | regex + Presidio NER | **7.0 ms** | ingestion, mining, storage, handoff |

`REALTIME` does not consider Presidio available at all, so a missing model cannot fail a
live turn. `BATCH` fails closed: no model, no egress. **Anything that is stored, mined,
or shown to a human must use `BATCH`** — `Normalizer` refuses a realtime pipeline outright.

| Module | Role |
|---|---|
| `engine.py` | `RedactionEngine` ABC + `merge_spans` overlap resolution |
| `regex_engine.py` | the hot path; `patterns/generic.yaml` + pack patterns layered on |
| `presidio_onnx.py` | local NER; allow-list for protocol vocabulary; fails closed |
| `leak_detector.py` | post-substitution re-scan; a hit forces `DIRTY` |
| `placeholder.py` | stable `[PERSON_1]` tokens, keyed by hash so no surface is retained |
| `validators.py` | `luhn`, `aba_routing`, `iban_mod97` — what makes a broad pattern precise |

### Ingestion — `src/ccas/ingestion/`

`RawRecord` is the one place unredacted text legitimately exists. Its `__repr__` is
blind, so a traceback or a debugger watch cannot leak it. `Normalizer` redacts, then
builds a `CallLog`; a record that does not come back CLEAN is **quarantined** — counted,
its id and reason recorded, and dropped. Never emitted in a weaker form.

### Mining — `src/ccas/mining/`

| Module | Role |
|---|---|
| `embedder.py` | `SentenceTransformerEmbedder` (real) and `HashingEmbedder` (CI); `EmbeddingCache` is content-addressed and model-scoped |
| `cluster.py` | HDBSCAN; its *refusal to assign* is the property k-means cannot give |
| `labeler.py` | LLM names clusters into L1/L2/L3 via strict structured output |
| `feasibility.py` | volume × complexity → `Quadrant` |
| `build_taxonomy.py` | orchestrates; merges duplicate leaves, rolls up parent volumes |

Three signal filters (ADR-0008): caller turns only, ≥8 characters, and the labeler's
`is_intent` flag. Without the third, the closing line is the taxonomy's biggest node.

`build_embedder` **refuses** to fall back to the hashing encoder unless
`--allow-hashing-embedder` is passed — a taxonomy silently mined from lexical overlap
would look plausible and describe nothing.

### Orchestration — `src/ccas/{tools,policies,nodes,graph}/`

| Package | Role |
|---|---|
| `tools/` | registry (global + pack, no shadowing), executor gate chain, mock backend |
| `policies/` | four pure policies + ordered engine; first decisive verdict wins |
| `nodes/` | greet · identify · route · clarify · slot_fill · tool_exec · respond · escalate · close |
| `graph/` | `StateGraph` assembly, front-door router, grounded responder, checkpointing |

One graph invocation handles **one caller turn**. It ends when the platform has spoken
and is waiting for a reply, which is what makes the turn ceiling and the latency ledger
mean anything. Every edge is a pure function of state, so reachability and termination
are read off the routing table rather than sampled.

The session `PlaceholderVault` (ADR-0010) is how a tool receives the reference a caller
read out while the model only ever sees `[ACCOUNT_REF_1]`. It is never a model field.

### Prompts — `src/ccas/llm/prompt.py`

`LLMRequest` only accepts `RedactedText`, so prompts are built through two narrow helpers
rather than ad-hoc bypasses:

- `authored(text)` — repository-authored strings. **Never** pass a runtime value.
- `compose(template, parts)` — interpolates already-redacted parts and inherits the
  *weakest* status among them, so a leak upstream cannot be laundered through an f-string.

### Datasets — `src/ccas/ingestion/datasets.py`

`require_role(source, role)` gates a pipeline stage on corpus suitability. See
`docs/adr/0004-dataset-role-separation.md` and `data/*/README.md`.

---

## 4. The demo console

`src/cli/demo.py` is the manual-review surface. Every run appends structured events to
`logs/execution.log`.

```bash
uv run python -m cli.demo pipeline "where is my delivery"   # full stage walk
uv run python -m cli.demo pipeline                          # interactive
uv run python -m cli.demo pack healthcare                   # inspect a pack
uv run python -m cli.demo datasets                          # role matrix + live refusal
uv run python -m cli.demo budget                            # budget + model bindings
uv run python -m cli.demo gate                              # watch Rule 2 refuse

uv run python scripts/fetch_datasets.py --all                       # hydrate data/raw/
uv run python scripts/fetch_datasets.py --corpus aixblock --include-large

uv run python scripts/ingest.py --source synthetic --domain retail --limit 50
# AIxBlock ships one JSONL per vertical, so --path is required or both are pooled:
uv run python scripts/ingest.py --source aixblock --domain retail \
  --path data/raw/aixblock/retail.jsonl

uv run python scripts/mine_taxonomy.py --domain retail --dry-run   # cluster, no LLM
uv run python scripts/mine_taxonomy.py --domain retail
```

Stages are `[LIVE]` or `[PHASE n]`. **A pending stage never fabricates output** — it
reports which phase delivers it and what it will produce. A demo that invented a
redaction it had not performed would be worse than no demo.

Add a stage to `src/cli/stages.py` as each module lands; flip it from `_pending(...)` to
a real call. `tests/unit/cli/test_demo.py` asserts every pending stage names a phase and
that no run writes the input transcript to the log.

---

### Voice — `src/ccas/voice/`

| Module | Role |
|---|---|
| `audio.py` | PCM frames. Pre-STT, therefore pre-redaction — never a model field, never logged |
| `transport.py` | `AudioTransport`: receive, send, **clear** (the barge-in primitive), dtmf |
| `drivers/` | `scripted` (deterministic, CI) · `livekit` (WebRTC, unverified) |
| `vad.py` | `EnergyVad` (no model) and `SileroVad`, sharing one hangover state machine |
| `stt/`, `tts/` | scripted stand-ins + Deepgram Nova-3 / Cartesia Sonic adapters |
| `dtmf.py` | keypad collection: terminator, max length, inter-digit timeout |
| `bargein.py` | clear playback → cancel generation → cancel turn, in that order |
| `session.py` | the turn loop, and the only place the ledger is assembled |
| `worker.py` | LiveKit entrypoint; **refuses to start** when preflight objects |

One graph invocation per caller turn. **Playback runs as a task** — awaiting it inline
makes barge-in structurally impossible (ADR-0011).

```bash
uv run python -m ccas.voice.worker preflight   # what is missing before a call can land
uv run python scripts/bench_latency.py         # per-stage distribution vs budget
```

## 4b. The workbench

`make workbench` serves an interactive console at `http://127.0.0.1:8000`. Type an
utterance and watch redaction, routing, every policy verdict, tool dispatch and the
latency ledger for that turn.

Two probes work with **no LLM provider configured**, which is most of what you want when
developing a pack:

| Probe | Answers |
|---|---|
| Redaction preview | exactly what a model would receive, and what was removed |
| Tool prober | one tool through the full gate chain — schema, authorisation, timeout, output redaction |

With no provider the session still opens; routing escalates honestly rather than
pretending to understand. Same principle as a pending demo stage: never fabricate.

It exposes session state, so it binds to loopback and is not a production endpoint.

## 5. Agent capabilities in the mesh

Every agent is a node (or subgraph) in a deterministic `StateGraph`. None of them are
free-running loops; the table is what each is *permitted* to do.

| Agent | Model binding | May do | May not do |
|---|---|---|---|
| **Front-door router** | `router` — Haiku 4.5 / Llama 3.3 70B | Classify into one taxonomy intent, emit calibrated confidence + alternatives | Call tools, make promises, answer questions |
| **Identify node** | none (deterministic) | Run the verification ladder, request step-up | Accept unverified identity for a guarded tool |
| **Clarify node** | `router` | Ask one disambiguating question, bounded by `max_clarifications` | Read a menu; loop unbounded |
| **Slot-fill node** | `task_agent` | Elicit and validate pack-declared slots, accept DTMF | Invent a slot not in the taxonomy |
| **Domain micro-agent** | `task_agent` — Opus 5 / Llama 3.3 70B | Choose among pack-registered tools, reason over results | Construct a call; answer beyond tool data |
| **Tool executor** | none (deterministic) | Validate args, enforce timeout/retry/idempotency, redact output | Dispatch without passing `authorized_by(spec)` |
| **Escalate node** | `summarizer` | Build a `HandoffContext` with a redacted summary | Emit a payload with any unredacted field |
| **Async judge** | `judge` — Haiku 4.5 / Llama 3.1 8B | Score faithfulness, PII leakage, task success, policy adherence on a 5% sample | Run on the call path |
| **Taxonomy labeler** | `taxonomy_labeler` | Name cluster centroids into L1/L2/L3 with slots | Run online; see unredacted text |

### Standing constraints on every agent

1. Receives `RedactedText` only. There is no code path that hands one raw input.
2. Chooses *which* declared tool, never *what* call.
3. Grounded answers only. Missing data escalates — it is never estimated.
4. Bounded by the pack's `max_turns` and its intent's `EscalationPolicy`.
5. Every node has a reachable escalation edge.

---

## 6. Adding a new vertical

No `src/` change is required, and the build will tell you if you made one.

1. `domains/<name>/pack.yaml` — queues, tools, prompts, thresholds, patterns, compliance.
2. Ingest a corpus for the vertical → `data/interim/`.
3. `uv run scripts/mine_taxonomy.py --domain <name>` → `domains/<name>/taxonomy.json`.
4. `uv run python -m cli.demo pack <name>` to eyeball it.
5. `uv run pytest --domain <name>` — the full Phase 4/5 suite must pass unchanged.
6. If step 5 needs a `src/` edit, that is the Rule 1 leak. Fix the core, not the test.

Three packs ship today: `retail`, `healthcare`, and `insurance`. The third was created
*from* a corpus rather than from a design sketch — the AIxBlock files turned out to hold
insurance sales calls whatever their filenames claimed (ADR-0013) — which is the order
this step list is really meant to run in. Profile the corpus before you write the pack.

Two tests keep the pack list honest and neither hard-codes it:
`test_every_pack_on_disk_is_discoverable` walks `domains/`, and
`test_no_committed_taxonomy_was_mined_by_a_stub` refuses a taxonomy whose provenance
names the hashing embedder or the stub labeler.

### Two ways a pack gets a taxonomy

**Mined** — `scripts/mine_taxonomy.py`, for an unlabelled corpus. Discovers what callers
actually said; covers them only as well as the clustering did.

**Adopted** — `scripts/adopt_taxonomy.py`, for a corpus that publishes its own labels. No
clustering happens, so Rule 9's ban on mining a labelled corpus does not apply — but four
things must hold, and `TaxonomyProvenance` exists so a reader can tell the two apart at a
glance (ADR-0020):

- The artefact declares `provenance: adopted` and names `adopted_from`.
- **Half the corpus is held out.** `derivation_split` records which half defined the
  labels, because grading a router on the rows that defined its label set measures nothing.
- The corpus must hold `NLU_GROUND_TRUTH`, not `INTENT_MINING`.
- `AutomationScore.confidence` stays low. The intents are real; their fit to *this*
  deployment's traffic is unverified.

A slot is **what you ask the caller for**. Scan the caller's side of a corpus only —
scanning the agent's side of Bitext turned response-template variables (`website_url`,
`first_step`) into 47 "slots" one intent would have interrogated callers for. And declare
which slots hold identifiers in the **pack** (`slot_pii`), never in `src/ccas/`: the core
may not know what a vertical calls things (Rule 1).

### Before a long run, check the machine

`ccas.mining.preflight` refuses to load an encoder the machine cannot hold. It exists
because the failure it prevents is invisible: a process that cannot get memory does not
raise, it sits in uninterruptible sleep, and twice on this project a mining run stalled for
half an hour at 0% CPU having produced no output at all.

- Resident sizes are declared per model in `MODEL_RESIDENT_MB`. Add a row before using a new
  encoder; an unknown model is *not* checked rather than checked against a guess.
- `require_memory()` is pure and takes the reading; `check_headroom()` is the edge that
  takes it. I/O at the edges (Rule 8).
- Unknown memory proceeds. A guard against stalling must not become a reason a run cannot
  start, and this is a performance guard — Rule 2's fail-closed rule is about egress.
- It does not help with the clustering phase, which is O(n·d) and cheap. **The encoder is
  the memory cliff and no sampling avoids it** — every run pays the resident set once.

### Mining a corpus that is not really diarized

`--min-chars` exists for this. AIxBlock publishes *pause-segmented* turns: one spoken
sentence arrives as several fragments, 182 of them per call, and the default 8-character
floor admits all of them. Clustering fragments produces clusters of fragments. Profile the
word-length distribution first, then set the floor where utterances start carrying a
reason for contact — 60 characters for AIxBlock. This is a knob because it is corpus
shape, not policy; `docs/future-scoped-work.md` 9.9 tracks removing it.

Speaker labels deserve the same suspicion. `AixBlockAdapter` records
`metadata["diarization_reliable"]` and does not try to repair a bad one: re-diarizing from
text would put invented structure into a corpus the rest of the system treats as observed.
Where the label is useless, the agent-speech filter falls to the labeler prompt instead.
