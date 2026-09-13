# omni-ai-ccas — Design Document

Architectural overview, data flows and subsystem integration.

Companion documents: `docs/tech-spec.md` (shapes and budgets), `docs/skills.md` (rules
and conventions), `docs/adr/` (why each choice was made), `CLAUDE.md` (binding rules).

---

## 1. What this system replaces

A legacy IVR/CCaaS stack is five separable things wearing one badge: telephony ingress,
DTMF-tree routing, scripted resolution, an agent desktop, and 1–2% manual QA sampling.
Replacing "the IVR" alone leaves four of them, which is why IVR-only modernisations
stall.

| Layer | Legacy | omni-ai-ccas |
|---|---|---|
| Ingress | TDM / hard-coded SIP trunks | WebRTC now, programmable SIP later (ADR-0001) |
| Routing | DTMF tree, rules-based ACD | Intent-aware concierge over a mined taxonomy |
| Resolution | Scripted VXML + escalation | Deterministic `StateGraph` with tool-calling micro-agents |
| Desktop | Vendor workspace | Web-native copilot with zero-touch wrap-up |
| QA | 1–2% manual sampling | Async LLM-judge on a 5% sample, 100% structured audit |

The strategic point: **the taxonomy is the asset**. Legacy call data is mined once into
an explicit L1/L2/L3 hierarchy with volume and feasibility scores, and that artifact then
drives routing, agent design, migration sequencing and evaluation. Everything downstream
is generated from it.

---

## 2. Design principles

### 2.1 The core knows nothing about any vertical

A vertical is a `DomainPack` — a YAML file plus a mined taxonomy. Intents are strings.
Slots, tools, queues, prompts, thresholds and redaction patterns are data. There is no
`if domain ==` anywhere in `src/ccas/`, and a test scans for vertical vocabulary and
fails the build on a hit.

Two packs ship from Phase 1 rather than one, because a single pack cannot demonstrate
the absence of hidden assumptions. (ADR-0006)

### 2.2 Invariants live in types, not in review attention

"Redact before sending to the model" holds for about six weeks. `RedactedText` holds
forever: it is the only string type that can cross a boundary, it cannot exist without a
`RedactionReport`, and `CallLog`, `HandoffContext` and `LLMRequest` refuse construction
without one. There is no unredacted instance to leak. (ADR-0005)

The same move applies elsewhere: the latency budget is a field on `SessionState`, not a
convention; tool schemas must be closed, so a model cannot smuggle arguments; the log
processor raises on raw-content keys rather than trusting the caller.

### 2.3 Determinism where the caller can hear it

Voice has no undo. Orchestration is an explicit graph with bounded loops and a reachable
escalation edge from every node. The model picks *which* declared tool; it never
constructs a call. (ADR-0002)

### 2.4 Two of everything that matters

Two LLM providers, both live and parity-tested. Two domain packs. Two transport drivers
(LiveKit and a deterministic mock). Any abstraction with one implementation is a guess.

---

## 3. System architecture

```
                     ┌──────────────────────────────────────────┐
  OFFLINE / BATCH    │  domains/<pack>/pack.yaml                │
  ─────────────      │  taxonomy · slots · tools · policies ·   │
                     │  redaction patterns · queues · compliance│
                     └────────────────────┬─────────────────────┘
  data/raw/aixblock                       │ loaded at runtime
  (mining corpus)                         │
          │                               │
          ▼                               │
  ┌────────────────────────────┐          │
  │ M1  Ingestion & Normalizer │          │
  │   SourceAdapter protocol   │          │
  └─────────────┬──────────────┘          │
                │ RawRecord               │
                ▼                         │
  ┌────────────────────────────┐          │   ⚠ the gate sits INSIDE ingestion:
  │ M2  Zero-Leakage Redactor  │          │     an unredacted CallLog is not a
  │   regex → presidio(onnx)   │          │     valid object at any point
  │   → leak detector (assert) │          │
  └─────────────┬──────────────┘          │
                │ CallLog (CLEAN)         │
                ▼                         │
  ┌────────────────────────────┐          │
  │ M3  Intent Miner           │          │
  │   embed → HDBSCAN →        │          │
  │   LLM label → slots → 2×2  │          │
  └─────────────┬──────────────┘          │
                │ IntentTaxonomy ─────────┘  written back into the pack
                │
                │        data/raw/bitext ──► grades the taxonomy (never mines it)
                │
  ══════════════╪═══════════════════════════════════════════════════════
  ONLINE / REALTIME
                ▼
  [Caller] ──WebRTC──► ┌───────────────────────────────────────┐
                       │ M5 Voice Engine (LiveKit worker)      │
                       │  Silero VAD · barge-in · DTMF         │
                       │  STT: Deepgram Nova-3 | Whisper       │
                       │  TTS: Cartesia Sonic | Piper          │
                       └───────┬───────────────────▲───────────┘
                      text ▼   │                   │ token stream
                       ┌───────┴───────────────────┴───────────┐
                       │ M2′ Realtime Redactor (per-utterance) │
                       │     same engine · 3 ms budget         │
                       └───────┬───────────────────────────────┘
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ M4  LangGraph Agentic Mesh        (SessionState)                │
  │                                                                 │
  │   greet → identify → ROUTE ─┬─► clarify ──┐                     │
  │                             │             ▼                     │
  │                             ├─► slot_fill → tool_exec → respond │
  │                             │             │          │          │
  │                             └─► escalate ◄┴──────────┘          │
  │                                    │                            │
  │   policies: confidence · sentiment · risk_tier · retry          │
  └────────────────────────────────────┬────────────────────────────┘
                    ToolPayload ▼      │ HandoffContext
         ┌──────────────────────┐      ▼
         │ Tool Registry /      │   ┌─────────────────────────────┐
         │ Enterprise Adapters  │   │ M6a Copilot (CTI/CRM)       │
         │ (mock ↔ real)        │   │  live transcript · summary  │
         └──────────────────────┘   └─────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────┐
                    │ M6b Evals: Ragas/DeepEval offline ·        │
                    │  async LLM-judge (5% sample) ·             │
                    │  variant parity (model ↔ model, ADR-0012)  │
                    └───────────────────────┬────────────────────┘
                                            │  data/raw/natcs ──► NLU ground truth
                    ┌───────────────────────▼────────────────────┐
                    │ OpenTelemetry spans · logs/execution.log   │
                    │ hash-chained immutable audit log           │
                    └────────────────────────────────────────────┘
```

### Four refinements over the original sketch

1. **Redaction is a stage inside ingestion, not a sibling of it.** As peers, there is a
   window where an unredacted `CallLog` is valid. As a validator, there is not.
2. **Realtime redaction was missing entirely.** The original design redacted the dataset
   but not the live call. Every STT final crossing into a provider passes the same
   engine, inside a 3 ms slice of the budget.
3. **The domain pack is an explicit input to both paths.** This is what makes
   domain-agnosticism testable rather than aspirational.
4. **Provider parity is a standing eval.** With two bindings shipped, unmeasured drift
   silently degrades the hybrid to whichever one was configured last.

---

## 4. Data flows

### 4.1 Offline — corpus to taxonomy

```
Hugging Face Hub
   │  scripts/fetch_datasets.py   streams the published archives, recovers turns
   │  from inter-word pauses (the corpus carries no diarization -- ADR-0016)
   ▼
data/raw/aixblock/<domain>.jsonl                     ← unredacted, never a model input
   │  require_role(AIXBLOCK, INTENT_MINING)          ← refuses the wrong corpus
   ▼
SourceAdapter.read()      → RawRecord
   ▼
redaction.pipeline.redact_many()  → RedactedText per utterance   [BATCH mode]
   │     regex (hot path) → presidio_onnx (names/places) → leak_detector (re-scan)
   │     any residual hit ⇒ DIRTY ⇒ the record is quarantined, never emitted
   │     one allocator per record: coreference holds inside a call, tokens never
   │     collide across calls
   ▼
normalizer.normalize()    → CallLog        (construction fails unless CLEAN)
   ▼
writer.write()            → data/interim/source=aixblock/date=…/part-000.jsonl.gz
   │                          quarantined ids + reasons → part-000.quarantine.jsonl
   ▼
mining.embedder           → dense vectors (bge-large-en-v1.5, cached)
mining.cluster            → HDBSCAN labels; −1 is noise
mining.labeler            → LLM names centroids into L1/L2/L3   [structured output]
mining.slot_miner         → SlotSpec induction, pii_entity tagging
mining.feasibility        → volume × complexity → Quadrant
   ▼
IntentTaxonomy → domains/retail/taxonomy.json
   ▼
data/raw/bitext           → every gold intent must resolve to a leaf   ← external validity
```

The corpus separation is the load-bearing part: mining on labelled data would
rediscover its own label set and score near-perfectly against itself. (ADR-0004)

### 4.2 Online — one voice turn

```
caller audio
  ├─ Silero VAD             end-of-speech                    100 ms
  ├─ Deepgram Nova-3        speech_final                     180 ms
  ├─ redaction.pipeline     RedactedText (REALTIME)          0.1 ms   ← Rule 2 on the hot path
  ├─ router                 IntentPrediction + confidence     90 ms
  │     confidence ≥ route        → dispatch to the L1 micro-agent
  │     clarify_floor ≤ c < route → clarify (bounded)
  │     c < clarify_floor         → escalate
  │     risk_tier REGULATED       → escalate without a tool call
  ├─ slot_fill              elicit + validate pack-declared slots
  ├─ tool_exec              authorized_by(spec) → dispatch → redact output   150 ms
  ├─ respond                grounded token stream                           180 ms TTFT
  └─ Cartesia Sonic         first audio chunk                               120 ms TTFB
                                                              ──────
                                                               823 ms declared
                                                               800 ms ceiling (p95)
```

Every stage writes into `SessionState.latency`, so `breached` is a fact about the session
rather than a metric to reconstruct later.

**Barge-in** runs concurrently: on detected speech during playback, send `clear` to TTS,
cancel the LLM generation, reset the turn, and increment `barge_in_count`. The
interruption path leaves state consistent — that is its test.

**DTMF** is an alternate capture channel for any slot with `dtmf_capturable`, for noisy
lines and accented speech. Digit runs ≥4 are masked before storage.

### 4.3 Escalation

```
policy fires (confidence · sentiment · risk · tool_failure · max_turns · caller request)
   ▼
EscalationDecision recorded on SessionState
   ▼
summarizer            → RedactedText summary grounded in the transcript
disposition           → code derived from taxonomy + outcome
   ▼
HandoffContext        → refuses construction if ANY field is unredacted
   ▼
copilot/crm adapter   → CTI attached data + WS push to the agent desktop
```

The desktop receives verified identity (hashed), the full intent path, collected slots,
a step-by-step summary, sentiment traversal, the redacted transcript, the tool trace and
next-best actions. Audio is a URI, never inline.

---

## 5. Subsystem integration

| Subsystem | Depends on | Integration point |
|---|---|---|
| M1 Ingestion | M2, `schemas`, `datasets` | Calls the redactor inline; emits `CallLog` |
| M2 Redaction | `schemas` only | A library, not a stage — used offline by M1 and online by M5 |
| M3 Mining | M1 output, `llm` | Reads parquet; writes `taxonomy.json` back into the pack |
| M4 Orchestration | `config`, `llm`, `schemas` | Builds the graph from pack + taxonomy at startup |
| M5 Voice | M2, M4, `config.budget` | Owns the turn loop; drives M4 per turn; writes the ledger |
| M6a Copilot | M4 output | Consumes `HandoffContext`; pushes over WS |
| M6b Evals | everything | Offline replay + 5% async production sampling |
| Observability | `schemas` | `TraceContext` on every payload; JSON lines + hash-chained audit |

### Cross-cutting seams

- **`schemas/` is the bottom.** No intra-project imports; the module gate asserts it.
- **`llm/` is the only place a provider SDK appears.** `capabilities.py` gates per-model
  parameters because the Messages API rejects unsupported ones rather than ignoring them.
- **`config/` is the only place environment is read.** One class to audit a deployment.
- **The pack is the only place a vertical appears.**

---

## 6. Deployment shape

```
┌───────────────┐   WebRTC    ┌────────────────────┐
│ browser / SIP │────────────►│ livekit-server     │
└───────────────┘             └─────────┬──────────┘
                                        │ room
                              ┌─────────▼──────────┐      ┌──────────────┐
                              │ ccas voice worker  │─────►│ Deepgram     │
                              │  (M5 + M2′ + M4)   │◄─────│ Cartesia     │
                              └──┬──────────┬──────┘      └──────────────┘
                                 │          │
                   ┌─────────────▼──┐   ┌───▼───────────────┐
                   │ redis          │   │ LLM provider      │
                   │ semantic cache │   │ anthropic ∥ vLLM  │
                   └────────────────┘   └───────────────────┘
                                 │
                   ┌─────────────▼──┐   ┌───────────────────┐
                   │ qdrant (RAG)   │   │ otel-collector    │
                   └────────────────┘   └───────────────────┘

  ccas api (FastAPI)  ──►  agent desktop (WS copilot, REST handoff)
  batch jobs          ──►  ingest · mine_taxonomy · run_evals
```

The voice worker is the latency-critical process and scales horizontally per concurrent
call. Batch jobs are independent. The API serves the copilot and handoff surface.

---

## 7. Migration strategy

Strangler fig, not a cutover. The taxonomy's feasibility matrix sequences it:

| Phase | Traffic | Intents | Legacy role |
|---|---|---|---|
| 1 | ~20% | `IMMEDIATE_MIGRATION` — high volume, low complexity | fallback for everything else |
| 2 | ~80% | `PHASED_AGENTIC` added | escalation only |
| 3 | ~95% | desktop replaced by the copilot | media servers only |
| 4 | 100% | trunks ported to the cloud gateway | decommissioned |

Canary sequence within each phase: internal testing → 1% off-peak → 10% → 50% → full.
Rollback is a SIP routing change, not a deploy.

---

## 8. Known limits of the current design

Honest statement of what this design does not yet answer:

- **Carrier reality is unmeasured.** WebRTC over a good link is the optimistic case. Real
  jitter and packet loss land with SIP; Phase 5 numbers are a floor, not a forecast.
- **The 823 ms declared total exceeds the 800 ms ceiling by design.** Either a stage
  comes in under budget in practice, or a budget needs an ADR. Negative slack makes that
  a conversation rather than a silent drift.
- **Presidio latency is now measured, and the estimate was wrong.** Regex alone is
  111 µs p99; adding a spaCy NER pass makes it 7.0 ms — 2.3× over the 3 ms slice, with the
  *small* model. Resolved by splitting the pipeline into `REALTIME` (regex) and `BATCH`
  (regex + NER) modes (ADR-0007). The residual risk is explicit: on the live path, a name
  spoken outside a self-identification cue reaches the provider. Nothing reaches storage,
  a taxonomy, a CRM or a human without a `BATCH` pass.
- **Graph sprawl is a real risk** as verticals accumulate. `agents/factory.py` generating
  one subgraph per L1 from the pack is the mitigation; whether it holds is a Phase 6
  question.
- **Rule 6 is enforced structurally but not yet behaviourally.**
  `tests/evals/test_provider_parity.py` now exists and executes. Its structural half —
  every node declares two variants naming distinct models — is green. Its live half
  reported `8/8 cases rate limited` against the OpenRouter free tier's 50-request daily
  cap, so it skips rather than passes. The gate is honest about being unmeasured
  (ADR-0014), which is the most a test can do about a quota; it is not a substitute for
  running it green.
- **The corpus is not what its filenames say.** Both AIxBlock files are insurance sales
  calls, and 99.3% of turns carry one speaker label (ADR-0013). This changed two things
  in the design rather than in the data: a third domain pack (`insurance`) mined from
  what the corpus actually contains, and the agent-speech filter demoted from a
  deterministic speaker-label gate to a rule in the labelling prompt. The demotion is a
  real weakening — a model judging "is this the agent talking" is softer than a field
  comparison — and it applies only to corpora whose diarization is unusable.
- **Voice biometrics, hierarchical RAG over policy documents, WFM recalibration and
  dual-region failover are out of scope.** See `docs/future-scoped-work.md`.
