# omni-ai-ccas — Technical Specification

Data models, API contracts, latency budgets and enforcement points.

- Version: 0.1.0 · schema version `1.0`
- Status: Phase 1 (foundation) shipped; Phases 2–6 specified below and not yet built
- Authority: `CLAUDE.md` for rules, this document for shapes, `src/ccas/schemas/` for truth

---

## 1. Contract layer

`src/ccas/schemas/` has no intra-project dependencies. Every model derives from `Frozen`
(immutable, `extra="forbid"`, `validate_assignment`) except `SessionState` and
`LatencyLedger`, which are mutable by necessity.

### 1.1 Primitives — `common.py`

| Type | Shape | Notes |
|---|---|---|
| `Slug` | `^[a-z0-9]+(?:[._-][a-z0-9]+)*$`, ≤128 | Intent ids, slot/tool/queue/domain names. **Never an enum** — Rule 1 |
| `SchemaVersion` | `Literal["1.0"]` | Bumped with a migration note |
| `TraceContext` | `trace_id` hex32 · `span_id` hex16 · `correlation_id` · `tenant_id` | W3C-shaped; rides on every boundary payload |
| `Channel` | voice · chat · sms · email · web | |
| `Speaker` | caller · bot · human_agent · system · ivr | |
| `RiskTier` | low · medium · high · regulated | Membership is pack-defined |
| `VerificationLevel` | none < soft < strong < step_up | Ordered; `.satisfies(required)` |
| `Urgency` | low · normal · high · critical | |

`DropsComputedFields` strips a model's own `@computed_field` names on input, so
`extra="forbid"` and computed fields coexist and every model round-trips through JSON.

### 1.2 Redaction — `pii.py`

```
PiiEntityType   person · org · location · address · phone · email · url · ip
                date_of_birth · age · national_id · payment_card · bank_account
                iban · account_ref · credential · health_info · biometric
                vehicle_id · custom(+custom_label)

RedactionStatus CLEAN      leak check passed  → egress permitted
                DIRTY      substitution failed → blocked
                UNVERIFIED engine unavailable  → blocked (fail closed)
                BYPASSED   test-only override  → blocked everywhere
```

| Model | Fields | Invariants |
|---|---|---|
| `RedactionSpan` | start · end · entity_type · custom_label · score · engine · replacement | `end > start`; `custom_label` iff `CUSTOM` |
| `RedactionReport` | status · spans · engines_run · policy_version · elapsed_us · leak_check_passed · residual_patterns | `CLEAN` requires a passed leak check, ≥1 engine, no residuals. Non-`CLEAN` cannot claim a passed check. **Carries no matched text** |
| `RedactedText` | text · report · source_sha256 | `require_egress()` raises `PermissionError` unless `CLEAN` |

`RedactedText` is the **only** string type permitted across an external boundary.

### 1.3 `CallLog` — Module 1 output

```
CallLog
  schema_version    "1.0"
  call_id           sha256(source|source_record_id)[:32]   deterministic → idempotent re-ingest
  source            aixblock | natcs | bitext | synthetic | generic_csv | live_capture
  source_record_id  str
  channel           Channel
  domain_hint       Slug | None          free-form; a new vertical needs no code change
  locale            str = "en-US"
  utterances        tuple[Utterance, ...]   ≥1, indices contiguous from 0
  dtmf_events       tuple[DtmfEvent, ...]
  duration_ms       int | None
  labels            GoldLabels | None       present only where the corpus ships them
  metadata          dict[str, JsonValue]    adapter passthrough, redacted
  redaction         RedactionReport
  ingested_at       datetime
```

| Sub-model | Fields |
|---|---|
| `Utterance` | index · speaker · **content: RedactedText** · start_ms · end_ms · asr_confidence · language |
| `DtmfEvent` | digits `^[0-9*#A-D]+$` · at_ms · redacted — runs ≥4 digits **must** be redacted |
| `GoldLabels` | intent · category · slots · resolution |

**Construction invariant:** refuses unless `redaction.status is CLEAN` *and* every
utterance's content is egress-permitted.

### 1.4 `IntentTaxonomy` — Module 3 output

```
IntentTaxonomy
  schema_version    "1.1"
  taxonomy_id · domain · version(semver)
  nodes             tuple[IntentNode, ...]    ≥1
  embedding_model   str                        e.g. bge-large-en-v1.5
  embedding_mean    tuple[float, ...]          the space every centroid lives in; () if raw
  clusterer         "hdbscan" | "bertopic"
  clusterer_params  dict                       includes centre, min_samples
  labeler_model     str
  coverage          float                      coverage + noise_ratio == 1.0
  noise_ratio       float
  source_call_ids   tuple[str, ...]
```

**Centroids live in centred space** (ADR-0017). Clustering subtracts the corpus mean, so
`IntentNode.centroid` is computed there too and `embedding_mean` is what maps a vector in.
A live utterance **cannot** centre itself — the mean of one vector is itself, so centring
it gives zero — which is why the mean ships with the taxonomy rather than being recomputed.
A validator rejects a taxonomy whose centroid width disagrees with its mean.

#### Migration — `IntentTaxonomy` 1.0 → 1.1

Additive; no stored taxonomies exist, so no data migration is required.

| 1.0 | 1.1 |
|---|---|
| — | `embedding_mean: tuple[float, ...] = ()` |
| `centroid` computed from raw vectors | computed in cluster space |
| `clusterer_params` without `centre` | `centre: bool`, `min_samples: int` recorded |

A 1.0 document still validates, because the field is defaulted. It is nonetheless unsafe
to route against: centroids with no mean cannot be compared to a live embedding. Treat
`embedding_mean == ()` together with `clusterer_params["centre"] is True` as a taxonomy
that must be re-mined.

```
IntentNode
  intent_id      Slug, dotted; depth == level     "billing.dispute.status"
  level          1 | 2 | 3
  parent_id      Slug | None                      L1 ⇔ None; child must nest under parent
  label · description · exemplars(≤20, redacted)
  cluster_id · centroid
  volume         VolumeStats
  automation     AutomationScore → computed Quadrant
  slots          tuple[SlotSpec, ...]             unique names
  required_tools tuple[Slug, ...]                 must resolve in the pack (checked at load)
  risk_tier      RiskTier
  escalation     EscalationPolicy
```

| Model | Fields |
|---|---|
| `SlotSpec` | name · slot_type · required · elicitation_prompt · reprompt · validation_regex · enum_values · **pii_entity** · dtmf_capturable · max_attempts |
| `VolumeStats` | utterance_count · call_count · share_of_total · avg_turns_to_resolve · historical_escalation_rate |
| `AutomationScore` | feasibility · complexity · confidence · rationale · volume_share · thresholds → `quadrant` |
| `EscalationPolicy` | min_intent_confidence(.82) · max_clarifications(2) · max_tool_failures(2) · frustration_threshold(.7) · auto_escalate · target_queue · required_skills |

**Quadrant matrix** (thresholds pack-configurable, default volume .05 / complexity .5):

| | low complexity | high complexity |
|---|---|---|
| **high volume** | `IMMEDIATE_MIGRATION` | `PHASED_AGENTIC` |
| **low volume** | `SELF_SERVICE_SCRIPTED` | `DIRECT_AGENT_ROUTE` |

Graph integrity: unique ids, every `parent_id` resolves, no cycles, `coverage + noise == 1`.
Helpers: `resolve()`, `path()` (L1→L2→L3), `children()`, `leaves()`.

### 1.5 `SessionState` — Modules 4/5 runtime

The LangGraph state. Append-only fields carry `operator.add` reducers so parallel nodes
merge rather than clobber.

```
SessionState
  session_id · trace · domain · channel · caller
  turns               Annotated[list[Turn], add]
  intent_history      Annotated[list[IntentPrediction], add]
  sentiment_trail     Annotated[list[SentimentSnapshot], add]
  tool_records        Annotated[list[ToolRecord], add]
  current_intent · active_agent · slots · pending_slot
  turn_index · clarification_count · no_input_count · no_match_count
  barge_in_count · tool_failure_count
  escalation · outcome · terminal
  latency             LatencyLedger
  started_at
```

| Sub-model | Fields |
|---|---|
| `CallerContext` | **caller_ref (hashed)** · verification · verified_attributes · locale · prior_interaction_count · omnichannel_context |
| `Turn` | index · speaker · content(RedactedText) · intent · tool_call_ids · barged_in · rtt_ms · at |
| `IntentPrediction` | intent_id · confidence · source(llm_router\|classifier\|dtmf\|gold\|fallback) · alternatives · latency_ms |
| `SlotValue` | name · raw(RedactedText) · parsed · valid · attempts · captured_via |
| `SentimentSnapshot` | valence(−1..1) · arousal(0..1) · frustration_index(0..1) · turn_index |
| `EscalationDecision` | reason(HandoffReason) · triggered_by · at_turn · urgency · detail |
| `SessionOutcome` | status(contained\|escalated\|abandoned\|error) · resolved_intent · turn_count · duration_ms · disposition_code |

Helpers: `sentiment` (latest), `escalated`, `missing_required_slots(required)` — treats
invalid as missing — and `failed_tool_calls()`.

### 1.6 Tools — `tools.py`

```
ToolSpec        name · description · input_schema · output_schema · domain
                idempotent · side_effecting · timeout_ms(50..10000)
                requires_verification · risk_tier · pii_output_fields · retry_policy

ToolPayload     tool_call_id · tool_name · session_id · intent_id · arguments
                idempotency_key · timeout_ms · verification_level · trace · issued_at

ToolResult      tool_call_id · status · data · error_code · error_message
                elapsed_ms · retryable · redaction

ToolRecord      payload · result · attempt          (ids must match)
```

`ToolStatus`: `ok · error · timeout · denied · not_found · invalid_args`.

**Invariants**

- `input_schema` must be `type: object` with `additionalProperties: false` — an open
  schema lets a model smuggle unvalidated arguments into a backend.
- A non-idempotent side-effecting tool must not auto-retry.
- `side_effecting ⇒ requires_idempotency_key`.
- `ToolPayload.authorized_by(spec)` is the dispatch gate: verification level satisfied
  **and** idempotency key present where required.
- `ToolResult.safe_for_model` — backend output is untrusted and must be redacted before
  a model reads it.
- An `OK` result carries no error; a failed result requires an `error_code`.

### 1.7 `HandoffContext` — CTI payload

The highest-risk egress surface: it leaves the process, lands in a third-party desktop,
and is retained.

```
HandoffContext
  schema_version    "1.1"
  handoff_id · session_id · trace · domain
  reason            HandoffReason
  urgency · target_queue · required_skills
  identity          VerifiedIdentity | None      (level must exceed NONE)
  intent_path       tuple[Slug, ...]             must descend L1→L2→L3
  intent_confidence float
  collected_slots   dict[str, SlotValue]
  summary           RedactedText                 LLM-generated
  next_best_actions tuple[NextBestAction, ...]    action/rationale are RedactedText
  sentiment_trail · transcript · tool_trace
  cti_attributes    dict[str, RedactedText]      UUI / attached data for Genesys, Cisco
  audio_recording_ref  str | None                object-store URI; audio never inline
  created_at
```

`HandoffReason`: `low_confidence · caller_request · negative_sentiment · risk_tier ·
tool_failure · max_turns · verification_failed · policy · unsupported_intent ·
system_error`.

**Construction invariant:** refuses if the summary, any transcript turn, any collected
slot, any tool result, any CTI attribute, any next-best action, or any verified-identity
attribute is not egress-permitted.

**Migration 1.0 → 1.1.** Three text-bearing fields were plain `str` and therefore outside
the invariant the model exists to hold: `cti_attributes` values, `NextBestAction.action`
and `.rationale`, and `VerifiedIdentity.attributes` values. All three are now
`RedactedText` and are checked on construction.

The gap was not academic. `cti_attributes` is what reaches Genesys or Cisco as attached
data and is retained there, and `VerifiedIdentity.attributes` is caller-derived by
definition — it is whatever the pack asked the caller to verify. The escalate node was
copying those straight out of `CallerContext.verified_attributes` unredacted; it now
passes them through the redactor. `cti_attributes` values written by the node are
repo-authored (a policy name, an enum member, a turn count) and use `authored()`.

*To migrate:* wrap each value. Caller-derived text goes through the redaction pipeline;
text written in this repository goes through `ccas.llm.prompt.authored`. There is no
in-place upgrade for a persisted 1.0 payload, because the original strings carry no report
and one cannot be invented after the fact — re-derive from the session or discard.

### 1.8 `DomainPack` — the vertical surface

```
DomainPack
  domain · version(semver) · display_name · locales
  taxonomy_ref          path relative to the pack directory
  tools                 tuple[ToolSpec, ...]     domain must match the pack
  queues                tuple[QueueSpec, ...]    ≥1; queues[0] is the default
  redaction_patterns    tuple[PatternSpec, ...]  layered over the global policy
  prompts               dict[str, str]
  greeting              str
  confidence            ConfidenceThresholds     clarify_floor ≤ route
  quadrant_thresholds   QuadrantThresholds
  max_turns             int
  compliance            ComplianceProfile
```

| Model | Fields |
|---|---|
| `QueueSpec` | name · display_name · skills · min_verification · max_risk_tier · cti_queue_id |
| `PatternSpec` | name · pattern · entity_type · custom_label · score · validator_name |
| `ConfidenceThresholds` | route(.82) · clarify_floor(.45) · slot_accept(.70) |
| `ComplianceProfile` | regimes(none\|pci_dss\|hipaa\|gdpr\|ccpa\|sox\|glba) · transcript_retention_days · audio_retention_days · recording_consent_required · consent_prompt |

Names must be unique across tools, queues and patterns. Consent required ⇒ a prompt.

### 1.9 LLM contracts — `llm.py`

```
ModelBinding   node · provider(anthropic|vllm) · model · effort · max_tokens
               temperature · stream · cache_prefix · timeout_ms · latency_budget_ms

Message        role(system|user|assistant) · content: RedactedText
LLMRequest     binding · system · messages(≥1) · tools · response_schema · stop_sequences
LLMChunk       index · text · tool_call_delta · is_final
LLMResponse    binding · text · tool_calls · parsed · stop_reason · usage · ttft_ms
               total_ms · refused → within_budget
LLMUsage       input_tokens · output_tokens · cache_read_tokens · cache_write_tokens
```

`Effort`: `low · medium · high · xhigh · max`.

Invariants: an Anthropic binding **rejects** `temperature` (the current generation does
not accept sampling params alongside thinking — use `effort`); an `LLMRequest` refuses
any message or system prompt that is not egress-permitted.

### 1.9b Ingestion contracts — `ingestion/base.py`

```
RawTurn     speaker · text (UNREDACTED) · start_ms · end_ms · confidence · language
RawDtmf     digits · at_ms
RawRecord   source · record_id · channel · turns(≥1) · dtmf · locale · domain_hint
            duration_ms · labels · metadata
```

`RawRecord` is the only unredacted structure in the codebase. `__repr__`/`__str__` are
overridden to render counts, never content, so a traceback cannot leak it. `metadata`
rejects strings over 128 chars — free text must live in a turn the redactor can see.

```
NormalizeOutcome   record_id · call_log | None · reason        (reason: names, never text)
NormalizeStats     seen · emitted · quarantined · reasons · entity_counts
WriteResult        path · quarantine_path · written · quarantined
```

### 1.10 Eval contracts — `eval.py`

```
JudgeDimension  faithfulness · pii_leakage · task_success · policy_adherence
JudgeScore      dimension · score(0..1) · rationale · passed
JudgeVerdict    session_id · judge_model · judge_provider · scores · sampled
                → by_dimension · passed · leaked_pii
EvalCase        case_id · domain · description · turns · expected_intent
                expected_tools · expected_outcome · must_escalate · tags
MetricValue     name · value · threshold · higher_is_better → passed
ProviderParityResult
                node · baseline · candidate           (variant names, not vendors)
                baseline_model · candidate_model      (must differ)
                agreement · agreement_threshold(.85)
                baseline_p95_ttft_ms · candidate_p95_ttft_ms
                sample_size(measured) · unmeasured · min_sample_size(1)
                → measured · inconclusive · passed
EvalReport      report_id · domain · suite · metrics · verdicts · parity · context
                → passed · pii_leaks
```

**Parity verdicts have three states, not two** (ADR-0014). `sample_size` counts cases a
model actually answered. A case nobody served — a 429, a quota, a cold model — lands in
`unmeasured` and leaves the denominator, because scoring it as disagreement invents a
defect that does not exist. `passed` additionally requires `min_sample_size`
measurements, so a run that measured nothing cannot read as 100% agreement.
`inconclusive` names the middle state: too few measurements, and throttling is why.

#### Migration — `EvalReport` 1.0 → 1.1

Breaking for `ProviderParityResult` only; no stored reports exist yet, so no data
migration is required.

| 1.0 | 1.1 |
|---|---|
| `baseline: ProviderName` | `baseline: str` — a *variant* name (`openrouter_alt`) |
| `candidate: ProviderName` | `candidate: str` |
| — | `baseline_model`, `candidate_model` — required, and validated distinct |
| `sample_size: int ≥ 1` | `sample_size: int ≥ 0` — measured cases only |
| — | `unmeasured: int = 0`, `min_sample_size: int = 1` |
| `passed = agreement ≥ threshold` | `passed = measured and agreement ≥ threshold` |

The type change is the point: with every model behind one gateway, two variants share a
provider and differ only in model (ADR-0012), which the 1.0 shape could not express —
it would have recorded `openrouter vs openrouter` and validated fine.

---

## 2. Latency budget

`configs/latency_budget.yaml` is the contract. `tests/latency/` asserts against the same
file an operator tunes, so the two cannot drift. Raising a number requires an ADR.

| Stage | Budget | Owner |
|---|---|---|
| VAD end-of-speech | 100 ms | `voice/vad.py` — Silero, tunable `min_silence_ms` |
| STT final | 180 ms | `voice/stt/deepgram.py` — Nova-3 `speech_final` |
| Realtime redaction | 3 ms | `redaction/pipeline.py` — regex only; **measured 111 µs p99** |
| Router + policy | 90 ms | `orchestration/router.py` — Haiku / small local, cached |
| Tool execution | 150 ms | `orchestration/tools/executor.py` — parallel where independent |
| LLM TTFT | 180 ms | `llm/` — streaming, prompt-cached prefix |
| TTS TTFB | 120 ms | `voice/tts/cartesia.py` — Sonic chunked |
| **Declared total** | **823 ms** | against an **800 ms** p95 ceiling |
| **Slack** | **−23 ms** | deliberately negative — regressions cannot hide in slack |

`LatencyLedger` carries the ceiling as data: `total_rtt_ms`, `breached` and `headroom_ms`
are answerable from a session with no I/O.

### Redaction modes (ADR-0007)

| Mode | Engines | p50 | p99 | Used by |
|---|---|---|---|---|
| `REALTIME` | regex | 86 µs | 111 µs | M5 voice turn loop |
| `BATCH` | regex + Presidio NER | 5.7 ms | 7.0 ms | M1 ingestion, M3 mining, M6 handoff |

Full NER is 2.3× over the 3 ms slice with the *small* spaCy model, so it does not run on
the call path. `Normalizer` rejects a `REALTIME` pipeline: stored records are exactly
what must not contain a caller's name.

### Per-node LLM budgets

| Node | Anthropic | vLLM | TTFT budget |
|---|---|---|---|
| `router` | `claude-haiku-4-5` (effort low) | `llama-3.3-70b-instruct` (temp 0.0) | 90 ms |
| `task_agent` | `claude-opus-5` (effort high) | `llama-3.3-70b-instruct` (temp 0.2) | 180 ms |
| `taxonomy_labeler` | `claude-opus-5` | `llama-3.3-70b-instruct` | offline |
| `judge` | `claude-haiku-4-5` | `llama-3.1-8b-instruct` | offline, 5% sample |
| `summarizer` | `claude-opus-5` | `llama-3.3-70b-instruct` | offline |

---

## 3. API contracts

### 3.1 `LLMProvider` (internal)

```python
class LLMProvider(ABC):
    name: ProviderName
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
    def      stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]: ...
    async def structured(self, request: LLMRequest) -> LLMResponse: ...   # parsed is schema-valid
    async def healthy(self) -> bool: ...
    async def aclose(self) -> None: ...
```

`ProviderUnavailableError` is raised rather than auto-failing-over — a silent failover
would make parity numbers meaningless.

**Anthropic shaping** (`build_kwargs`, pure): `thinking={"type":"adaptive"}` and
`output_config.effort` only for models that accept them (`llm/capabilities.py`);
`output_config.format` for structured output; top-level `cache_control` for the stable
prefix; `betas=["server-side-fallback-2026-07-01"]` + `fallbacks="default"` on
Opus-5-class models. Unknown model ids get the conservative shape. No assistant prefill.

**vLLM shaping** (`build_payload`, pure): OpenAI `/v1/chat/completions`; system prompt as
`messages[0]`; `response_format.json_schema` for guided decoding; SSE streaming.

### 3.2 HTTP surface — workbench (Phase 4, shipped)

Development console. Binds to loopback; not a production endpoint.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness |
| `GET` | `/ready` | provider configured?, per-pack load status, taxonomy presence |
| `GET` | `/v1/domains` | pack shape: tools, queues, thresholds, compliance |
| `POST` | `/v1/sessions` | open a chat-channel session → snapshot |
| `POST` | `/v1/sessions/{id}/turns` | submit an utterance → snapshot |
| `GET` | `/v1/sessions/{id}` | current snapshot |
| `DELETE` | `/v1/sessions/{id}` | drop the session **and its vault** |
| `GET` | `/v1/redaction/preview` | what a model would see — **no provider needed** |
| `GET` | `/v1/tools` | catalogue with scope (shared/pack) and schemas |
| `POST` | `/v1/tools/{name}/invoke` | one tool through the full gate chain — **no provider needed** |
| `GET` | `/` | the console |

A snapshot carries the redacted transcript, the routing prediction, **all four policy
verdicts** (not only the decisive one), slots, counters, tool records and the latency
ledger. Everything is re-read from objects the graph produced, so nothing reaches this
layer that could not also reach a model.

With no provider configured the session still opens and redaction, tools and policies all
work; routing escalates honestly rather than pretending to understand.

### 3.2a Agent-desktop surface (Phase 6, planned)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/handoffs/{id}` | fetch a `HandoffContext` for an agent desktop |
| `WS` | `/v1/ws/copilot/{session_id}` | live redacted transcript + suggestions |

Every response body is a `schemas/` model. `correlation_id` is echoed on every response
and present on every log line.

### 3.2b Tool layer (Phase 4)

Registry layers `configs/tools.yaml` (`domain: core`) under each pack's own tools; a
pack may add but not shadow. `ToolExecutor` is the single gate chain:

```
resolve -> validate(jsonschema) -> authorise -> dispatch(timeout) -> retry -> redact -> record
```

| Step | Failure status | Reaches a backend? |
|---|---|---|
| unregistered tool | `NOT_FOUND` | no |
| schema violation | `INVALID_ARGS` | no |
| `authorized_by(spec)` false | `DENIED` | no |
| per-attempt timeout | `TIMEOUT` | yes |
| total budget spent | `TIMEOUT` / `budget_exhausted` | no further attempts |
| output not egress-clean | `ERROR` / `redaction_failed`, `data` withheld | yes |

`total_budget_ms` bounds **all** attempts, not each one: without it a two-attempt retry
on a 900 ms tool spends 1,924 ms against a 150 ms slice. Idempotency keys are
`sha256(session_id|tool|args)[:32]` — stable across a retry, different for a new request.

Output redaction runs twice over: `pii_output_fields` are replaced wholesale, every other
string passes the redaction pipeline (ADR-0009).

### 3.2c Policy engine (Phase 4)

Pure functions, `(SessionState, PolicyContext) -> PolicyDecision`. First decisive verdict
wins; its name becomes `EscalationDecision.triggered_by`.

| Order | Policy | Escalates on |
|---|---|---|
| 1 | `risk` | `RiskTier.REGULATED` or `auto_escalate` |
| 2 | `retry` | turn ceiling · tool failures · no-input · no-match |
| 3 | `sentiment` | frustration ≥ threshold, **or** three consecutive rises above half it |
| 4 | `confidence` | below `clarify_floor`, or clarifications exhausted |

`confidence` returns `CLARIFY` in the middle band. `sentiment` precedes `confidence`
deliberately: the reverse asks a furious caller to rephrase.

### 3.3 Tool backend adapter (Phase 4)

```python
class ToolBackend(Protocol):
    async def invoke(self, spec: ToolSpec, payload: ToolPayload) -> ToolResult: ...
```

The executor owns validation, timeout, retry, idempotency and output redaction. A
backend implements transport only. `mock_backend` serves deterministic fixtures so the
graph is testable with no enterprise system.

---

## 4. Storage layout

```
data/raw/{aixblock,bitext,natcs}/   corpora, role-restricted, unredacted by design
  aixblock/<name>.jsonl             turns from word timing; <name> is a path, not a
                                    domain -- both files are insurance (ADR-0013, ADR-0016)
  aixblock/.archives/               the published ZIPs, cached so a re-fetch is free
  bitext/customer_support.csv       as published; the adapter reads this
  natcs/dialogues.jsonl             4,205 labelled caller turns, 82 intents. Turns are
                                    sampled, not consecutive -- supervision, not a
                                    trajectory (ADR-0015)
data/interim/source=<s>/date=<d>/   part-000.jsonl.gz         CallLog, gzipped JSON lines
                                    part-000.quarantine.jsonl record ids + reasons only
data/processed/                     taxonomies, embeddings, eval fixtures
domains/<pack>/taxonomy.json        mined IntentTaxonomy, loaded at runtime
logs/execution.log                  JSON lines, correlation_id on every event
```

The quarantine file is the audit trail for what was dropped and why. It holds record ids
and reason *names* — the record that failed redaction is not written anywhere.

Only the `ccas` logger namespace reaches `logs/execution.log`; third-party libraries keep
their default stderr behaviour, so the file stays parseable as JSON lines.

A parquet sink is specified but not implemented — `create_writer(..., "parquet")` raises
with an actionable message. Tracked in `docs/future-scoped-work.md`.

Audio lives in encrypted object storage referenced by URI. It never enters a Pydantic
model, a log line, or a prompt.

---

## 5. Enforcement summary

| Invariant | Mechanism | Test |
|---|---|---|
| No unredacted egress | `RedactedText` + construction validators | `tests/security/test_egress_gate.py` |
| Fail closed on engine loss | `UNVERIFIED` blocks egress | same |
| No raw content in logs | `_forbid_raw_content` raises | `tests/unit/observability/test_logging.py` |
| No vertical vocabulary in core | whole-word scan of `src/ccas/` | `tests/security/test_no_domain_literals.py` |
| 800 ms ceiling | `configs/latency_budget.yaml` | `tests/latency/` |
| Every node has ≥2 variants | `load_bindings` refuses a single-variant node | `tests/unit/llm/test_bindings.py` |
| Variants name distinct models | structural half of the Rule 6 gate | `tests/evals/test_provider_parity.py` |
| Variants agree on decisions | live run, ≥75% over ≥5 measured cases | same (`integration`; skips when throttled) |
| A committed taxonomy is traceable | provenance must not name a stub or the hashing embedder | `tests/test_module_3_mining.py` |
| Corpora not mixed | `require_role()` raises | `tests/unit/ingestion/test_datasets.py` |
| Tools resolve | checked when a taxonomy loads | `tests/unit/config/test_domain_loader.py` |
| Contracts round-trip | JSON round-trip per model | `tests/unit/schemas/test_roundtrip.py` |
