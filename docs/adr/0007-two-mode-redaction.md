# ADR 0007 — Two redaction modes: regex on the call path, NER in batch

Status: accepted · 2026-09-12
Supersedes the single-pipeline assumption in ADR-0005 (which remains correct about the
type gate; only the engine composition changes).

## Context

ADR-0005 established that redaction is a type gate and that realtime redaction sits on
the call path inside a 3 ms slice of the 800 ms budget. The 3 ms figure was an estimate,
made before anything was built.

It was wrong. Measured on the reference pipeline (8 representative ASR utterances,
p50/p99 over 1,600 runs):

| Composition | p50 | p99 | vs 3 ms slice |
|---|---|---|---|
| regex only | 86 µs | **111 µs** | 27× headroom |
| regex + spaCy NER (`en_core_web_sm`) | 5.7 ms | **7.0 ms** | **2.3× over** |

That is with the *small* model. `en_core_web_lg`, which production wants for detection
quality, is slower. No amount of tuning closes a 2.3× gap on a per-utterance model pass.

Three options were live: raise the budget, drop NER entirely, or split the pipeline.

## Decision

`RedactionPipeline` takes a `RedactionMode`.

| Mode | Engines | Where it runs |
|---|---|---|
| `REALTIME` | regex only | the live call path, per utterance, before any provider call |
| `BATCH` | regex + Presidio NER | ingestion, mining, stored transcripts, the handoff summary — everything with an audience or a retention period |

`REALTIME` does not merely skip the model — it does not consider Presidio *available*,
so a missing model cannot fail a live turn. `BATCH` keeps the fail-closed behaviour
unchanged: no model, no egress.

To narrow the gap this opens, the regex set gained a context-anchored
`person_name_self_id` rule that catches self-identification ("my name is …", "you're
speaking with …") at regex speed, using `capture_group` so the cue survives and only the
name is replaced. Self-identification is where a caller's name actually first appears.

## Consequences

- The 3 ms slice in `configs/latency_budget.yaml` is now a real, measured commitment with
  27× headroom, gated by `tests/latency/test_budget_stages.py` — including a test that
  fails if headroom drops below 5×.
- **Residual risk, stated plainly: on the realtime path, a caller's name spoken outside a
  self-identification cue reaches the LLM provider.** Every structured identifier — card,
  national id, IBAN, bank account, email, phone, credentials, postal code — is removed
  completely, in both modes. Names are the gap, and only in `REALTIME`.
- That risk is bounded by what `REALTIME` output is used for: it feeds an in-flight model
  call under contract. Nothing reaches storage, a taxonomy, a CRM, or a human agent
  without a `BATCH` pass, where full NER runs and fails closed.
- The mitigation path is a faster name detector, not a faster spaCy: a quantised ONNX NER
  or a gazetteer pass. Tracked as future-scoped-work 5.1.
- The spaCy model is now named explicitly in policy (`spacy_model`) rather than inherited
  from Presidio's default. Detection quality differs materially between `sm` and `lg`, and
  a silent fallback would make two deployments disagree about what counts as a name. A
  missing model is reported, never downgraded.
- Presidio attempts a network download and then `sys.exit(1)` on an unknown model name.
  The engine now checks `importlib.util.find_spec` first and also catches `SystemExit`,
  because a missing model must make an engine unavailable, not terminate the process.
- Cost: two modes to reason about, and a standing obligation that every storage and
  handoff path uses `BATCH`. `HandoffContext` and `CallLog` validators enforce that the
  text is clean; they cannot enforce *which mode* cleaned it. That is a review concern,
  and the reason the mode is on the pipeline rather than a per-call flag.
