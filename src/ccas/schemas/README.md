# `schemas/` — the frozen contract layer

Every shape the platform passes around. **This package imports nothing from the rest of
the project**, and everything else depends on it. That one-way rule is what stops the
contracts drifting toward whichever module was edited last.

All models inherit `Frozen` (Pydantic v2, `frozen=True`, `extra="forbid"`). The single
exception is `SessionState`, which is mutable because a live call mutates it.

## Files

| File | Holds |
|---|---|
| `common.py` | `Slug`, `Speaker`, `Channel`, `RiskTier`, `Frozen`, shared aliases |
| `pii.py` | `RedactedText`, `RedactionReport`, `RedactionStatus`, `sha256_hex` |
| `call_log.py` | `CallLog`, `Utterance`, `DtmfEvent`, `DatasetSource`, `GoldLabels` |
| `taxonomy.py` | `IntentTaxonomy`, `IntentNode`, `VolumeStats`, `AutomationScore` |
| `session.py` | `SessionState` — the one mutable model |
| `tools.py` | `ToolSpec`, `ToolResult`, argument schemas |
| `domain.py` | `DomainPack`, `SlotSpec`, queue and threshold declarations |
| `handoff.py` | `HandoffContext` — refuses construction unless redaction is `CLEAN` |
| `escalation.py` | `EscalationPolicy` |
| `llm.py` | `ProviderName`, binding and usage shapes |
| `eval.py` | `EvalReport`, `ProviderParityResult` |

## Why the validators matter

Two models — `CallLog` and `HandoffContext` — refuse to exist if redaction did not come
back `CLEAN`. This is Rule 2 implemented as a type gate rather than a checklist
([ADR-0005](../../../docs/adr/0005-redaction-as-type-gate.md)): you cannot write an
unredacted record to a sink, because you cannot construct one to write.

`RedactedText.egress_permitted` is the single predicate the rest of the platform asks
before anything leaves the process.

## Changing anything here

A public API change needs a `schema_version` bump **and** a migration note in
`docs/tech-spec.md` §1.4. `IntentTaxonomy` is at 1.1 — 1.0 added `embedding_mean`
([ADR-0017](../../../docs/adr/0017-mean-centre-before-clustering.md)). Additive, defaulted fields keep old documents valid;
anything else does not.

Coverage floor here is **100%**, not the 85% that applies elsewhere.

```bash
uv run pytest tests/unit/schemas -q
```
