# `redaction/` — M2: zero leakage

A library, not a pipeline stage. Anything that might cross a boundary goes through here
first, and the result is a `RedactedText` whose report says whether egress is permitted.

## Files

| File | Holds |
|---|---|
| `pipeline.py` | `build_pipeline()`, `RedactionMode`, engine orchestration and readiness |
| `regex_engine.py` | Pattern pass. Fast enough for the call path |
| `presidio_onnx.py` | Local Presidio + spaCy NER. Batch mode only |
| `leak_detector.py` | Re-scans output; any residual hit means `DIRTY` |
| `placeholder.py` | `[ENTITY_TYPE_N]` allocation, stable per session |
| `vault.py` | `PlaceholderVault` — session-scoped, never on a Pydantic model |
| `policy.py` | Policy loading from `configs/redaction_policy.yaml` + pack patterns |
| `validators.py` | Reusable contract-level checks |

## Two modes, for one reason

Local NER costs milliseconds the call path does not have, so
([ADR-0007](../../../docs/adr/0007-two-mode-redaction.md)):

- **`CALL_PATH`** — regex only, ~111 µs p99, inside the 3 ms slice
- **`BATCH`** — regex + Presidio + leak detector, ~7 ms, for anything stored, mined, or
  shown to a human

## Never fail open

If an engine is unavailable the status is `UNVERIFIED` and egress is **forbidden** — fail
the turn and escalate. `scripts/ingest.py` refuses to start rather than silently
degrading to regex-only, because a partially redacted record that looks normal is the
worst artefact this pipeline could produce.

## Redact by context, not only by shape

Six bare digits match no pattern. If the caller was asked for a reference and gave one,
the answer is replaced wholesale (`redact_value`). Shape-matching alone would miss it.

The vault is how a real backend still gets real values: `ToolExecutor` restores
placeholders immediately before dispatch and nowhere else
([ADR-0010](../../../docs/adr/0010-placeholder-vault.md)). A registered tool backend is
an internal system of record, not an external boundary.

Coverage floor here is **100%**. Hypothesis fuzzes generated identifiers.

```bash
make security
uv run python -m cli.demo gate      # watch the gate refuse
```
