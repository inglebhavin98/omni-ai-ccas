# `ingestion/` — M1: corpus → redactor → `CallLog`

Reads a published corpus, hands every utterance to M2, and emits contract-valid
`CallLog`s. An adapter translates a file format and makes no quality judgements;
everything else happens once, in the normalizer.

## Files

| File | Holds |
|---|---|
| `base.py` | `SourceAdapter`, `RawRecord`, `RawTurn`, `RawDtmf` |
| `adapters/` | One per corpus: `aixblock`, `bitext`, `natcs`, `generic_csv`, `synthetic` |
| `normalizer.py` | `Normalizer` — redact, validate, emit or quarantine |
| `datasets.py` | `DATASET_ROLES`, `require_role()`, `DatasetRoleError` |
| `registry.py` | `adapter_for(source)` |
| `writer.py` | Partitioned gzipped-JSONL sink + quarantine file |

## `RawRecord` is the one place unredacted text legitimately exists

It is in-process, short-lived, and consumed by the normalizer. Its `__repr__` is
deliberately blind — `RawTurn(speaker=caller, chars=214)` — so it cannot leak through a
traceback or a debugger watch. It has no path to a log, a prompt or a file.

Records whose redaction does not come back `CLEAN` are **quarantined**: counted, their id
and reason recorded, and dropped. Never emitted in a weaker form.

## Dataset roles are enforced, not documented

`require_role(source, role)` raises at the point of the mistake. Mining from a labelled
corpus rediscovers its own labels and proves nothing, so Bitext is refused for
`INTENT_MINING` ([ADR-0004](../../../docs/adr/0004-dataset-role-separation.md), amended
by [ADR-0015](../../../docs/adr/0015-natcs-is-supervision-not-a-trajectory.md)).

`DIALOGUE_CONTEXT_BENCHMARK` and `STATE_GRAPH_BENCHMARK` are declared and **unsourced** —
no corpus on disk supports them. Do not grant one to unblock a benchmark.

**Verify a corpus before trusting its description.** Both role assignments that have been
wrong were wrong because the dataset card said one thing and the bytes said another.

```bash
uv run python scripts/fetch_datasets.py --all
uv run python scripts/ingest.py --source aixblock --domain insurance \
  --path data/raw/aixblock/healthcare.jsonl --out data/interim-x
uv run pytest tests/test_module_1_ingestion.py -q
```
