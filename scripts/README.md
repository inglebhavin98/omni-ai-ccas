# `scripts/` — operational entry points

Batch jobs that run outside the call path. They print to stdout by design (the `T20`
lint rule is waived here) and log JSON to `logs/execution.log` like everything else.

| Script | Does |
|---|---|
| `fetch_datasets.py` | Downloads and hydrates the three corpora into `data/raw/` |
| `aixblock_transcripts.py` | AIxBlock archive → adapter-ready records. Pure, no network |
| `ingest.py` | Corpus → redactor → `CallLog` partitions in `data/interim/` |
| `mine_taxonomy.py` | Ingested logs → clusters → LLM labels → `domains/<pack>/taxonomy.json` |
| `run_evals.py` | Offline eval + provider parity |
| `bench_latency.py` | Per-stage latency against the budget |

## Typical order

```bash
uv run python scripts/fetch_datasets.py --all

# AIxBlock ships one JSONL per output name, so --path is required or both are pooled
uv run python scripts/ingest.py --source aixblock --domain insurance \
  --path data/raw/aixblock/healthcare.jsonl --out data/interim-healthcare

uv run python scripts/mine_taxonomy.py --domain insurance --source aixblock \
  --min-chars 60 --dry-run          # cluster and report, no LLM spend
uv run python scripts/mine_taxonomy.py --domain insurance --source aixblock --min-chars 60
```

Start with `--dry-run`. One LLM call per cluster is the entire cost of mining, so it is
worth getting the cluster count right before paying for labels.

## Gates these scripts enforce

- **Dataset roles.** `ingest.py` refuses a corpus not admitted for transcript ingestion;
  `mine_taxonomy.py` refuses one not admitted for mining. Both raise at the point of the
  mistake.
- **Redaction availability.** `ingest.py` will not start if the batch engines are
  unavailable. It fails closed rather than silently degrading to regex-only.
- **Encoder headroom.** The embedder refuses to load a model that will not fit, rather
  than stalling in uninterruptible sleep.

## Known environment trap

`uv run` re-resolves the pinned `en-core-web-sm` wheel from GitHub on every invocation.
When that host is slow the command hangs with no child process and 0% CPU, then fails
with `Request failed after 3 retries` — and `make check` dies at the lint step before
running anything. Use the venv directly (`.venv/bin/python -m pytest`) until it is fixed.
See `docs/future-scoped-work.md` 9.14.
