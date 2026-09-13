# data/raw/bitext — the supervision corpus

**Bitext Customer Support Intent dataset.** Labelled intent + category pairs with slot
annotations. Small, clean, single-turn.

## Permitted roles

| Role | Why this corpus |
|---|---|
| `nlu_ground_truth` | Labelled intents grade the front-door router's accuracy. |
| `slot_extraction_ground_truth` | Slot annotations grade tool-argument extraction. |
| `judge_evaluation` | Known-correct answers calibrate the LLM-as-a-Judge. |

## Denied roles

`intent_mining` — **this is the important one.** Clustering a labelled set rediscovers
its own label taxonomy and proves nothing about the platform's ability to find structure
in real traffic. Mining runs on `aixblock`; Bitext then *grades* the result.

`transcript_ingestion`, `redaction_corpus` — single-turn and already sanitised, so it
exercises neither the multi-turn normaliser nor the redaction hot path.

## Where it is used

- **Module 3** (`tests/evals/test_taxonomy_quality.py`) — every Bitext gold intent must
  resolve to a leaf in the mined taxonomy. This is the taxonomy's external validity check.
- **Module 4** (`tests/unit/orchestration/`) — router accuracy and slot-extraction
  fixtures.
- **Module 6** (`evals/deepeval_suite.py`, `evals/judge.py`) — offline scoring and judge
  calibration.

## Needed by

**Phase 3 exit, now.** This is what makes the mined taxonomy's coverage number mean
something. Without it, "coverage 0.85" is a statement about the corpus agreeing with
itself.

## Layout, as verified against the live download

```
data/raw/bitext/
├── customer_support.csv     # the published file, unmodified -- this is what the adapter reads
└── customer_support.jsonl   # row-for-row mirror, for tooling that would rather not parse 19 MB of CSV
```

```bash
uv run python scripts/fetch_datasets.py --corpus bitext
```

The published columns are `flags, instruction, category, intent, response` — exactly what
`BitextAdapter` expects, so this corpus needed no reshaping at all. 26,872 labelled pairs,
**27 intents across 11 categories**. Slot values appear as `{{Order Number}}`-style
placeholders, so the corpus is pre-sanitised.

`customer_support.csv` is the file of record; the JSONL is a convenience mirror and
nothing reads it yet.

The adapter also accepts `utterance`/`query`/`text` for the caller column and
`answer`/`reply` for the response, and slugifies `SCREAMING_SNAKE` labels into dotted
lowercase ids.

## Note on ingestion

Bitext is **not** admitted for `transcript_ingestion` — `scripts/ingest.py` will refuse
it, by design. It is read directly by the eval suites, which compare its gold labels
against the taxonomy mined from AIxBlock.
