# data/raw/aixblock — the mining corpus

**AIxBlock 92k Call Center Scripts.** Unlabelled, high volume, conversational.

## Permitted roles

| Role | Why this corpus |
|---|---|
| `transcript_ingestion` | Multi-turn text normalises cleanly into `CallLog`. |
| `redaction_corpus` | The only corpus large enough to measure redaction p99 honestly. |
| `intent_mining` | Unlabelled, so HDBSCAN discovers structure rather than confirming it. |

## Denied roles

`nlu_ground_truth`, `slot_extraction_ground_truth`, `judge_evaluation` — there are no
labels here, so any "accuracy" computed against it would be circular.

`dialogue_context_benchmark`, `state_graph_benchmark` — these are scripts, not captured
speech; turn timing and real disfluency are absent.

## Where it is used

- **Module 1** (`ingestion/adapters/aixblock.py`) — scripts → `CallLog`.
- **Module 2** (`tests/latency/`) — redaction throughput and p99 measurement.
- **Module 3** (`scripts/mine_taxonomy.py`) — the L1/L2/L3 taxonomy is mined from this
  corpus and nothing else.

## Needed by

**Phase 3, now.** This is the blocking corpus: without it there is no real
`domains/retail/taxonomy.json`, only a synthetic one that proves the code runs.

## Layout, as verified against the live download

```
data/raw/aixblock/
├── retail.jsonl             # hydrated by scripts/fetch_datasets.py
├── healthcare.jsonl
└── .archives/               # the downloaded ZIPs, kept so a re-run is free
```

Fetch it with:

```bash
uv run python scripts/fetch_datasets.py --corpus aixblock
uv run python scripts/fetch_datasets.py --corpus aixblock --include-large   # + 826 MB
```

**The upstream shape is not what this file used to assume.** The corpus publishes twelve
themed ZIP archives of per-call AssemblyAI transcripts, not JSONL of tagged turns:

```
{"text": "...",            # ONE string, both parties merged, no turn boundaries
 "confidence": 0.96,
 "audio_duration": 749,    # seconds
 "words": [{"text": "Hello", "start": 240, "end": 640,
            "confidence": 0.59, "speaker": null}, ...],
 "redacted_pii_policies": [...]}
```

`speaker` is **null on every word of every archive sampled** — the corpus has word-level
timing but no diarization. `scripts/fetch_datasets.py` therefore recovers turns from
inter-word silence (`--gap-ms`, default **2000 ms**) and attributes them with a
conservative agent-boilerplate filter, then writes the `turns` shape `AixBlockAdapter`
already reads. **No change to `adapters/aixblock.py` was needed.** ADR-0016 has the
trade-off; ADR-0013 has what these calls actually are. Read both before trusting a volume
number from this corpus.

> **The gap default was 500 ms and that was wrong.** 500 ms splits breath groups, not
> turns — 200 segments per call, median 7 words — and mining them returned 11.3%
> coverage. 2000 ms gives a median of 18 words. If you need the old behaviour for
> barge-in or latency work, pass `--gap-ms 500`; the right value depends on the consumer.

### These two files are insurance calls

`retail.jsonl` and `healthcare.jsonl` are **misnomers.** The names came from the ZIP
filenames, which describe the BPO campaign that recorded the calls, not the vertical they
belong to. Profiling 300 calls per file:

| file | insurance | medicare | delivery | order | retail goods |
|---|---|---|---|---|---|
| `retail.jsonl` | **100%** | 5% | 13% | 0% | 0% |
| `healthcare.jsonl` | 84% | **89%** | 7% | 0% | 0% |

Both are life- and health-insurance sales and servicing. They are mined into
`domains/insurance/`, never into `retail` (ADR-0013). The names persist only because
partitions on disk reference them; the rename is future-scoped 9.13. **Do not infer a
domain pack from these filenames** — the manifest in `scripts/aixblock_transcripts.py`
carries the real `campaign` for each archive, and `output` is only a path.

### The archive names are not what the calls are about

Profiled over 300 calls per archive. Three of the published names have now been
contradicted by their own contents:

| archive | name suggests | measured |
|---|---|---|
| `customer_service_general_inbound` | general customer service | 97% life insurance / final expense |
| `automotive-stereo-inbound` | car audio | 3% car audio, 18% auto insurance — no dominant topic |
| `medical_equipment_outbound` | medical equipment | **89% Medicare coverage**, 12% equipment |
| `automotive_and_healthcare_insurance_inbound` | auto + health insurance | 47% Medicare, 9% auto — mixed |

`scripts/aixblock_transcripts.ARCHIVES` now carries a measured `campaign` and a
`verified` flag. An archive nobody has profiled reads `"unverified"` rather than a topic
guessed from its file name — that guess is how two of the four above were mislabelled in
the first place, by this repo rather than by AIxBlock.

**Profile before concluding anything about a cut of this corpus.** It has misdescribed
itself at the file level (ADR-0013), the role level (ADR-0015) and now the archive level.

### What is in `data/raw/` now

Default archives (4 of 12), at the 2000 ms gap:

| | `retail.jsonl` | `healthcare.jsonl` |
|---|---|---|
| campaigns (measured) | **life insurance 97%**, then one archive with no dominant topic (3% car audio despite its name) | **Medicare 89%** despite "medical_equipment" in its name, then a mixed insurance archive (47% Medicare) |
| dialogues | 1,959 | 2,531 |
| turns | 80,225 | *(not re-segmented)* |
| median words/utterance | 20 | — |

4,498 transcripts read at 500 ms, 0 unparsable. At 2000 ms, 8 retail calls fall below the
4-turn floor and are dropped. `healthcare.jsonl` on disk is still the 500 ms segmentation —
re-run the fetcher to bring it in line.

Because `--source aixblock` defaults to the whole directory, **pass `--path` explicitly**
or both files are pooled into one ingest run:

```bash
uv run python scripts/ingest.py --source aixblock --domain insurance \
  --path data/raw/aixblock/retail.jsonl --out data/interim-insurance
```

## Then

```bash
uv run python scripts/ingest.py --source aixblock --domain retail
uv run python scripts/mine_taxonomy.py --domain retail --dry-run     # cluster, no LLM spend
uv run python scripts/mine_taxonomy.py --domain retail
```

Start with `--dry-run`: it clusters and reports cluster count, coverage and size
distribution without calling a model. Tune `--min-cluster-size` there first — one LLM
call per cluster is the entire cost of mining, so it is worth getting the cluster count
right before paying for labels.

**Suggested starting point for ~92k scripts.** Expect roughly 150k–250k caller turns
after the caller-only and minimum-length filters. `--min-cluster-size 50` is a reasonable
first guess (roughly 0.03% of the corpus); too low fragments one intent across a dozen
clusters, too high merges distinct intents. Aim for 40–120 clusters and coverage above
0.75. Embedding 200k utterances with `bge-large-en-v1.5` is the slow step; it is cached
in `data/processed/embeddings.npz`, so re-tuning the clusterer is cheap after the first
pass.
