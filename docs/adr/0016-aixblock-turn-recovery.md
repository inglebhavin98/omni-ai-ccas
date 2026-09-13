# ADR 0016 — Recovering turns from an undiarized corpus

Status: accepted · 2026-09-13
Amended: 2026-09-13 — the gap was wrong; see Decision 1 and the Correction below.
Renumbered from 0010 on 2026-09-13: written concurrently with, and unaware of,
[0010](0010-placeholder-vault.md), which was created first. Nothing referenced this
file under the old number outside this repo.
See also: [0013](0013-aixblock-corpus-is-insurance-and-undiarized.md), which establishes
what these calls actually are and why the length floor is a mitigation.

## Context

AIxBlock is the corpus the taxonomy is mined from (ADR-0004), and ADR-0008 makes
**caller turns only** the first and most important mining filter: agent lines describe
the resolution, not the reason for contact.

First contact with the live download shows the corpus cannot support that filter as
designed. The published shape is twelve themed ZIP archives of per-call AssemblyAI
transcripts, each holding:

```
{"text": "...", "confidence": 0.96, "audio_duration": 749,
 "words": [{"text": "Hello", "start": 240, "end": 640, "confidence": 0.59,
            "speaker": null}, ...],
 "redacted_pii_policies": [...]}
```

Two things follow. `text` is a single string with **both parties merged** — there are no
turn boundaries at all. And `speaker` is **null on every word of every file sampled**,
across an inbound archive, an outbound archive and a re-uploaded PII-redacted archive.
The dataset card advertises word-level timestamps and ASR confidence; it never claims
diarization, and none is present.

So the corpus offers turn *timing* but not turn *attribution*, and the ingestion adapter
expects neither: it looks for a `turns` list or a `Speaker: text` script, and finds
neither shape here.

Three options were considered for attribution:

1. **Alternate speakers at each boundary.** Rejected. Measured on the corpus, a single
   speaker frequently continues across a pause, which flips the parity of every
   subsequent turn in the call. It would be wrong roughly half the time and wrong
   silently.
2. **Tag every turn as the caller.** Honest about ignorance, but puts all agent and IVR
   speech into the mining pool, where stock phrases cluster tightly and outrank real
   intents — exactly the failure ADR-0008 was written to prevent.
3. **A conservative boilerplate filter.** Attribute a turn to the agent only when it says
   something only an agent or an IVR says; leave everything else with the caller.

## Decision

Turns are recovered from **inter-word silence**, and attributed by a **conservative
agent-boilerplate filter** — option 3. Neither step claims to be diarization.

- **Segmentation.** A gap of `>= 2000 ms` between the end of one word and the start of
  the next ends a turn.

  **This started at 500 ms and 500 ms was wrong.** The reasoning was sound and the unit
  was not: measured on the corpus the intra-word gap is 0 ms at the median and 320 ms at
  p90, so 500 ms does sit above within-turn pausing — but what it separates is *breath
  groups*, not turns. It produced 200 segments per call with a median of 7 words, 46% of
  them six words or fewer, and a first mining run returned **11.3% coverage, 88.7%
  noise**. HDBSCAN was right to reject them: a four-word fragment clears the 8-character
  floor while carrying no reason for contact.

  That 11.3% was also measured with `min_samples=None`, which HDBSCAN reads as
  `min_cluster_size` — its most conservative setting, since corrected to a default of 5
  (ADR-0017) — and on a cut that was **19.0% exact duplicates**, the highest of any
  corpus either session has clustered. Duplicates inflate coverage, so the true figure
  was worse than 11.3%, not better. The conclusion here holds and its magnitude was
  understated; the number should not be quoted as a measurement of the segmentation
  alone, because three things were wrong at once.

  Re-measured over 60 calls:

  | gap | turns/call | utterances | median words | ≤6 words |
  |---|---|---|---|---|
  | 500 ms | 200 | 10,212 | 7 | 45% |
  | 1000 ms | 123 | 6,623 | 10 | 35% |
  | 2000 ms | 53 | 3,039 | 18 | 21% |
  | 3000 ms | 27 | 1,539 | 33 | 15% |

  2000 ms is the knee: the median utterance becomes 18 words and the corpus shrinks 4.1x,
  which also relieves the clustering ceiling below. Re-hydrated and re-ingested at 2000 ms
  the retail file yields 1,959 calls and 72,619 mineable utterances, median 20 words, 20%
  under six — against 282,717 fragments at 500 ms.

  The general rule the first attempt missed: **the gap is chosen for the consumer, not
  for the acoustics.** A latency or barge-in measurement wants breath groups; mining wants
  utterances. Same signal, different threshold.
- **Attribution.** `scripts/aixblock_transcripts._AGENT_CUES` matches generic
  contact-centre and IVR boilerplate — "thank you for calling", "your call may be
  recorded", "press one for", "may I have your", "is there anything else I can help".
  A match makes the turn the agent's; everything else stays with the caller.
- The asymmetry is deliberate. A missed agent turn adds boilerplate to the mining pool,
  where a reviewer can see it in the taxonomy and prune it. A wrongly excluded caller
  turn silently deletes the intent the corpus was mined for. Only the first failure is
  visible, so the filter is tuned to make only that one.
- Every phrase in the cue list is generic call-centre vocabulary. No vertical's words
  appear, so Rule 1 holds and the same filter serves every pack.
- Record ids are `sha256(archive/member)[:16]`. AIxBlock file names embed the dialled
  number — `..._6149286164_4712234-all_transcript.json` — and carrying that into
  `record_id` would put a phone number into every `CallLog` and every log line
  downstream (Rule 2).
- Output is written as a `turns` list of `{speaker, text}`, which
  `AixBlockAdapter` already reads. **No change to `src/ccas/` was needed to hydrate the
  corpus**, and none was made.

## Consequences

- **ADR-0008's first filter is degraded on this corpus, and that is now a known
  quantity rather than an assumption.** Measured over the hydrated sample, the filter
  attributes 1.3% of turns in `retail.jsonl` and 3.0% in `healthcare.jsonl` to the agent,
  and fires at least once in 76% and 90% of calls respectively. It removes scripted
  openings and IVR menus; it does not separate a conversation. ADR-0013 reaches the same
  conclusion from the published `speaker` field and keeps `diarization_reliable=False`,
  which is the right call: 1.3% tagging does not make the field trustworthy.
- The remaining two filters therefore carry the load: the 8-character minimum, and the
  labeler's `is_intent` filler judgement. The filler counts recorded in
  `clusterer_params` become the primary evidence that mining worked, not a footnote.
- A taxonomy mined from this corpus must be read as *reasons for contact as expressed in
  the call*, with some agent phrasing mixed in — not as a clean caller-only distribution.
  Any migration sequencing built on its volumes inherits that caveat.
- Turn counts here are not comparable to a diarized corpus's: they are pause boundaries,
  not speaker changes, and the threshold that defines them is a tuning choice.
- `start_ms`/`end_ms` are preserved in `data/raw/` but dropped at the adapter boundary,
  which reads only `speaker` and `text`. Recovering them is future-scoped work (9.x), not
  a silent loss.
- Over-fine segmentation also multiplied the clustering cost: at 500 ms, 1,967 calls
  yielded 282,717 utterances (228,914 unique) against 72,619 at 2000 ms. HDBSCAN's time
  is ~n^1.91 here (measured: n=4,000 → 26.1 s, n=8,000 → 98.4 s), so a 4.1x reduction in
  n is roughly a 14x reduction in clustering time.

  An earlier version of this bullet said the 500 ms corpus was "~24 hours" and implied a
  hard ceiling. Both are withdrawn. The 24-hour figure was extrapolated from a timing
  taken while the machine was swapping, and the ceiling rested on a claim — since refuted
  in future-scoped 9.8 — that hdbscan allocates an n x n matrix above 60 dimensions. It
  does not: dense input with a `FAST_METRICS` metric dispatches to `_hdbscan_prims_kdtree`,
  whose memory is O(n·d). At 2000 ms the full corpus is ~112 minutes and ~0.55 GiB, which
  is a budget, not a wall. Mining still runs on a sample, but for wall-clock convenience
  rather than necessity.

  The lesson is the same one the gap taught: a number measured under contention describes
  the machine that day, not the algorithm.
- If a diarized release appears, the cue filter should be deleted rather than tuned. It
  is a workaround for a missing field, not a component worth keeping.


## Correction — the archive→domain manifest was wrong

The first version of `ARCHIVES` in `scripts/aixblock_transcripts.py` assigned each ZIP a
`domain` by reading its name: `customer_service_general_inbound` became `retail`,
`medical_equipment_outbound` became `healthcare`. Nothing profiled the transcripts.

They are insurance sales calls. Independent topic profiling over 300 calls per file:
`retail.jsonl` is 100% insurance, 0% order, 0% retail-goods vocabulary; `healthcare.jsonl`
is 89% Medicare. ADR-0013 found this before any taxonomy was written, which is the only
reason it did not ship as `domains/retail/taxonomy.json`.

The manifest now separates the two facts it had conflated: `campaign` is what the calls
are, measured; `output` is only a filename. `retail.jsonl` and `healthcare.jsonl` keep
their names for now because partitions on disk reference them — the rename is
future-scoped 9.13, not a decision to keep them.

The lesson is narrow and worth stating: **a corpus's file names are a claim, not
evidence.** Profile before routing, especially when the routing decides which pack's
thresholds and queues get tuned against the data.
