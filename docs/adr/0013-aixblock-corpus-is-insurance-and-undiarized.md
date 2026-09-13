# ADR-0013 — The AIxBlock corpus is insurance sales calls, and it is not diarized

- **Status:** accepted
- **Date:** 2026-09-13
- **Supersedes:** none
- **Amends:** [0008](0008-mining-signal-selection.md) (the caller-turn filter)
- **See also:** [0016](0016-aixblock-turn-recovery.md) — the segmentation gap that
  produced the fragments this ADR calls sub-utterance

## Context

Module 3 was ready to mine its first real taxonomy. Before running it I profiled
`data/raw/aixblock/`, because a taxonomy is the one artefact in this system that looks
authoritative whether or not it is. Two findings stopped the run.

### Finding 1 — the filenames are wrong

Topic-word profiling over 300 calls per file:

| file | insurance | billing | account | medical | order | warranty | delivery |
|---|---|---|---|---|---|---|---|
| `retail.jsonl` | 100% | 76% | 66% | 32% | 30% | 12% | **2%** |
| `healthcare.jsonl` | 84% | — | 20% | 38% | 30% | — | — |

Both files are life-insurance sales and servicing calls. `retail.jsonl` mentions
delivery in 2% of calls; the word "order" that lifts it to 30% is almost always "in
order to". The `order`/`warranty` hits are incidental vocabulary, not retail contacts.

### Finding 2 — the speaker labels are meaningless

Sampling 400 records: `diarization=pause_segmented` on every one, and
`speakers={'customer': 85321, 'agent': 627}` — 99.3% of turns carry the same label. The
transcripts plainly alternate between two speakers; the publisher segmented them by
silence, not by voice, and then labelled every segment with the same value.

This matters because ADR-0008 chose "caller utterances only" as the mining signal,
specifically so that agent stock phrases ("let me look that up") would not become the
corpus's largest clusters. On this corpus that filter is inoperative: it admits
everything.

## Decision

**1. Mine this corpus into a new `insurance` domain pack, not into `retail`.**

`domains/insurance/` is created with its own tools (`get_policy_status`,
`get_premium_schedule`, `request_callback_from_advisor`), queues (including a licensed
advisor line), reference patterns and a 7-year transcript retention. A taxonomy mined
from these calls and written to `domains/retail/taxonomy.json` would be an insurance
taxonomy wearing a retail label — worse than shipping no taxonomy, because every
downstream consumer would trust it.

**2. Raise the mining length floor, and treat it as a mitigation rather than a fix.**

`--min-chars` (default 8, set to 60 for this corpus) drops fragments too short to carry a
reason for contact. It works, and it is second-best: it discards long-but-fragmented
speech along with the noise. ADR-0016 attacks the same problem at the source by widening
the segmentation gap, and its measurement over 60 calls is the reason this is written as a
mitigation:

| gap | turns/call | utterances | median words | ≤6 words |
|---|---|---|---|---|
| 500 ms | 200 | 10,212 | 7 | 45% |
| 1000 ms | 123 | 6,623 | 10 | 35% |
| 2000 ms | 53 | 3,039 | 18 | 21% |
| 3000 ms | 27 | 1,539 | 33 | 15% |

At 500 ms the splits fall on breath groups, which is where 182 "turns" per call come from.
At 2000 ms the median utterance is 18 words and nothing is thrown away. Re-segmenting is
the better lever; the length floor stays because it also guards corpora whose segmentation
cannot be changed.

**3. The adapter records unreliable diarization rather than correcting it.**

`_diarization_is_reliable()` in `src/ccas/ingestion/adapters/aixblock.py` marks a record
unreliable when one speaker holds more than 95% of turns in a call of four turns or
more, and writes `metadata["diarization_reliable"] = False`. It does *not* attempt to
re-diarize. Guessing turn boundaries from text would put invented structure into a
corpus that downstream stages treat as observed fact.

**4. The labeler carries the agent-speech filter as a backstop.**

Because the speaker-label filter cannot work here, the cluster-labelling prompt gains an
explicit rule: a cluster that is the agent speaking — scripted openings, disclosures,
probing questions, hold messages, anything offering or explaining a product — is
`is_intent: false`. This is the same intent as ADR-0008's filter, moved from a
deterministic gate to a model judgement, which is strictly weaker and is why the gate
stays in place for corpora that *are* diarized.

ADR-0016's `_AGENT_CUES` runs first and is the deterministic half. It is not sufficient
on its own: it tags 1.3–3% of turns, against a corpus where roughly half the speech is
the agent's. So "primary" here means *first*, not *enough* — the prompt rule is doing most
of the real work on this corpus, and that is the weakness this decision accepts rather
than one it resolves.

## Alternatives considered

- **Re-diarize with a speaker-change model.** Correct in principle, and out of scope: it
  adds a model to the locked stack (Rule 5) to fix one publisher's export.
- **Alternate turns heuristically.** Cheap and wrong. Consecutive same-speaker turns are
  common in these calls; the result would be confidently mislabelled.
- **Attribute by boilerplate cues instead of by voice.** Taken, by ADR-0016, and it is
  the reason this ADR's rule-in-the-prompt is a backstop rather than the whole answer.
  `_AGENT_CUES` matches generic contact-centre phrasing and is deliberately asymmetric: a
  missed agent turn adds boilerplate a reviewer can prune from the taxonomy, while a
  wrongly excluded caller turn silently deletes the intent being mined for. It tags only
  1.3–3% of turns, so it does not make the published `speaker` field trustworthy — which
  is why `diarization_reliable` still says what it says.
- **Mine into `retail` and rename later.** The rename never happens, and in the meantime
  the retail pack's confidence thresholds and queues would be tuned against insurance
  traffic.
- **Drop AIxBlock.** It is still the only unlabelled conversational corpus available, and
  ADR-0004 gives it the mining role precisely because it is unlabelled. The data is
  usable; only its metadata is wrong.

## Consequences

- A third domain pack now exists, so Rule 1's "works for both packs" bar becomes three.
  That is a feature: `insurance` is the first pack created from observed data rather than
  from a design sketch.
- **The filenames are wrong and are nobody's provenance.** An earlier draft of this ADR
  argued for keeping them because they were the names AIxBlock published. They are not:
  the published artefacts are ZIPs (`customer_service_general_inbound.zip`,
  `medical_equipment_outbound.zip`), and `retail.jsonl` / `healthcare.jsonl` were derived
  by `scripts/fetch_datasets.py` from an archive→domain manifest built off the ZIP names
  without profiling their contents. So the argument for keeping them was based on a fact
  that is not true, and by this ADR's own reasoning a file named `retail.jsonl` holding
  100% insurance calls is precisely the mislabelling it objects to. The rename belongs to
  the fetcher's manifest, which owns the naming; it is listed in
  `docs/future-scoped-work.md` rather than done here, because changing it mid-run would
  invalidate partitions already on disk.
- The `diarization_reliable` flag is recorded but not yet acted on anywhere except this
  decision. A future diarized corpus should assert it before relying on speaker labels;
  that check is listed in `docs/future-scoped-work.md`.
- ADR-0004's rationale for AIxBlock ("unlabelled") is unaffected — these calls carry no
  intent labels. Its rationale for NatCS is separately in question and tracked in
  `docs/future-scoped-work.md`.
