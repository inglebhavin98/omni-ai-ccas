# ADR 0008 — What counts as signal when mining a taxonomy

Status: accepted · 2026-09-12

## Context

Clustering a corpus of call transcripts does not, on its own, produce an intent
taxonomy. It produces clusters, and most of the biggest ones are not intents.

Measured on a 300-call corpus (773 caller utterances, `bge-large-en-v1.5`, HDBSCAN
`min_cluster_size=12`): 13 clusters, 100% coverage — and **the single largest cluster,
at 300 members, was the closing line** ("No, that's everything. Thanks."). Left in, it
would have been the taxonomy's highest-volume node, would have landed in
`IMMEDIATE_MIGRATION`, and would have topped the migration sequencing the 2×2 exists to
drive.

Three filters were considered: positional (mine only the first caller turn), lexical
(a stop-phrase list), and letting the labeler decide.

## Decision

Three filters, applied in order:

1. **Caller turns only.** Agent lines describe the resolution, not the reason for
   contact. Including them produces clusters of stock phrases like "let me look that up".
2. **A minimum length** (8 characters). "Yes", "okay" and "mm" carry no intent and would
   otherwise form the largest cluster in any real corpus.
3. **`is_intent` from the labeler.** The model marks a cluster as conversational filler —
   greetings, closings, hold acknowledgements, confirmations — and the builder excludes
   it from the taxonomy.

A positional filter was rejected: intents frequently surface on the third turn, after
authentication. A stop-phrase list was rejected because it is vertical-specific by
nature, which Rule 1 forbids in the core.

Filler is excluded from **volume as well as structure**: `share_of_total` is measured
against intent-bearing utterances, and `coverage` states how much *intent-bearing* text
was successfully clustered. Counting filler as noise would understate the miner; counting
it as coverage would overstate it.

The counts survive in `clusterer_params` (`clusters_found`,
`clusters_labelled_filler`, `filler_utterances`) so the decision is auditable rather than
invisible.

## Consequences

- The taxonomy describes reasons for contact, not conversational surface.
- A labelling failure now has two ways to go wrong: naming filler as an intent (noise in
  the taxonomy) or marking a real intent as filler (a silent gap). The second is worse
  and harder to see, so the counts are recorded and `build()` raises if *every* cluster
  comes back as filler.
- `HashingEmbedder` exists for CI, and `build_embedder` refuses to fall back to it unless
  `--allow-hashing-embedder` is passed. A taxonomy silently mined from lexical overlap
  would look entirely plausible and describe nothing.
- Two clusters labelled into the same leaf are **merged**, not disambiguated with a
  `_2` suffix: the clusterer split something the labeler considers one intent, and
  inventing a second name would put a fiction into the taxonomy.
- Parent (L1/L2) volumes are the sum of their children, never an independent estimate.
  An LLM asked to guess volume would guess, and the quadrant would stop meaning anything.
