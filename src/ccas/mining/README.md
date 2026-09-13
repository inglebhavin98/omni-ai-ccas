# `mining/` — M3: corpus → `IntentTaxonomy`

Embed caller utterances, cluster them, have an LLM name the clusters, and emit an
L1/L2/L3 taxonomy. Pure functions; I/O at the edges.

## Files

| File | Holds |
|---|---|
| `embedder.py` | `SentenceTransformerEmbedder`, `HashingEmbedder`, `EmbeddingCache` |
| `cluster.py` | `HdbscanClusterer`, `centre_vectors`, `centroids`, `representatives` |
| `labeler.py` | `IntentLabeler` — cluster exemplars → name, description, `is_intent` |
| `build_taxonomy.py` | `TaxonomyBuilder`, `collect_utterances` |
| `homogeneity.py` | Between-call overlap with a null. Read `lift`, never raw overlap |
| `feasibility.py` | Pre-mining checks |
| `preflight.py` | Refuses to load an encoder that will not fit in memory |

## What counts as signal

Clustering a corpus does not produce a taxonomy; it produces clusters, and most of the
biggest ones are not intents. Three filters, in order
([ADR-0008](../../../docs/adr/0008-mining-signal-selection.md)):

1. **Caller turns only** — agent lines describe the resolution, not the reason for contact
2. **A minimum length** — "yes", "okay", "mm" carry no intent
3. **`is_intent` from the labeler** — greetings, closings, hold acknowledgements excluded

Filler is excluded from volume as well as structure, so `share_of_total` is measured
against intent-bearing utterances.

## Two defaults that were wrong, and are now not

- `min_samples` defaulted to `None`, which HDBSCAN reads as `min_cluster_size` — its most
  conservative setting. Correcting it to **5** moved coverage 9.8% → 18.5% on the same
  corpus. It is the largest single lever measured.
- Embeddings are **mean-centred** before clustering. bge-large is strongly anisotropic
  (‖mean‖ / mean‖x‖ = 0.72 measured), which compresses every pairwise distance into a
  narrow band. The corpus mean is stored on the result and on the taxonomy, because a
  single live utterance cannot recompute it — the mean of one vector is itself.

## Read coverage with suspicion

Coverage is not a quality metric on its own. Three explanations for the 18.8%–28.1% range
measured across two archives of one corpus — call direction, campaign variety, lexical
repetition — have each been tested and refuted. Ship the coverage figure alongside any
taxonomy, never the taxonomy alone.

```bash
uv run python scripts/mine_taxonomy.py --domain insurance --source aixblock --dry-run
uv run pytest tests/test_module_3_mining.py -q
```
