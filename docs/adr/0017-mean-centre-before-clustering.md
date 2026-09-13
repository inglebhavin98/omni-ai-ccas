# ADR-0017 — Mean-centre embeddings before clustering

- **Status:** accepted
- **Date:** 2026-09-13
- **See also:** [0008](0008-mining-signal-selection.md), [0013](0013-aixblock-corpus-is-insurance-and-undiarized.md)

## Context

Mining the insurance corpus produced 12.1% coverage — HDBSCAN declined to assign 88% of
21,032 caller utterances. A taxonomy describing a tenth of traffic is worse than none,
because it ships looking exactly as authoritative as one describing most of it.

Two explanations were live, and they call for opposite actions:

1. **The parameters are wrong.** Tune and the coverage appears.
2. **The corpus has no intent structure at this granularity.** Then HDBSCAN is behaving
   correctly and the honest output is no taxonomy.

Parameter search answered the first partly. `min_cluster_size` is not the lever: cutting
it 25 → 15 produced 2.5× the clusters for 1.7 points of coverage, meaning the new clusters
were carved out of existing ones rather than out of the noise. `min_samples` is the lever
— it defaults to `min_cluster_size`, HDBSCAN's most conservative core-distance setting, and
setting it to 5 moved full-corpus coverage 12.1% → 20.4%.

That left the second explanation, and the evidence offered for it was a cosine
distribution: mean 0.513, sd 0.089, a narrow unimodal band. **That reasoning was wrong**,
and a peer session caught it before it was recorded. A narrow band well above zero is also
the signature of an **anisotropic** embedding space — one where every vector shares a large
common direction, so every pair starts at a high baseline and semantic variation is a thin
layer on top. That band looks the same whether or not the corpus has structure.

## The control, and the trap in it

The obvious null is to permute each dimension independently across rows: every marginal is
preserved, all cross-dimension correlation — all semantics — is destroyed.

**The mean cannot discriminate.** A column-wise permutation leaves each column's mean
unchanged, so the corpus mean vector survives intact, and mean pair cosine is
`||mu||² / E||x||²` — identical in raw and null by construction. Verified on synthetic
spaces with known ground truth (256 dims, 4,000 points, strong common direction):

| space | condition | mean | sd | p99 | ≥0.80 |
|---|---|---|---|---|---|
| planted clusters | raw | +0.168 | 0.276 | +0.896 | 12.48% |
| planted clusters | null | +0.168 | 0.064 | +0.313 | 0.00% |
| no structure | raw | +0.336 | 0.052 | +0.452 | 0.00% |
| no structure | null | +0.336 | 0.052 | +0.453 | 0.00% |

Means agree to three decimals in **both** worlds. The mean vector is preserved to 5e-15.
The discriminator is **spread and tail**.

## What the corpus actually says

6,000 utterances sampled from the 21,032 cached vectors, 400k random pairs:

```
 condition     mean       sd      p50      p99    p99.9    >=0.80    >=0.90
       raw   +0.513   0.0886   +0.507   +0.749   +0.856   0.3416%   0.0328%
      null   +0.513   0.0206   +0.514   +0.560   +0.575   0.0000%   0.0000%
   centred   +0.000   0.1338   -0.017   +0.412   +0.663   0.0255%   0.0058%
```

`||mean vector|| / mean ||vector|| = 0.716` — **72% of the geometry is one shared
direction.** The anisotropy is severe and the concern was justified.

And the structure is real anyway. Raw sd is **4.3× the null's**, and the null contains
**zero** pairs above 0.80 where the real data has 0.34%. Semantic structure exists over and
above the shared direction.

So neither original explanation was right. The corpus has structure; the encoder's offset
was hiding a large part of it from the clusterer.

## Decision

`HdbscanClusterer` subtracts the corpus mean and re-normalises before fitting.
`DEFAULT_CENTRE = True`; `centre=False` restores the old behaviour; the flag lands in
`params` and therefore in `IntentTaxonomy.clusterer_params`, so a taxonomy records the
space it was clustered in.

```python
centred = vectors - vectors.mean(axis=0)
return centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
```

Measured at n=6,000, `min_cluster_size=25`, `min_samples=5`: **13.6% → 18.1% coverage**,
11 → 16 clusters.

### Correction: that gain does not survive scale

The 18.1% above was measured at n=6,000 and quoted as the justification. At full corpus it
is much smaller, and the honest table is this one — n=20,489 unique vectors, duplicates
removed, same parameters:

| centre | min_samples | clusters | coverage | largest cluster |
|---|---|---|---|---|
| off | None (= `min_cluster_size`) | 14 | 9.8% | 488 |
| off | 5 | 39 | 18.5% | 801 |
| **on** | **5** | **50** | **19.7%** | **704** |

`min_samples` is the lever: **9.8% → 18.5%**, nearly a doubling. Centring adds
**1.2 points** on top of it, not the 4.5 the small sample implied. The benefit shrinks with
n, which is what you would expect — at small n the density estimate is poor and the
anisotropic compression costs more; with enough points HDBSCAN resolves structure despite
the offset.

So the coverage case for centring is weak, and I overstated it by quoting a magnitude
measured at one scale as though it were general.

**The decision still stands, on two grounds that do survive.** First, centring resolves the
corpus more finely at the same coverage: 50 clusters against 39, with the largest cluster
*shrinking* from 801 to 704 (3.9% → 3.4% of the corpus in one node). More intents, less mass
dumped in one bucket, which is what a taxonomy is for. Second — and this is independent of
coverage entirely — the centroid-space and routing-margin argument below is about how
clusters are *described* and compared, not how many are found, and nothing here touches it.

The cached vectors on disk stay **raw**. Centring is a property of the clustering step, not
of the encoding, so the cache remains reusable under a different transform.

### The mean is part of the taxonomy

Centring inside `fit_predict` and discarding the mean would leave the artefact
incoherent: membership decided in one space, the stored `IntentNode.centroid` computed in
another, and nothing recording the map between them. `clusterer_params["centre"] = True`
says *that* a transform happened, not *what* it was.

That is not fixable at call time. **A single live vector cannot centre itself** — the mean
of one vector is itself, so centring it gives zero. The transform is reconstructible only
from a stored mean.

So `ClusterResult` carries `embedding_mean`, `IntentTaxonomy` persists it, and centroids
and exemplars are both computed in `result.in_cluster_space(vectors)`. A validator rejects
a taxonomy whose centroid width disagrees with its mean, because that combination can only
describe two different spaces.

It is **stored rather than recomputed** because it is a property of the corpus that was
mined. Re-deriving it later from a different corpus, or from a sample, silently moves every
centroid.

Why it matters concretely — 6,000 utterances, 13 clusters, the same labels, centroids
computed from each space:

| centroids from | mean inter-centroid cosine | min | max |
|---|---|---|---|
| raw vectors | 0.696 | 0.529 | 0.861 |
| cluster space | −0.001 | −0.426 | 0.507 |

Raw centroids are all mutually similar — the closest pair at 0.861, none below 0.529 —
because averaging within a cluster *concentrates* the shared direction: the orthogonal
parts cancel and the common part does not. In cluster space the same 13 clusters are
essentially orthogonal. A nearest-centroid router over raw centroids would be choosing
between intents separated by a sliver on top of a large common component, which is exactly
the compression centring removed, reintroduced at inference.

Measured on the case that matters — 500 vectors held out of a 6,000 cut, clustered on the
rest, then routed **one at a time**, which is the shape a live router has and the shape
that cannot recompute a mean:

| routing | margin (best − 2nd), mean | median |
|---|---|---|
| raw vector vs raw centroids | 0.054 | 0.041 |
| mapped vector vs cluster-space centroids | 0.153 | **0.102** |

A median top-two gap of 0.041 is inside the variation an ASR alternative or a disfluency
produces, so a router would flip intents on rephrasing. That surfaces as *"the classifier
is unstable"* — debugged in the router, the prompt, the thresholds, anywhere but the space
the centroids were written in.

### The exemplars were worse than the centroids

`representatives()` picks the k members nearest their cluster's centroid, and it had the
same split. Under anisotropy, nearest-to-a-raw-centroid selects for proximity to the shared
direction — that is, for **the most generic members**, which on a sales corpus means the
most scripted ones. So the labeller was being shown the blandest utterances in every
cluster, which is precisely when `is_intent` is hardest to judge.

That compounds ADR-0013, where the agent-speech filter already leans on the labeller's
judgement because the corpus's speaker labels are unusable. Two independent pressures
toward naming clusters after boilerplate, from opposite ends of the pipeline.

Centroids only matter once something routes. Exemplars matter immediately.

Nothing reads `IntentNode.centroid` today, so the centroid half was latent rather than
live; the exemplar half was not. Both were caught by a peer session reviewing the change,
not by a test, and the tests exist now — including one that routes held-out vectors singly
and asserts the margin widens.

## Alternatives considered

- **Leave it and report low coverage.** What this ADR was nearly written to say. It would
  have recorded a property of bge-large as a finding about insurance calls.
- **Whitening / all-but-the-top.** Standard anisotropy corrections and probably stronger.
  Both introduce fitted parameters that must be stored and versioned with the taxonomy, and
  all-but-the-top needs a component count chosen per corpus. Mean-centring has no
  hyperparameter and no stored state. Revisit if 20% is still the ceiling.
- **A different encoder.** `bge-large-en-v1.5` is a locked-stack row (Rule 5); changing it
  is a separate ADR and should follow evidence that centring is insufficient.
- **Normalising inside the embedder.** Would poison the cache for any other consumer and
  make the stored vectors a lie about what the model produced.

## Consequences

- **Every mined taxonomy from here is clustered in a different space.** Nothing has shipped
  yet, so there is nothing to re-mine. Any future comparison against a pre-0017 taxonomy
  must check `clusterer_params["centre"]`.
- **`IntentTaxonomy.schema_version` goes 1.0 → 1.1**, adding `embedding_mean`. Additive and
  defaulted, so a 1.0 document still validates — but one that has centroids and no mean is
  a taxonomy nothing can safely route against. Migration note in `docs/tech-spec.md` §1.4.
- A taxonomy file grows by one float per embedding dimension: ~20 KB at 1024-d. Worth it.
- One existing test pins `centre=False`: its fixture's outlier is defined in the raw space,
  and subtracting a "corpus mean" from a handful of toy vectors absorbs it. That is a
  fixture artifact, and pinning it keeps the test measuring what it was written to measure.
- **Coverage lands at 19.7% on this cut, and that number does not generalise.** At matched
  n=10,388 the same parameters give **18.8% on `retail.jsonl` and 28.1% on
  `healthcare.jsonl`** — a 9.3-point spread between two slices of one corpus. 19.7% is a
  property of what was mined, not of the platform, the encoder, or AIxBlock.
- **I proposed an explanation and it was refuted.** I said these are outbound sales calls,
  so the caller is answering rather than asking. `healthcare.jsonl` is the outbound corpus
  on two independent measures (44.8%/70.0% pitch openings against 8.0%/5.5%) and it
  clustered **9.3 points higher**, with 68% more clusters and a smaller largest node. Every
  direction is against the hypothesis, so it is not a near miss. Recorded in
  `docs/future-scoped-work.md` 9.17 as tested and refuted, with the numbers, so it does not
  get re-proposed.
- A taxonomy over the top intents is still defensible — 50 clusters, none holding more than
  3.4% — provided the coverage figure ships next to it.
- **A magnitude quoted from a 6,000-point sample did not hold at 20,489.** The correction is
  in the Decision section rather than appended, because a reader who stops at the headline
  should get the corrected number. Nothing about the transform changed; only my claim about
  what it buys.
- The methodological lesson is worth more than the transform: a distribution that looks
  like evidence for a claim about the data may be evidence about the instrument. **The
  control has to be able to come out the other way.** The first null proposed here could
  not have, and it took a second pair of eyes to notice.
