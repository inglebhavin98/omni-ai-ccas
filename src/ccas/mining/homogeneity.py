"""How much do callers repeat each other? The context ``coverage`` needs.

Density-based clustering needs density. A corpus of near-identical calls produces dense
regions and scores superb coverage while yielding one useful intent; a corpus where every
call is lexically unique produces a diffuse space and scores badly however well-structured
its intents are. So **coverage is not a quality metric**, and a migration decision made on
it alone would be made on the wrong number.

Measured across two single-archive cuts of one corpus: between-call overlap 0.130 against
0.081, and coverage 28.1% against 18.8% -- the more repetitive corpus scored better. Two
points and one direction, so this ships as *context reported alongside* coverage, never as
a correction applied to it.

Getting the *measure* right took three attempts, and the first two were size controls that
smuggled the treatment in with them:

1. Sampling K **tokens** per call. A repetitive call yields fewer distinct words from K
   tokens, which is exactly the quantity being measured.
2. Sampling K **distinct words** per call. Better, and still wrong: two *identical* calls
   drawn from a 245-word vocabulary score only 0.44 at K=150, while identical calls from a
   160-word vocabulary score 0.88. The ceiling depends on vocabulary size, so a corpus of
   terse callers outscores a corpus of wordy ones whatever either is saying.

What works is not a size correction at all but a **null**: measure the overlap, then
measure it again on a corpus where each call's vocabulary is redrawn at random from the
corpus-wide word distribution at that call's own size. The null absorbs every size effect
by construction, because it is computed pair-by-pair on the same geometry. The reported
number is the ratio -- how much more callers repeat each other than chance alone predicts.

``lift == 1.0`` means a corpus no more repetitive than random. Higher means real shared
structure, which is what makes clusters findable and coverage high.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from ccas.schemas.taxonomy import CorpusHomogeneity

__all__ = [
    "DEFAULT_PAIRS",
    "MIN_QUALIFYING_CALLS",
    "MIN_VOCABULARY",
    "STOPWORDS",
    "content_vocabulary",
    "measure_homogeneity",
]

#: Smallest vocabulary a call must have to be measured at all. Not a sampling budget --
#: nothing is subsampled -- just a floor below which a "call" is too thin to compare.
MIN_VOCABULARY = 40

#: Random call pairs sampled. Enough for a stable mean; the measure is cheap either way.
DEFAULT_PAIRS = 20_000

#: Below this many qualifying calls the number describes a handful of outliers.
MIN_QUALIFYING_CALLS = 30

#: Generic English. No domain vocabulary (Rule 1) -- this list must serve every pack.
STOPWORDS = frozenset(
    (
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "back",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "but",
        "by",
        "can",
        "come",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "for",
        "from",
        "further",
        "get",
        "give",
        "go",
        "going",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "let",
        "like",
        "look",
        "make",
        "may",
        "me",
        "might",
        "more",
        "most",
        "must",
        "my",
        "need",
        "no",
        "not",
        "now",
        "of",
        "off",
        "ok",
        "okay",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "really",
        "right",
        "said",
        "same",
        "say",
        "see",
        "shall",
        "she",
        "should",
        "so",
        "some",
        "such",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "think",
        "this",
        "those",
        "through",
        "to",
        "too",
        "uh",
        "um",
        "under",
        "until",
        "up",
        "us",
        "very",
        "want",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "yeah",
        "yes",
        "you",
        "your",
    )
)

#: Three or more letters, so digits and punctuation drop out with the short function words.
_WORD = re.compile(r"[a-z']{3,}")

#: `[ACCOUNT_REF_1]` and friends appear in every redacted call and would inflate overlap
#: toward 1.0 on a corpus that simply mentions people a lot.
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


def content_vocabulary(text: str) -> set[str]:
    """The distinct content words of one call. Pure."""
    without_placeholders = _PLACEHOLDER.sub(" ", text)
    return {w for w in _WORD.findall(without_placeholders.lower()) if w not in STOPWORDS}


def _mean_cosine(
    vocabularies: Sequence[frozenset[str]], left: np.ndarray, right: np.ndarray
) -> float:
    """Mean binary cosine over the given pairs: |A n B| / sqrt(|A| |B|)."""
    scores = [
        len(vocabularies[i] & vocabularies[j])
        / ((len(vocabularies[i]) * len(vocabularies[j])) ** 0.5)
        for i, j in zip(left, right, strict=True)
        if i != j
    ]
    return float(np.mean(scores)) if scores else 0.0


def _null_corpus(
    vocabularies: Sequence[frozenset[str]], rng: np.random.Generator
) -> list[frozenset[str]]:
    """Shuffle which call each word belongs to, preserving sizes and word frequencies.

    A degree-preserving shuffle of the call-word bipartite graph: the flat list of every
    (call, word) incidence is permuted and dealt back out into calls of their original
    sizes. Per-call vocabulary size is preserved exactly and each word stays exactly as
    common corpus-wide; only *co-occurrence* is destroyed, which is the structure being
    measured.

    Drawing from a frequency distribution instead -- the obvious alternative -- was tried
    and biased the null *upward*: weighted sampling without replacement concentrates on
    common words more than proportionally, making null pairs more alike than real ones and
    pushing lift below 1.0 on a corpus that was random by construction.

    Dealing can hand one call the same word twice; the duplicate collapses, so a null call
    is occasionally a word or two short. That shrinks the null slightly, which makes lift a
    marginally *conservative* estimate rather than a flattering one.
    """
    incidences = [word for vocabulary in vocabularies for word in vocabulary]
    rng.shuffle(incidences)
    out: list[frozenset[str]] = []
    cursor = 0
    for vocabulary in vocabularies:
        size = len(vocabulary)
        out.append(frozenset(incidences[cursor : cursor + size]))
        cursor += size
    return out


def measure_homogeneity(
    vocabularies: Sequence[set[str]],
    pairs: int = DEFAULT_PAIRS,
    seed: int = 17,
    min_vocabulary: int = MIN_VOCABULARY,
) -> CorpusHomogeneity:
    """How much more do callers repeat each other than chance predicts?

    Nothing is subsampled. The size effect is absorbed by the null rather than corrected
    for, because every correction attempted encoded the effect it was correcting.
    """
    rng = np.random.default_rng(seed)
    eligible = [frozenset(v) for v in vocabularies if len(v) >= min_vocabulary]
    qualifying_share = len(eligible) / len(vocabularies) if vocabularies else 0.0

    if len(eligible) < MIN_QUALIFYING_CALLS:
        # Unmeasurable is not zero: 0.0 reads as "no repetition", the opposite of unknown.
        return CorpusHomogeneity(
            observed_overlap=None,
            null_overlap=None,
            min_vocabulary=min_vocabulary,
            qualifying_share=round(qualifying_share, 4),
            calls_measured=len(eligible),
        )

    left = rng.integers(0, len(eligible), pairs)
    right = rng.integers(0, len(eligible), pairs)
    observed = _mean_cosine(eligible, left, right)
    null = _mean_cosine(_null_corpus(eligible, rng), left, right)
    return CorpusHomogeneity(
        observed_overlap=round(observed, 6),
        null_overlap=round(null, 6),
        min_vocabulary=min_vocabulary,
        qualifying_share=round(qualifying_share, 4),
        calls_measured=len(eligible),
    )
