"""Unsupervised clustering of utterance embeddings (Module 3).

HDBSCAN is the primary because it finds clusters of varying density and, critically,
*refuses to assign* points it cannot justify -- that noise label is signal. k-means
would force every utterance into a bucket and manufacture intents that do not exist.

Vectors arrive L2-normalised, so euclidean distance is monotonic in cosine distance and
the two produce the same partition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ccas.schemas.common import JsonValue

__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "NOISE_LABEL",
    "ClusterResult",
    "Clusterer",
    "HdbscanClusterer",
    "build_clusterer",
    "centre_vectors",
    "centroids",
    "corpus_mean",
    "representatives",
]

#: HDBSCAN's label for a point it declines to assign.
NOISE_LABEL = -1

#: Core-distance neighbourhood. Deliberately decoupled from ``min_cluster_size``:
#: leaving it to default means "as conservative as possible", which is a choice, not
#: a neutral starting point. Pass ``None`` to get the library's behaviour back.
DEFAULT_MIN_SAMPLES = 5

#: Subtract the corpus mean before clustering. Sentence encoders are anisotropic: their
#: vectors share a large common direction, which offsets every pair's similarity into a
#: narrow band and compresses the distances HDBSCAN works in. Measured on 6,000 real
#: utterances under bge-large, that shared direction is 72% of the geometry
#: (||mean|| / mean ||x|| = 0.716), and removing it moved coverage 13.6% -> 18.1%.
#: See docs/adr/0017-mean-centre-before-clustering.md.
DEFAULT_CENTRE = True

Vectors = NDArray[np.float32]
Labels = NDArray[np.int_]


@dataclass(slots=True)
class ClusterResult:
    labels: Labels
    probabilities: NDArray[np.float64]
    algorithm: str
    params: dict[str, JsonValue] = field(default_factory=dict)
    #: The corpus mean subtracted before fitting, or None when clustering was raw.
    #: Membership was decided in that space, so anything describing a cluster -- its
    #: centroid, its exemplars -- has to be computed there too, and anything compared
    #: against it later has to be mapped in. A single live vector cannot reconstruct this
    #: for itself: the mean of one vector is itself, and centring it gives zero.
    embedding_mean: tuple[float, ...] | None = None

    def in_cluster_space(self, vectors: Vectors) -> Vectors:
        """Put vectors into the space these labels were assigned in.

        Replays the recorded transform rather than recomputing a mean, so a later caller
        mapping in a handful of live utterances lands in the same space as the corpus
        that was mined -- not in a space defined by whatever it happens to hold.
        """
        if self.embedding_mean is None:
            return np.asarray(vectors, dtype=np.float32)
        return _recentre(vectors, np.asarray(self.embedding_mean, dtype=np.float32))

    @property
    def n_points(self) -> int:
        return int(self.labels.shape[0])

    @property
    def cluster_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(label) for label in set(self.labels) if label != NOISE_LABEL))

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_ids)

    @property
    def noise_count(self) -> int:
        return int(np.sum(self.labels == NOISE_LABEL))

    @property
    def noise_ratio(self) -> float:
        return self.noise_count / self.n_points if self.n_points else 0.0

    @property
    def coverage(self) -> float:
        return 1.0 - self.noise_ratio

    def indices_for(self, cluster_id: int) -> NDArray[np.int_]:
        return np.flatnonzero(self.labels == cluster_id)

    def sizes(self) -> dict[int, int]:
        return {cid: int(np.sum(self.labels == cid)) for cid in self.cluster_ids}


class Clusterer(ABC):
    algorithm: str

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def fit_predict(self, vectors: Vectors) -> ClusterResult: ...

    @property
    @abstractmethod
    def params(self) -> dict[str, JsonValue]: ...


class HdbscanClusterer(Clusterer):
    algorithm = "hdbscan"

    def __init__(
        self,
        min_cluster_size: int = 15,
        # None means "use min_cluster_size", HDBSCAN's own default and its most
        # conservative core-distance setting: it maximises the noise label. Measured on
        # 6,000 real utterances at min_cluster_size=25 -- None gave 8.4% coverage, 10 gave
        # 11.3%, 5 gave 14.0%, 3 gave 15.8%. Nearly a doubling, where cutting
        # min_cluster_size by 40% moved coverage 1.7 points. It is the strongest lever
        # here, so it is set explicitly rather than inherited.
        min_samples: int | None = DEFAULT_MIN_SAMPLES,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        centre: bool = DEFAULT_CENTRE,
    ) -> None:
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.centre = centre

    @property
    def available(self) -> bool:
        try:
            import hdbscan  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def params(self) -> dict[str, JsonValue]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "centre": self.centre,
        }

    def fit_predict(self, vectors: Vectors) -> ClusterResult:
        try:
            import hdbscan
        except ImportError as exc:
            raise RuntimeError("hdbscan is not installed; run `uv sync --extra mining`") from exc

        # Fewer points than a cluster needs: everything is noise, and saying so is more
        # honest than shrinking the parameter until something appears.
        if vectors.shape[0] < self.min_cluster_size:
            return ClusterResult(
                labels=np.full(vectors.shape[0], NOISE_LABEL, dtype=np.int_),
                probabilities=np.zeros(vectors.shape[0], dtype=np.float64),
                algorithm=self.algorithm,
                params=self.params,
            )

        mean = corpus_mean(vectors) if self.centre else None
        points = _recentre(vectors, mean) if mean is not None else vectors
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=False,
        )
        labels = model.fit_predict(np.asarray(points, dtype=np.float64))
        return ClusterResult(
            labels=np.asarray(labels, dtype=np.int_),
            probabilities=np.asarray(
                getattr(model, "probabilities_", np.zeros(len(labels))), dtype=np.float64
            ),
            algorithm=self.algorithm,
            params=self.params,
            embedding_mean=None if mean is None else tuple(float(v) for v in mean),
        )


def centroids(vectors: Vectors, result: ClusterResult) -> dict[int, Vectors]:
    """Mean vector per cluster, re-normalised so it lives on the same unit sphere."""
    out: dict[int, Vectors] = {}
    for cluster_id in result.cluster_ids:
        member = vectors[result.indices_for(cluster_id)]
        mean = member.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        out[cluster_id] = (mean / norm if norm else mean).astype(np.float32)
    return out


def representatives(
    vectors: Vectors, result: ClusterResult, cluster_id: int, k: int = 8
) -> list[int]:
    """Indices of the ``k`` members closest to their centroid.

    These become the exemplars an LLM sees when naming the cluster, so they should be
    the most typical members rather than an arbitrary slice.
    """
    indices = result.indices_for(cluster_id)
    if indices.size == 0:
        return []
    member = vectors[indices]
    centroid = member.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm:
        centroid = centroid / norm
    similarity = member @ centroid
    order = np.argsort(-similarity)[:k]
    return [int(indices[i]) for i in order]


def _recentre(vectors: Vectors, mean: Vectors) -> Vectors:
    """Subtract a given mean and re-normalise. The shared half of ``centre_vectors``."""
    centred = np.asarray(vectors, dtype=np.float32) - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    # A vector sitting exactly at the corpus mean has nothing left; leave it at zero
    # rather than dividing by it.
    return np.asarray(centred / np.maximum(norms, 1e-12), dtype=np.float32)


def corpus_mean(vectors: Vectors) -> Vectors:
    """The shared direction ``centre_vectors`` removes. Persist it with the taxonomy."""
    return np.asarray(vectors, dtype=np.float32).mean(axis=0)


def centre_vectors(vectors: Vectors) -> Vectors:
    """Move the corpus mean to the origin and re-normalise.

    Two lines, and they change what the clusterer sees. A sentence encoder puts every
    vector near one shared direction, so cosine similarities sit in a narrow band well
    above zero and the density differences HDBSCAN looks for are squeezed into the
    remainder. Subtracting the mean spends the offset and leaves the variation.

    Pure: the caller's array is not modified, and the cache on disk stays raw so vectors
    remain reusable under a different transform.
    """
    centred = np.asarray(vectors, dtype=np.float32) - np.asarray(vectors, dtype=np.float32).mean(
        axis=0
    )
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    # A vector sitting exactly at the corpus mean has nothing left; leave it at zero
    # rather than dividing by it.
    return np.asarray(centred / np.maximum(norms, 1e-12), dtype=np.float32)


def build_clusterer(algorithm: str = "hdbscan", **kwargs: int | float | str | None) -> Clusterer:
    if algorithm != "hdbscan":
        raise ValueError(
            f"unsupported clusterer {algorithm!r}; only 'hdbscan' is implemented "
            "(bertopic is declared in the locked stack but not yet wired -- see "
            "docs/future-scoped-work.md)"
        )
    return HdbscanClusterer(**kwargs)  # type: ignore[arg-type]
