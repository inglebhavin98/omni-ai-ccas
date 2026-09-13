from __future__ import annotations

import numpy as np
import pytest

from ccas.mining.cluster import (
    DEFAULT_MIN_SAMPLES,
    NOISE_LABEL,
    ClusterResult,
    HdbscanClusterer,
    build_clusterer,
    centre_vectors,
    centroids,
    representatives,
)
from ccas.mining.embedder import HashingEmbedder

TOPIC_A = ["I want to check the status of my request"] * 14
TOPIC_B = ["there is a charge I do not recognise on my statement"] * 14
TOPIC_C = ["I need to change the details on my account"] * 14
OUTLIER = ["an entirely unrelated sentence about migratory birds"]


@pytest.fixture(scope="module")
def corpus() -> tuple[list[str], np.ndarray]:
    texts = TOPIC_A + TOPIC_B + TOPIC_C + OUTLIER
    return texts, HashingEmbedder().encode(texts)


def test_separable_topics_are_recovered(corpus: tuple[list[str], np.ndarray]) -> None:
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    assert result.n_clusters == 3


def test_an_outlier_is_left_as_noise(corpus: tuple[list[str], np.ndarray]) -> None:
    """Refusing to assign is the property k-means cannot give us.

    ``centre=False`` because this fixture's outlier is defined in the raw space. Centring
    a handful of toy vectors subtracts a "corpus mean" that means nothing at this size and
    absorbs the outlier -- a fixture artifact, not a finding about the transform.
    """
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5, centre=False).fit_predict(vectors)
    assert result.noise_count == 1
    assert result.labels[-1] == NOISE_LABEL


def test_coverage_and_noise_are_complementary(corpus: tuple[list[str], np.ndarray]) -> None:
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    assert result.coverage + result.noise_ratio == pytest.approx(1.0)


def test_sizes_sum_to_the_assigned_points(corpus: tuple[list[str], np.ndarray]) -> None:
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    assert sum(result.sizes().values()) == result.n_points - result.noise_count


def test_too_few_points_yields_all_noise_rather_than_a_forced_cluster() -> None:
    """Shrinking the parameter until something appears would manufacture an intent."""
    vectors = HashingEmbedder().encode(["a", "b", "c"])
    result = HdbscanClusterer(min_cluster_size=15).fit_predict(vectors)
    assert result.n_clusters == 0
    assert result.noise_ratio == 1.0


def test_centroids_stay_on_the_unit_sphere(corpus: tuple[list[str], np.ndarray]) -> None:
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    for vector in centroids(vectors, result).values():
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_representatives_are_the_most_typical_members(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    texts, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    for cluster_id in result.cluster_ids:
        picked = representatives(vectors, result, cluster_id, k=3)
        assert len(picked) == 3
        assert len({texts[i] for i in picked}) == 1, "a cluster should be coherent"


def test_representatives_are_capped_by_cluster_size(
    corpus: tuple[list[str], np.ndarray],
) -> None:
    _, vectors = corpus
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    cluster_id = result.cluster_ids[0]
    assert len(representatives(vectors, result, cluster_id, k=1000)) == len(
        result.indices_for(cluster_id)
    )


def test_params_are_recorded_for_provenance() -> None:
    params = HdbscanClusterer(min_cluster_size=9, min_samples=3).params
    assert params["min_cluster_size"] == 9
    assert params["min_samples"] == 3


def test_a_degenerate_min_cluster_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        HdbscanClusterer(min_cluster_size=1)


def test_an_unimplemented_algorithm_says_so() -> None:
    with pytest.raises(ValueError, match="only 'hdbscan' is implemented"):
        build_clusterer("kmeans")


def test_result_helpers_agree_on_an_empty_input() -> None:
    empty = ClusterResult(
        labels=np.array([], dtype=np.int_),
        probabilities=np.array([], dtype=np.float64),
        algorithm="hdbscan",
    )
    assert empty.n_points == 0
    assert empty.n_clusters == 0
    assert empty.noise_ratio == 0.0


def test_min_samples_is_set_explicitly_not_inherited() -> None:
    """`None` means "use min_cluster_size", which is HDBSCAN's most conservative setting.

    Measured on 6,000 real utterances at min_cluster_size=25: None → 8.4% coverage,
    10 → 11.3%, 5 → 14.0%, 3 → 15.8%. Inheriting the default is a choice that costs
    roughly half the achievable coverage, so it is made deliberately here.
    """
    assert HdbscanClusterer().min_samples == DEFAULT_MIN_SAMPLES
    assert DEFAULT_MIN_SAMPLES is not None
    # The library's behaviour stays reachable for anyone who wants it.
    assert HdbscanClusterer(min_samples=None).min_samples is None


def test_min_samples_is_recorded_in_the_params() -> None:
    """It lands in `IntentTaxonomy.clusterer_params`, so a taxonomy is reproducible."""
    assert HdbscanClusterer(min_cluster_size=25, min_samples=3).params["min_samples"] == 3


def test_mean_centring_is_on_by_default() -> None:
    """bge-large is strongly anisotropic and the offset compresses every distance.

    Measured on 6,000 real utterances: ||mean vector|| / mean ||vector|| = 0.716, i.e.
    72% of the geometry is one shared direction. Removing it moved coverage 13.6% -> 18.1%
    at min_cluster_size=25, min_samples=5. See ADR-0017.
    """
    assert HdbscanClusterer().centre is True
    assert HdbscanClusterer(centre=False).centre is False


def test_centring_is_recorded_in_the_params() -> None:
    """A taxonomy is only reproducible if the space it was clustered in is recorded."""
    assert HdbscanClusterer().params["centre"] is True


def test_centring_removes_the_shared_direction() -> None:
    """The transform itself: the shared direction goes, rows stay unit norm.

    Re-normalising after subtracting the mean reintroduces a small offset, so the test is
    that the shared direction collapses by orders of magnitude -- not that the mean lands
    exactly on zero, which renormalisation makes impossible.
    """
    rng = np.random.default_rng(0)
    common = rng.normal(size=32)
    common /= np.linalg.norm(common)
    raw = rng.normal(scale=0.3, size=(200, 32)) + 3.0 * common
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    centred = centre_vectors(raw.astype(np.float32))

    before = float(np.linalg.norm(raw.mean(axis=0)))
    after = float(np.linalg.norm(centred.mean(axis=0)))
    # 0.716 is what bge-large measures on the real corpus; this fixture is worse.
    assert before > 0.75, "fixture should be strongly anisotropic"
    assert after < before / 20
    assert np.allclose(np.linalg.norm(centred, axis=1), 1.0, atol=1e-5)


def test_centring_leaves_an_already_centred_space_alone() -> None:
    """No shared direction to remove means nothing meaningful to change."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(200, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centred = centre_vectors(vectors)
    assert np.linalg.norm(centred.mean(axis=0)) < np.linalg.norm(vectors.mean(axis=0)) + 1e-6


def test_centring_never_divides_by_zero() -> None:
    """A vector sitting exactly at the corpus mean becomes the zero vector."""
    vectors = np.ones((10, 8), dtype=np.float32)
    centred = centre_vectors(vectors)
    assert np.isfinite(centred).all()


def test_the_result_carries_the_space_it_clustered_in() -> None:
    """Centring inside fit_predict and discarding the mean makes the taxonomy incoherent.

    Membership is decided in centred space; if the stored centroid is computed from raw
    vectors, the two describe different spaces and the mean that maps between them is
    gone. A live utterance cannot be centred at call time either -- the mean of one
    vector is itself, so centring it gives zero. The transform is only reconstructible
    from a stored mean.
    """
    rng = np.random.default_rng(0)
    common = rng.normal(size=16)
    common /= np.linalg.norm(common)
    vectors = (rng.normal(scale=0.3, size=(120, 16)) + 3.0 * common).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    result = HdbscanClusterer(min_cluster_size=5).fit_predict(vectors)
    assert result.embedding_mean is not None
    assert len(result.embedding_mean) == vectors.shape[1]

    # The recorded mean reproduces the space exactly.
    replayed = result.in_cluster_space(vectors)
    assert np.allclose(replayed, centre_vectors(vectors), atol=1e-6)


def test_an_uncentred_result_records_no_mean() -> None:
    """`centre=False` clusters raw, so there is no transform to record or undo."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(80, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    result = HdbscanClusterer(min_cluster_size=5, centre=False).fit_predict(vectors)
    assert result.embedding_mean is None
    assert np.allclose(result.in_cluster_space(vectors), vectors)


def test_centroids_are_computed_in_cluster_space() -> None:
    """Under anisotropy, raw centroids are all mutually similar and barely separate.

    Averaging within a cluster concentrates the shared direction -- the orthogonal parts
    cancel and the common part does not -- so raw centroids separate worse than the
    members did. That is the compression centring removed, reintroduced at inference.
    """
    rng = np.random.default_rng(2)
    common = rng.normal(size=24)
    common /= np.linalg.norm(common)
    blobs = rng.normal(size=(4, 24)) * 0.8
    labels = rng.integers(0, 4, 400)
    vectors = (blobs[labels] + rng.normal(scale=0.2, size=(400, 24)) + 3.0 * common).astype(
        np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    result = HdbscanClusterer(min_cluster_size=10).fit_predict(vectors)
    assert result.n_clusters >= 2

    raw = np.vstack(list(centroids(vectors, result).values()))
    cluster_space = np.vstack(list(centroids(result.in_cluster_space(vectors), result).values()))

    def off_diagonal_mean(m: np.ndarray) -> float:
        gram = m @ m.T
        n = len(gram)
        return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))

    assert off_diagonal_mean(raw) > 0.5, "raw centroids should be crowded by the offset"
    assert off_diagonal_mean(cluster_space) < off_diagonal_mean(raw) - 0.3


def test_one_vector_maps_into_cluster_space_without_collapsing() -> None:
    """The live-router shape: a single utterance, arriving alone.

    This is why the mean is stored rather than recomputed. Centring one vector against
    its own mean gives the zero vector -- it would route every caller to whichever
    centroid happens to win a tie between zeros.
    """
    rng = np.random.default_rng(3)
    common = rng.normal(size=16)
    common /= np.linalg.norm(common)
    corpus = (rng.normal(scale=0.3, size=(200, 16)) + 3.0 * common).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    result = HdbscanClusterer(min_cluster_size=5).fit_predict(corpus)

    live = corpus[:1]
    mapped = result.in_cluster_space(live)

    assert mapped.shape == (1, 16)
    assert np.linalg.norm(mapped) > 0.5, "a lone vector must not collapse to zero"
    # Identical to how the same vector was placed when the corpus was clustered.
    assert np.allclose(mapped[0], result.in_cluster_space(corpus)[0], atol=1e-6)
    # What recomputing a mean over one row would have given.
    assert np.linalg.norm(centre_vectors(live)) < 1e-6


def test_cluster_space_widens_the_routing_margin() -> None:
    """The consequence: how far apart the top two intents are for a live utterance.

    A margin inside ASR-variation noise makes a router flip intents on rephrasing, and
    that presents as "the classifier is unstable" rather than as a space mismatch.
    """
    rng = np.random.default_rng(4)
    common = rng.normal(size=24)
    common /= np.linalg.norm(common)
    blobs = rng.normal(size=(4, 24)) * 0.8
    labels = rng.integers(0, 4, 500)
    corpus = (blobs[labels] + rng.normal(scale=0.2, size=(500, 24)) + 3.0 * common).astype(
        np.float32
    )
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    held_out, fitted = corpus[:60], corpus[60:]
    result = HdbscanClusterer(min_cluster_size=10).fit_predict(fitted)
    assert result.n_clusters >= 3

    def median_margin(queries: np.ndarray, centres: np.ndarray) -> float:
        scores = np.sort(queries @ centres.T, axis=1)
        return float(np.median(scores[:, -1] - scores[:, -2]))

    raw = np.vstack(list(centroids(fitted, result).values()))
    space = np.vstack(list(centroids(result.in_cluster_space(fitted), result).values()))

    # One at a time, which is the shape that cannot recompute a mean.
    mapped = np.vstack([result.in_cluster_space(held_out[i : i + 1]) for i in range(len(held_out))])

    assert median_margin(mapped, space) > median_margin(held_out, raw) * 1.5
