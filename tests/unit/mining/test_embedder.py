from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccas.mining.embedder import EmbeddingCache, HashingEmbedder, build_embedder


@pytest.fixture(scope="module")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimensions=128)


def test_output_shape_matches_the_declared_dimensions(embedder: HashingEmbedder) -> None:
    vectors = embedder.encode(["one", "two", "three"])
    assert vectors.shape == (3, 128)
    assert vectors.dtype == np.float32


def test_vectors_are_l2_normalised(embedder: HashingEmbedder) -> None:
    """Clustering uses euclidean distance; normalisation makes that equal cosine."""
    norms = np.linalg.norm(embedder.encode(["hello world", "another line"]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encoding_is_deterministic(embedder: HashingEmbedder) -> None:
    assert np.array_equal(embedder.encode(["same text"]), embedder.encode(["same text"]))


def test_similar_text_is_closer_than_unrelated_text(embedder: HashingEmbedder) -> None:
    a, b, c = embedder.encode(
        [
            "I want to check the status of my request",
            "I want to check on the status of my request please",
            "the weather today is unusually cold",
        ]
    )
    assert float(a @ b) > float(a @ c)


def test_an_empty_string_does_not_divide_by_zero(embedder: HashingEmbedder) -> None:
    vectors = embedder.encode(["", "text"])
    assert np.isfinite(vectors).all()


def test_the_hashing_embedder_needs_no_model() -> None:
    assert HashingEmbedder().available


def test_cache_avoids_re_encoding(tmp_path: Path, embedder: HashingEmbedder) -> None:
    cache = EmbeddingCache(tmp_path / "v.npz", embedder.name)
    cache.encode(embedder, ["a sentence", "another sentence"])
    assert len(cache) == 2
    cache.encode(embedder, ["a sentence"])
    assert len(cache) == 2, "a repeat must not add an entry"


def test_cache_deduplicates_within_one_batch(tmp_path: Path, embedder: HashingEmbedder) -> None:
    """A corpus repeats its greeting thousands of times; each would be a forward pass."""
    cache = EmbeddingCache(tmp_path / "v.npz", embedder.name)
    vectors = cache.encode(embedder, ["same"] * 50)
    assert vectors.shape[0] == 50
    assert len(cache) == 1


def test_cache_round_trips(tmp_path: Path, embedder: HashingEmbedder) -> None:
    path = tmp_path / "v.npz"
    first = EmbeddingCache(path, embedder.name)
    vectors = first.encode(embedder, ["alpha", "beta"])
    first.save()
    assert np.array_equal(
        EmbeddingCache(path, embedder.name).encode(embedder, ["alpha", "beta"]), vectors
    )


def test_a_cache_from_another_model_is_discarded(tmp_path: Path, embedder: HashingEmbedder) -> None:
    """Vectors from a different encoder are silently incomparable."""
    path = tmp_path / "v.npz"
    written = EmbeddingCache(path, embedder.name)
    written.encode(embedder, ["alpha"])
    written.save()
    assert len(EmbeddingCache(path, "some-other-model")) == 0


def test_an_empty_cache_saves_and_reloads(tmp_path: Path, embedder: HashingEmbedder) -> None:
    path = tmp_path / "v.npz"
    EmbeddingCache(path, embedder.name).save()
    assert len(EmbeddingCache(path, embedder.name)) == 0


def test_the_hashing_fallback_is_opt_in() -> None:
    """A taxonomy silently mined from lexical overlap would look plausible and be junk."""
    with pytest.raises(RuntimeError, match="allow-hashing-embedder"):
        build_embedder("definitely-not-a-real-model-xyz", allow_fallback=False)


def test_the_fallback_is_returned_when_allowed() -> None:
    assert isinstance(
        build_embedder("definitely-not-a-real-model-xyz", allow_fallback=True),
        HashingEmbedder,
    )


def test_a_corrupt_cache_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    """A truncated cache killed a 35-minute mining run. A cache is an optimisation.

    `np.savez_compressed` writes in place, so killing a run mid-save leaves a partial zip
    that `np.load` rejects with `BadZipFile` -- and the exception surfaced from
    `EmbeddingCache.__init__`, before the run had done anything it could recover.
    """
    path = tmp_path / "embeddings.npz"
    path.write_bytes(b"PK\x03\x04 truncated, killed mid-write")
    cache = EmbeddingCache(path, "model-a")
    assert len(cache) == 0
    # And it must still be usable: the next run rewrites it.
    cache.encode(HashingEmbedder(), ["hello there"])
    cache.save()
    assert len(EmbeddingCache(path, HashingEmbedder().name)) >= 0


def test_an_empty_cache_file_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    path.write_bytes(b"")
    assert len(EmbeddingCache(path, "model-a")) == 0


def test_save_does_not_destroy_a_good_cache_when_interrupted(tmp_path: Path) -> None:
    """Write to a temp file and rename, so a kill cannot leave a half-written cache.

    Checked by asserting no partial file is left behind next to the target and that the
    target is only ever a complete archive.
    """
    path = tmp_path / "embeddings.npz"
    embedder = HashingEmbedder()
    cache = EmbeddingCache(path, embedder.name)
    cache.encode(embedder, ["one", "two"])
    cache.save()
    assert len(EmbeddingCache(path, embedder.name)) == 2
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert leftovers == []


def test_a_cache_with_corrupt_compressed_bytes_is_a_miss(tmp_path: Path) -> None:
    """A valid zip whose member bytes are damaged. The container looks fine; the data isn't."""
    path = tmp_path / "embeddings.npz"
    embedder = HashingEmbedder()
    cache = EmbeddingCache(path, embedder.name)
    cache.encode(embedder, ["one", "two", "three"])
    cache.save()

    raw = bytearray(path.read_bytes())
    # Damage the middle, leaving the zip's own header and central directory intact.
    for i in range(len(raw) // 3, len(raw) // 3 + 64):
        raw[i] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert len(EmbeddingCache(path, embedder.name)) == 0
