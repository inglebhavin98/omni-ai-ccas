"""Utterance embedding (Module 3).

Two implementations by design: the real sentence-transformers encoder, and a
dependency-free hashing encoder so the mining pipeline is testable in CI without a
2 GB torch install. Both are deterministic and L2-normalised, so cosine similarity is
a dot product and euclidean clustering behaves like cosine clustering.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ccas.mining.preflight import check_headroom
from ccas.observability.logging import get_logger
from ccas.schemas.pii import sha256_hex

__all__ = ["DEFAULT_MODEL", "Embedder", "EmbeddingCache", "HashingEmbedder", "build_embedder"]

LOG = get_logger("mining.embedder")

#: Approximate resident set of a loaded encoder, in MB. Used only by the memory preflight,
#: so an over-estimate costs a refusal and an under-estimate costs a stall -- err high.
#: A model absent from this map is not checked; guessing a number for an unknown model
#: would be worse than not guessing.
MODEL_RESIDENT_MB: dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1400,
    "BAAI/bge-base-en-v1.5": 450,
    "BAAI/bge-small-en-v1.5": 140,
}

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

Vectors = NDArray[np.float32]

#: Reserved key inside a cache archive; never a text hash.
_MODEL_KEY = "__model__"

_TOKEN = re.compile(r"[a-z0-9']+")


def _l2_normalise(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    """Encode utterances into a dense, L2-normalised matrix."""

    name: str
    dimensions: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Vectors:
        """Return an ``(len(texts), dimensions)`` float32 matrix."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """False when a required model or dependency is missing."""


class HashingEmbedder(Embedder):
    """Hashed bag-of-words. Deterministic, no dependencies, no model download.

    Not a semantic encoder -- it captures lexical overlap only. That is enough to prove
    the clustering and labelling pipeline works, and it keeps the test suite honest
    about *which* component it is exercising. Never use it to mine a shipped taxonomy.
    """

    name = "hashing-bow"

    def __init__(self, dimensions: int = 256, ngram: int = 1) -> None:
        self.dimensions = dimensions
        self.ngram = ngram

    @property
    def available(self) -> bool:
        return True

    def encode(self, texts: Sequence[str]) -> Vectors:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            grams = tokens + [
                " ".join(tokens[i : i + self.ngram])
                for i in range(len(tokens) - self.ngram + 1)
                if self.ngram > 1
            ]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimensions
                matrix[row, index] += 1.0
        return _l2_normalise(matrix)


class SentenceTransformerEmbedder(Embedder):
    """The real encoder. Lazily imported so the base install stays light."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL, batch_size: int = 64, device: str | None = None
    ) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model: object | None = None
        self._load_error: str | None = None
        self.dimensions = 0

    def _ensure_model(self) -> object | None:
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._load_error = "sentence_transformers_not_installed"
            return None
        needed = MODEL_RESIDENT_MB.get(self.name)
        if needed is not None:
            # Before the allocation, not after: an encoder that cannot get memory does not
            # raise, it stalls in uninterruptible sleep (docs/future-scoped-work.md 9.15).
            check_headroom(needed, f"encoder {self.name}")
        try:
            model = SentenceTransformer(self.name, device=self._device)
        except Exception as exc:  # any load failure means unavailable, never silent
            self._load_error = f"model_load_failed:{type(exc).__name__}"
            return None
        self._model = model
        # Renamed in sentence-transformers 6; support both so the pin can move.
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        dimensions = getter() if getter is not None else None
        if dimensions is None:
            self._model = None
            self._load_error = "model_reports_no_dimension"
            return None
        self.dimensions = int(dimensions)
        return model

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_model()
        return self._load_error

    def encode(self, texts: Sequence[str]) -> Vectors:
        model = self._ensure_model()
        if model is None:
            raise RuntimeError(f"embedder {self.name!r} unavailable: {self._load_error}")
        raw = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(raw, dtype=np.float32)


class EmbeddingCache:
    """Content-addressed vector cache.

    Mining is re-run constantly during tuning, and re-encoding 92k utterances each time
    is the slowest thing in the pipeline. Keys are text hashes, so the cache holds no
    utterances -- only vectors (CLAUDE.md Rule 2).

    Stored as two parallel arrays rather than one entry per key: a single 2-D matrix
    compresses far better than tens of thousands of tiny arrays, and it loads in one read.
    """

    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.model_name = model_name
        self._vectors: dict[str, Vectors] = {}
        if path.is_file():
            self._load()

    def _load(self) -> None:
        """Read the cache, or decide there isn't one.

        A cache is an optimisation, so nothing here may be fatal. A truncated or
        unreadable file means a re-encode, not a dead run -- the alternative cost a
        35-minute mining run to a file that had been killed mid-save.
        """
        try:
            self._read()
        # zlib.error is listed even though zipfile currently validates the member CRC
        # and raises BadZipFile first. Relying on that is relying on an implementation
        # detail of the container, not on a guarantee.
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            OSError,
            ValueError,
            KeyError,
        ) as exc:
            LOG.warning(
                "mining.cache.unreadable",
                correlation_id="embedding-cache",
                path=self.path.name,
                bytes=self.path.stat().st_size if self.path.exists() else 0,
                reason=type(exc).__name__,
            )
            self._vectors = {}

    def _read(self) -> None:
        with np.load(self.path) as archive:
            # A cache written by a different encoder is worse than no cache: the vectors
            # would be silently incomparable. Discard rather than mix.
            stored = str(archive["model"][0]) if "model" in archive.files else ""
            if stored != self.model_name:
                return
            keys = [str(k) for k in archive["keys"]]
            matrix = np.asarray(archive["vectors"], dtype=np.float32)
            self._vectors = dict(zip(keys, matrix, strict=True))

    def encode(self, embedder: Embedder, texts: Sequence[str]) -> Vectors:
        keys = [sha256_hex(text) for text in texts]
        missing = [text for text, key in zip(texts, keys, strict=True) if key not in self._vectors]
        if missing:
            # Deduplicate before encoding: a corpus repeats the same greeting thousands
            # of times, and each one would otherwise be a forward pass.
            unique = list(dict.fromkeys(missing))
            fresh = embedder.encode(unique)
            for text, vector in zip(unique, fresh, strict=True):
                self._vectors[sha256_hex(text)] = np.asarray(vector, dtype=np.float32)
        return np.vstack([self._vectors[key] for key in keys]).astype(np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._vectors)
        matrix = (
            np.vstack([self._vectors[key] for key in keys]).astype(np.float32)
            if keys
            else np.zeros((0, 1), dtype=np.float32)
        )
        # Write beside the target and rename. `np.savez_compressed` writes in place, so
        # a kill part-way through leaves a partial archive where a good cache used to be;
        # `Path.replace` is atomic, so an interrupted save loses the new cache and keeps the
        # old one instead of destroying both.
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            np.savez_compressed(
                tmp,
                model=np.array([self.model_name]),
                keys=np.array(keys),
                vectors=matrix,
            )
            # numpy appends .npz when the name lacks it.
            written = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
            written.replace(self.path)
        finally:
            for leftover in (tmp, tmp.with_name(tmp.name + ".npz")):
                leftover.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._vectors)


def build_embedder(model_name: str = DEFAULT_MODEL, *, allow_fallback: bool = False) -> Embedder:
    """Real encoder, or the hashing stand-in only when explicitly allowed.

    The fallback is opt-in because a taxonomy silently mined from lexical overlap would
    look plausible and be worthless.
    """
    embedder = SentenceTransformerEmbedder(model_name)
    if embedder.available:
        return embedder
    if not allow_fallback:
        raise RuntimeError(
            f"embedder {model_name!r} unavailable ({embedder.load_error}); "
            "install with `uv sync --extra mining`, or pass --allow-hashing-embedder "
            "to mine a throwaway taxonomy from lexical overlap"
        )
    return HashingEmbedder()
