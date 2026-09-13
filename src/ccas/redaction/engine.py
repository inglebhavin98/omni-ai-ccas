"""Redaction engine interface.

An engine only *detects*. Substitution, placeholder allocation, leak checking and report
construction all live in the pipeline, so adding an engine cannot change the guarantees
(CLAUDE.md Rule 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, Slug
from ccas.schemas.pii import EngineName, PiiEntityType

__all__ = ["DetectedSpan", "RedactionEngine", "merge_spans"]


class DetectedSpan(Frozen):
    """A candidate entity, pre-substitution.

    Deliberately carries offsets rather than matched text: nothing downstream of an
    engine needs the original, so nothing downstream is given a copy of it.
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    entity_type: PiiEntityType
    score: float = Field(ge=0.0, le=1.0)
    engine: EngineName
    custom_label: Slug | None = None
    pattern_name: Slug | None = None
    """Which rule fired. Safe to log; the text it matched is not."""

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must exceed start ({self.start})")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: DetectedSpan) -> bool:
        return self.start < other.end and other.start < self.end


class RedactionEngine(ABC):
    """Detect PII in a string. Implementations must be side-effect free and reentrant."""

    name: EngineName

    @property
    @abstractmethod
    def available(self) -> bool:
        """False when a required model or dependency is missing.

        An unavailable engine makes the pipeline report ``UNVERIFIED``, which blocks
        egress. It never degrades silently to "found nothing".
        """

    @abstractmethod
    def detect(self, text: str) -> tuple[DetectedSpan, ...]:
        """Candidate spans, in any order. The pipeline resolves overlaps."""

    @property
    @abstractmethod
    def handled_entities(self) -> frozenset[PiiEntityType]:
        """What this engine can find. Used to skip engines that add nothing."""


def merge_spans(spans: tuple[DetectedSpan, ...]) -> tuple[DetectedSpan, ...]:
    """Resolve overlaps, keeping the strongest detection.

    Ordering is by score, then by length: a longer match usually means a more specific
    pattern fired (a full card number beats the digit run inside it), and a specific
    pattern is more trustworthy than a broad one at the same confidence.
    """
    ranked = sorted(spans, key=lambda s: (-s.score, -s.length, s.start))
    kept: list[DetectedSpan] = []
    for span in ranked:
        if not any(span.overlaps(existing) for existing in kept):
            kept.append(span)
    return tuple(sorted(kept, key=lambda s: s.start))
