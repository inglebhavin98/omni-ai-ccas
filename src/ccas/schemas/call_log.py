"""``CallLog`` -- the unified normalized record produced by Module 1.

Every source adapter (AIxBlock, NatCS, Bitext, generic CSV, synthetic) converges on
this shape, so Module 3 never learns which dataset a record came from.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ccas.schemas.common import (
    Channel,
    Frozen,
    JsonValue,
    SchemaVersion,
    Slug,
    Speaker,
    utcnow,
)
from ccas.schemas.pii import RedactedText, RedactionReport, sha256_hex

__all__ = ["CallLog", "DatasetSource", "DtmfEvent", "GoldLabels", "Utterance"]


class DatasetSource(StrEnum):
    AIXBLOCK = "aixblock"
    NATCS = "natcs"
    BITEXT = "bitext"
    SYNTHETIC = "synthetic"
    GENERIC_CSV = "generic_csv"
    LIVE_CAPTURE = "live_capture"


class DtmfEvent(Frozen):
    """Touch-tone input. Digit runs are masked by default -- a 16-digit run is a card."""

    digits: str = Field(pattern=r"^[0-9*#A-D]+$", min_length=1, max_length=64)
    at_ms: int = Field(ge=0)
    redacted: bool = True

    @model_validator(mode="after")
    def _mask_long_runs(self) -> Self:
        if not self.redacted and len(self.digits) >= 4:
            raise ValueError(
                f"DTMF run of {len(self.digits)} digits must be redacted before storage"
            )
        return self


class Utterance(Frozen):
    index: int = Field(ge=0)
    speaker: Speaker
    content: RedactedText
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    asr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _check_timing(self) -> Self:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) precedes start_ms ({self.start_ms})")
        return self

    @property
    def duration_ms(self) -> int | None:
        if self.start_ms is None or self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


class GoldLabels(Frozen):
    """Supervision, present only where the source dataset ships it (e.g. Bitext).

    Values are slugs so a pack's taxonomy can be validated against them without the
    core knowing what any particular label means.
    """

    intent: Slug | None = None
    category: Slug | None = None
    slots: dict[Slug, str] = Field(default_factory=dict)
    resolution: str | None = Field(default=None, max_length=64)


class CallLog(Frozen):
    schema_version: SchemaVersion = "1.0"
    call_id: str = Field(min_length=8, max_length=64)
    source: DatasetSource
    source_record_id: str = Field(min_length=1, max_length=256)
    channel: Channel
    domain_hint: Slug | None = None
    """Free-form pack slug. Never an enum -- a new vertical must not require a code change."""

    locale: str = Field(default="en-US", max_length=32)
    utterances: tuple[Utterance, ...] = Field(min_length=1)
    dtmf_events: tuple[DtmfEvent, ...] = ()
    duration_ms: int | None = Field(default=None, ge=0)
    labels: GoldLabels | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    redaction: RedactionReport
    ingested_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _enforce_clean(self) -> Self:
        """A ``CallLog`` that is not egress-clean must not be constructible (Rule 2)."""
        if not self.redaction.egress_permitted:
            raise ValueError(
                f"CallLog {self.call_id}: redaction status is "
                f"{self.redaction.status.value}, refusing construction"
            )
        for utterance in self.utterances:
            if not utterance.content.egress_permitted:
                raise ValueError(
                    f"CallLog {self.call_id}: utterance {utterance.index} carries "
                    f"redaction status {utterance.content.report.status.value}"
                )
        return self

    @model_validator(mode="after")
    def _check_utterance_ordering(self) -> Self:
        expected = list(range(len(self.utterances)))
        actual = [u.index for u in self.utterances]
        if actual != expected:
            raise ValueError(f"utterance indices must be contiguous from 0, got {actual}")
        return self

    @staticmethod
    def make_call_id(source: DatasetSource, source_record_id: str) -> str:
        """Deterministic id so re-ingesting the same corpus is idempotent."""
        return sha256_hex(f"{source.value}|{source_record_id}")[:32]

    @property
    def caller_utterances(self) -> tuple[Utterance, ...]:
        return tuple(u for u in self.utterances if u.speaker is Speaker.CALLER)

    @property
    def turn_count(self) -> int:
        return len(self.utterances)
