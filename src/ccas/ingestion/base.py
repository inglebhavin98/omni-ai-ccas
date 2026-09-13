"""Source adapter interface.

``RawRecord`` is the one place in the codebase where unredacted text legitimately
exists. It is in-process, short-lived, and consumed by the normalizer, which redacts it
before anything else can see it. It has no path to a log, a prompt, or a file -- and its
``__repr__`` is deliberately blind so it cannot leak through a traceback or a debugger
watch (CLAUDE.md Rule 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from ccas.ingestion.datasets import DatasetRole, roles_for
from ccas.schemas.call_log import DatasetSource, GoldLabels
from ccas.schemas.common import Channel, Frozen, JsonValue, Slug, Speaker

__all__ = ["RawDtmf", "RawRecord", "RawTurn", "SourceAdapter"]


class RawTurn(Frozen):
    """One unredacted turn. Never serialized, never logged."""

    speaker: Speaker
    text: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = None

    def __repr__(self) -> str:
        return f"RawTurn(speaker={self.speaker.value}, chars={len(self.text)})"

    __str__ = __repr__


class RawDtmf(Frozen):
    digits: str = Field(min_length=1, max_length=64)
    at_ms: int = Field(default=0, ge=0)

    def __repr__(self) -> str:
        return f"RawDtmf(digits={len(self.digits)}, at_ms={self.at_ms})"

    __str__ = __repr__


class RawRecord(Frozen):
    """A single conversation as the source dataset expresses it."""

    source: DatasetSource
    record_id: str = Field(min_length=1, max_length=256)
    channel: Channel = Channel.VOICE
    turns: tuple[RawTurn, ...] = Field(min_length=1)
    dtmf: tuple[RawDtmf, ...] = ()
    locale: str = "en-US"
    domain_hint: Slug | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    labels: GoldLabels | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Adapter passthrough. Must contain no free text -- it is copied onto the CallLog,
    which the redactor does not inspect."""

    @model_validator(mode="after")
    def _check_metadata_is_not_a_side_channel(self) -> Self:
        for key, value in self.metadata.items():
            if isinstance(value, str) and len(value) > 128:
                raise ValueError(
                    f"metadata[{key!r}] is {len(value)} chars; free text must go in a "
                    "turn where the redactor can see it, not in metadata"
                )
        return self

    def __repr__(self) -> str:
        return (
            f"RawRecord(source={self.source.value}, record_id={self.record_id!r}, "
            f"turns={len(self.turns)})"
        )

    __str__ = __repr__

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(turn.text for turn in self.turns)


class SourceAdapter(ABC):
    """Reads one corpus format and yields ``RawRecord``.

    An adapter does no redaction and makes no quality judgements; it translates a file
    format. Everything else happens once, in the normalizer.
    """

    source: DatasetSource

    @property
    def roles(self) -> frozenset[DatasetRole]:
        """What this corpus may be used for. See ``ccas.ingestion.datasets``."""
        return roles_for(self.source)

    @property
    @abstractmethod
    def expected_layout(self) -> str:
        """Human-readable description of the files this adapter expects.

        Surfaced in CLI errors, because "no records found" is a much worse message than
        "expected a CSV with an ``instruction`` column".
        """

    @abstractmethod
    def read(self, path: Path, limit: int | None = None) -> Iterator[RawRecord]:
        """Yield records from ``path`` (a file or a directory of files)."""

    def supports(self, role: DatasetRole) -> bool:
        return role in self.roles
