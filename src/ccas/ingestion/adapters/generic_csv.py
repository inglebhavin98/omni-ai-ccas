"""Generic CSV adapter, driven by a column mapping.

The escape hatch for a customer's own export. Unrestricted by role, because a private
corpus carries none of the shared corpora's structural bias.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path

from pydantic import Field

from ccas.ingestion.base import RawRecord, RawTurn, SourceAdapter
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Channel, Frozen, Slug, Speaker

__all__ = ["ColumnMapping", "GenericCsvAdapter"]

_CALLER_WORDS = frozenset({"customer", "caller", "client", "user", "member", "a", "0"})


class ColumnMapping(Frozen):
    """Which columns hold what. Everything but ``text`` is optional."""

    text: str = "text"
    speaker: str | None = "speaker"
    conversation_id: str | None = "conversation_id"
    """When set, consecutive rows sharing a value become one multi-turn record."""

    start_ms: str | None = "start_ms"
    end_ms: str | None = "end_ms"
    """Absent columns are simply ignored, so these defaults cost nothing."""

    intent: str | None = None
    category: str | None = None
    channel: Channel = Channel.VOICE
    locale: str = "en-US"
    domain_hint: Slug | None = None
    delimiter: str = Field(default=",", min_length=1, max_length=1)


class GenericCsvAdapter(SourceAdapter):
    source = DatasetSource.GENERIC_CSV

    def __init__(self, mapping: ColumnMapping | None = None) -> None:
        self.mapping = mapping or ColumnMapping()

    @property
    def expected_layout(self) -> str:
        m = self.mapping
        return (
            f"a CSV with a {m.text!r} column; optional {m.speaker!r} and "
            f"{m.conversation_id!r} columns group rows into multi-turn records"
        )

    def read(self, path: Path, limit: int | None = None) -> Iterator[RawRecord]:
        emitted = 0
        mapping = self.mapping
        for file in _csv_files(path):
            with file.open(encoding="utf-8", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle, delimiter=mapping.delimiter)
                    if (row.get(mapping.text) or "").strip()
                ]
            for record in self._group(file, rows):
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _group(self, file: Path, rows: list[dict[str, str]]) -> Iterator[RawRecord]:
        mapping = self.mapping
        key_column = mapping.conversation_id
        if key_column and rows and key_column in rows[0]:
            for key, group in groupby(rows, key=lambda r: r.get(key_column) or ""):
                yield self._build(f"{file.stem}:{key}", list(group))
            return
        for index, row in enumerate(rows):
            yield self._build(f"{file.stem}:{index}", [row])

    def _build(self, record_id: str, rows: list[dict[str, str]]) -> RawRecord:
        mapping = self.mapping
        turns = tuple(
            RawTurn(
                speaker=_speaker(row.get(mapping.speaker) if mapping.speaker else None),
                text=(row.get(mapping.text) or "").strip(),
                start_ms=_int(row.get(mapping.start_ms) if mapping.start_ms else None),
                end_ms=_int(row.get(mapping.end_ms) if mapping.end_ms else None),
            )
            for row in rows
        )
        return RawRecord(
            source=self.source,
            record_id=record_id,
            channel=mapping.channel,
            turns=turns,
            locale=mapping.locale,
            domain_hint=mapping.domain_hint,
        )


def _speaker(value: str | None) -> Speaker:
    return Speaker.CALLER if (value or "").strip().lower() in _CALLER_WORDS else Speaker.HUMAN_AGENT


def _int(value: str | None) -> int | None:
    try:
        parsed = int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is None or parsed >= 0 else None


def _csv_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path] if path.is_file() else []
