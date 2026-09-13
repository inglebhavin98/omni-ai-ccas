"""Bitext customer-support intent dataset -- the supervision corpus.

Single-turn, labelled, already sanitised. It grades the router and the judge; it must
never be mined (``require_role`` enforces that). See ADR-0004.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ccas.ingestion.base import RawRecord, RawTurn, SourceAdapter
from ccas.schemas.call_log import DatasetSource, GoldLabels
from ccas.schemas.common import Channel, Speaker

__all__ = ["BitextAdapter"]

_UTTERANCE_COLUMNS = ("instruction", "utterance", "query", "text")
_RESPONSE_COLUMNS = ("response", "answer", "reply")


def _slugify(value: str) -> str | None:
    """Bitext labels are SCREAMING_SNAKE; our intent ids are dotted lowercase slugs."""
    cleaned = value.strip().lower().replace(" ", "_")
    kept = "".join(c if (c.isalnum() or c in "._-") else "_" for c in cleaned)
    kept = "_".join(part for part in kept.split("_") if part)
    return kept or None


class BitextAdapter(SourceAdapter):
    source = DatasetSource.BITEXT

    @property
    def expected_layout(self) -> str:
        return (
            "a CSV (or directory of CSVs) with an utterance column "
            f"({' | '.join(_UTTERANCE_COLUMNS)}), plus optional 'category', 'intent' "
            f"and a response column ({' | '.join(_RESPONSE_COLUMNS)})"
        )

    def read(self, path: Path, limit: int | None = None) -> Iterator[RawRecord]:
        emitted = 0
        for file in _csv_files(path):
            with file.open(encoding="utf-8", newline="") as handle:
                for row_index, row in enumerate(csv.DictReader(handle)):
                    record = self._to_record(file, row_index, row)
                    if record is None:
                        continue
                    yield record
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return

    def _to_record(
        self, file: Path, row_index: int, row: dict[str, str | None]
    ) -> RawRecord | None:
        utterance = _first_present(row, _UTTERANCE_COLUMNS)
        if not utterance:
            return None

        turns = [RawTurn(speaker=Speaker.CALLER, text=utterance)]
        response = _first_present(row, _RESPONSE_COLUMNS)
        if response:
            turns.append(RawTurn(speaker=Speaker.HUMAN_AGENT, text=response))

        intent = (row.get("intent") or "").strip()
        category = (row.get("category") or "").strip()
        labels = GoldLabels(
            intent=_slugify(intent) if intent else None,
            category=_slugify(category) if category else None,
        )

        return RawRecord(
            source=self.source,
            record_id=f"{file.stem}:{row_index}",
            channel=Channel.CHAT,
            turns=tuple(turns),
            labels=labels if (labels.intent or labels.category) else None,
            metadata={"flags": (row.get("flags") or "").strip()},
        )


def _first_present(row: dict[str, str | None], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return None


def _csv_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path] if path.is_file() else []
