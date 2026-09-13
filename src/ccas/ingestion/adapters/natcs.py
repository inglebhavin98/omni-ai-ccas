"""NatCS spoken dialogues -- the multi-turn benchmark corpus.

Real turn-taking with timing, which is what makes it the only corpus able to benchmark
context retention and state-graph transitions. Too small and too narrow to mine. See
ADR-0004.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ccas.ingestion.base import RawRecord, RawTurn, SourceAdapter
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Channel, Speaker

__all__ = ["NatcsAdapter"]

_TURN_KEYS = ("turns", "utterances", "dialogue", "segments")
_TEXT_KEYS = ("text", "transcript", "utterance", "content")
_SPEAKER_KEYS = ("speaker", "role", "channel", "party")
_START_KEYS = ("start_ms", "start", "begin", "start_time")
_END_KEYS = ("end_ms", "end", "stop", "end_time")

_CALLER_WORDS = frozenset({"customer", "caller", "client", "user", "a", "0", "ch0"})


def _to_ms(key: str, value: Any) -> int | None:
    """Timings appear as seconds (float) or milliseconds (int) depending on the file.

    An explicit ``_ms`` suffix is authoritative; guessing from magnitude would turn a
    legitimate ``start_ms: 100`` into 100 seconds.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    if key.endswith("_ms"):
        return int(number)
    # Otherwise guess: a dialogue turn is never 1,000+ seconds long, so a large value
    # is already milliseconds.
    return int(number) if number > 1000 else round(number * 1000)


class NatcsAdapter(SourceAdapter):
    source = DatasetSource.NATCS

    @property
    def expected_layout(self) -> str:
        return (
            "one JSON file per dialogue (or a directory of them, or JSONL). Each carries "
            f"turns under one of {_TURN_KEYS}, with a text field ({_TEXT_KEYS}), a "
            f"speaker ({_SPEAKER_KEYS}) and optional timings ({_START_KEYS}/{_END_KEYS})"
        )

    def read(self, path: Path, limit: int | None = None) -> Iterator[RawRecord]:
        emitted = 0
        for file in _dialogue_files(path):
            for index, payload in enumerate(_iter_dialogues(file)):
                record = self._to_record(file, index, payload)
                if record is None:
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _to_record(self, file: Path, index: int, payload: dict[str, Any]) -> RawRecord | None:
        raw_turns = next(
            (payload[key] for key in _TURN_KEYS if isinstance(payload.get(key), list)), None
        )
        if not raw_turns:
            return None

        turns: list[RawTurn] = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            text = next(
                (str(item[k]).strip() for k in _TEXT_KEYS if str(item.get(k) or "").strip()), ""
            )
            if not text:
                continue
            label = next((str(item[k]) for k in _SPEAKER_KEYS if item.get(k) is not None), "")
            start = next((_to_ms(k, item[k]) for k in _START_KEYS if k in item), None)
            end = next((_to_ms(k, item[k]) for k in _END_KEYS if k in item), None)
            if start is not None and end is not None and end < start:
                start, end = end, start
            turns.append(
                RawTurn(
                    speaker=(
                        Speaker.CALLER
                        if label.strip().lower() in _CALLER_WORDS
                        else Speaker.HUMAN_AGENT
                    ),
                    text=text,
                    start_ms=start,
                    end_ms=end,
                )
            )
        if not turns:
            return None

        last_end = next((t.end_ms for t in reversed(turns) if t.end_ms is not None), None)
        record_id = str(payload.get("id") or payload.get("dialogue_id") or f"{file.stem}:{index}")
        return RawRecord(
            source=self.source,
            record_id=record_id,
            channel=Channel.VOICE,
            turns=tuple(turns),
            duration_ms=last_end,
            metadata={"file": file.name},
        )


def _dialogue_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted([*path.glob("*.json"), *path.glob("*.jsonl")])
    return [path] if path.is_file() else []


def _iter_dialogues(file: Path) -> Iterator[dict[str, Any]]:
    text = file.read_text(encoding="utf-8").strip()
    if not text:
        return
    if file.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))
    elif isinstance(payload, dict):
        yield payload
