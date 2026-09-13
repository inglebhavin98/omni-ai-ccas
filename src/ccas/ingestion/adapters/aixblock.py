"""AIxBlock call-centre scripts -- the mining corpus.

Unlabelled, high volume, multi-turn. This is the only corpus a taxonomy is mined from,
and the one redaction throughput is measured against. See ADR-0004.

The published files are JSONL or JSON-array; each record holds either a list of
speaker-tagged turns or a single script string with speaker prefixes. Both shapes are
handled because the corpus is not internally consistent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ccas.ingestion.base import RawRecord, RawTurn, SourceAdapter
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Channel, Speaker

__all__ = ["AixBlockAdapter"]

_TURN_KEYS = ("turns", "messages", "dialogue", "conversation", "utterances")
_SCRIPT_KEYS = ("script", "transcript", "text", "content")
_TEXT_KEYS = ("text", "utterance", "content", "message", "value")
_SPEAKER_KEYS = ("speaker", "role", "from", "agent")

#: "Agent: ...", "CUSTOMER - ...", "[Caller] ..." all appear in the wild. A bracketed
#: speaker needs no separator; a bare one does, or every sentence starting with "user"
#: would be read as a speaker tag.
_SPEAKER_ALTERNATION = (
    "agent|advisor|rep|representative|operator|bot|ivr|system|customer|caller|client|user|member"
)
_PREFIX = re.compile(
    rf"^\s*(?:[\[(]\s*({_SPEAKER_ALTERNATION})\s*[\])]\s*[:\-\u2013]?\s*"
    rf"|({_SPEAKER_ALTERNATION})\s*[:\-\u2013]\s*)",
    re.IGNORECASE,
)

_CALLER_WORDS = frozenset({"customer", "caller", "client", "user", "member"})
_SYSTEM_WORDS = frozenset({"system", "ivr", "bot"})


def _speaker_from(label: str | None) -> Speaker:
    word = (label or "").strip().lower()
    if word in _CALLER_WORDS:
        return Speaker.CALLER
    if word in _SYSTEM_WORDS:
        return Speaker.IVR if word == "ivr" else Speaker.BOT
    return Speaker.HUMAN_AGENT


class AixBlockAdapter(SourceAdapter):
    source = DatasetSource.AIXBLOCK

    @property
    def expected_layout(self) -> str:
        return (
            "JSONL or a JSON array (.json/.jsonl, or a directory of them). Each record "
            f"carries turns under one of {_TURN_KEYS}, or a single script string under "
            f"one of {_SCRIPT_KEYS} with 'Speaker: text' lines"
        )

    def read(self, path: Path, limit: int | None = None) -> Iterator[RawRecord]:
        emitted = 0
        for file in _data_files(path):
            for index, payload in enumerate(_iter_json(file)):
                record = self._to_record(file, index, payload)
                if record is None:
                    continue
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _to_record(self, file: Path, index: int, payload: dict[str, Any]) -> RawRecord | None:
        turns = _structured_turns(payload) or _script_turns(payload)
        if not turns:
            return None
        record_id = str(payload.get("id") or payload.get("call_id") or f"{file.stem}:{index}")
        reliable = _diarization_is_reliable(turns)
        return RawRecord(
            source=self.source,
            record_id=record_id,
            channel=Channel.VOICE,
            turns=tuple(turns),
            domain_hint=None,
            metadata={
                "file": file.name,
                "diarization": str(payload.get("diarization") or "unknown"),
                # Recorded, not corrected. Published AIxBlock records are
                # `pause_segmented`: turns are split by silence, and the speaker label is
                # the same value for ~99% of them. Downstream stages that assume speaker
                # labels mean something need to know they do not here.
                "diarization_reliable": reliable,
            },
        )


#: Below this, one speaker holds so much of the conversation that the labels cannot be
#: describing a two-party call.
_DIARIZATION_MIN_MINORITY_SHARE = 0.05


def _diarization_is_reliable(turns: list[RawTurn]) -> bool:
    if len(turns) < 4:
        return True
    counts: dict[Speaker, int] = {}
    for turn in turns:
        counts[turn.speaker] = counts.get(turn.speaker, 0) + 1
    if len(counts) < 2:
        return False
    minority = min(counts.values())
    return minority / len(turns) >= _DIARIZATION_MIN_MINORITY_SHARE


def _structured_turns(payload: dict[str, Any]) -> list[RawTurn]:
    raw = next((payload[key] for key in _TURN_KEYS if isinstance(payload.get(key), list)), None)
    if raw is None:
        return []
    turns: list[RawTurn] = []
    for item in raw:
        if isinstance(item, str):
            turns.extend(_split_script(item))
            continue
        if not isinstance(item, dict):
            continue
        text = next(
            (str(item[k]).strip() for k in _TEXT_KEYS if str(item.get(k) or "").strip()), ""
        )
        if not text:
            continue
        label = next((str(item[k]) for k in _SPEAKER_KEYS if item.get(k)), None)
        turns.append(RawTurn(speaker=_speaker_from(label), text=text))
    return turns


def _script_turns(payload: dict[str, Any]) -> list[RawTurn]:
    script = next(
        (str(payload[key]) for key in _SCRIPT_KEYS if str(payload.get(key) or "").strip()), ""
    )
    return _split_script(script) if script else []


def _split_script(script: str) -> list[RawTurn]:
    turns: list[RawTurn] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _PREFIX.match(stripped)
        speaker = _speaker_from(match.group(1) or match.group(2)) if match else Speaker.CALLER
        text = stripped[match.end() :].strip() if match else stripped
        if text:
            turns.append(RawTurn(speaker=speaker, text=text))
    return turns


def _data_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted([*path.glob("*.jsonl"), *path.glob("*.json")])
    return [path] if path.is_file() else []


def _iter_json(file: Path) -> Iterator[dict[str, Any]]:
    text = file.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if isinstance(payload, list):
            yield from (item for item in payload if isinstance(item, dict))
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if isinstance(item, dict):
            yield item
