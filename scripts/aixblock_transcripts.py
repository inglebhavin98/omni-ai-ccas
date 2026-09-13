"""AIxBlock archive -> adapter-ready dialogue records.

The corpus publishes themed ZIP archives of per-call AssemblyAI transcripts. Each
transcript carries a flat ``text`` with both parties merged, word-level timing, an ASR
confidence, and the upstream PII policy list -- but **no speaker diarization**: the
``speaker`` field on every word is null in every archive sampled.

So turns are recovered from inter-word silence, and each turn is attributed by a
conservative agent-boilerplate filter rather than a diarizer. Output is shaped for
``ccas.ingestion.adapters.aixblock.AixBlockAdapter``, which reads a ``turns`` list of
``{speaker, text}`` objects -- so hydrating the corpus needs no change to the adapter.

Pure and blocking by design: no redaction, no network, no logging. Redaction happens
once, downstream, in ``scripts/ingest.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = ["ARCHIVES", "DEFAULT_GAP_MS", "MIN_TURNS", "Archive", "read_archive", "segment_turns"]


@dataclass(frozen=True, slots=True)
class Archive:
    """One published archive, the file it feeds, and what the calls in it actually are.

    ``campaign`` and ``output`` are separate on purpose, because they disagree. The
    upstream ZIP names describe the BPO *campaign* that recorded the calls, not the
    vertical the calls belong to -- and an earlier version of this file read them as
    verticals. Profiling the transcripts showed `retail` is 100% insurance and 0% orders
    (ADR-0013), so the name was wrong and every consumer downstream inherited the error.

    ``output`` is therefore a path, not a claim. Route on ``campaign``; never infer a
    domain pack from ``output``.
    """

    filename: str
    output: str
    """Basename of the JSONL this archive feeds. ``retail``/``healthcare`` are known
    misnomers, kept only so existing paths stay valid -- see ADR-0013."""

    campaign: str
    """What the calls are about.

    Set from *measured* vocabulary where ``verified`` is True, and ``"unverified"``
    otherwise. It is deliberately never inferred from the archive name: the published
    names describe a BPO campaign and have now been wrong three times running.
    """

    verified: bool = False
    """Whether ``campaign`` was measured against the transcripts, or is a placeholder."""

    large: bool = False


#: ``medicare_inbound`` is 826 MB and is opt-in behind ``--include-large``.
#: ``campaign`` is measured, never read off the file name. Percentages are the share of
#: 300 sampled calls matching that vocabulary; an archive with no dominant topic stays
#: ``"unverified"`` rather than taking a second guess.
ARCHIVES: Final[tuple[Archive, ...]] = (
    # 97% final-expense / life-insurance vocabulary.
    Archive(
        "customer_service_general_inbound.zip", "retail", "life_insurance_sales", verified=True
    ),
    # Named "automotive-stereo"; measured 3% car-audio against 18% auto-insurance. No
    # dominant topic, so the name is discarded without a replacement being invented.
    Archive(
        "(re-uploaded)PII_Redacted_Transcripts_aixblock-automotive-stereo-inbound-104h.zip",
        "retail",
        "unverified",
    ),
    Archive("home_service_inbound.zip", "retail", "unverified", large=True),
    Archive("automotive_inbound.zip", "retail", "unverified", large=True),
    # Named "medical_equipment"; measured 89% Medicare vocabulary against 12% equipment.
    # These are coverage calls, and it is the corpus that clustered at 28.1%.
    Archive("medical_equipment_outbound.zip", "healthcare", "medicare_coverage", verified=True),
    # 47% Medicare, 9% auto-insurance -- mixed, no single dominant topic.
    Archive("automotive_and_healthcare_insurance_inbound.zip", "healthcare", "unverified"),
    Archive("medicare_inbound.zip", "healthcare", "unverified", large=True),
)

#: Call-centre and IVR boilerplate. Matching a cue marks a turn as the agent side.
#:
#: This is a *filter*, not a diarizer: the corpus carries no speaker labels, so claiming
#: to know who spoke would be an invention. A turn is called the agent's only when it
#: says something only an agent or an IVR says; everything else stays with the caller.
#: The bias is deliberate -- a missed agent turn adds boilerplate to the mining pool,
#: which a reviewer can see and prune, while a wrongly excluded caller turn silently
#: deletes the intent the corpus was mined for.
#:
#: Every phrase is generic contact-centre vocabulary; no vertical's words appear here.
_AGENT_CUES: Final = re.compile(
    r"""
    thank\s+you\s+for\s+(calling|choosing|holding|waiting|your\s+patience)
  | thanks\s+for\s+(calling|holding|waiting)
  | (your|this)\s+calls?\s+(may|will|is|are)\s+(be\s+)?(being\s+)?(monitored|recorded)
  | for\s+quality\s+(and|&)\s+(training|assurance)
  | (how|what)\s+(can|may)\s+i\s+(help|assist|direct)
  | is\s+there\s+anything\s+else\s+i\s+can\s+(help|do|assist)
  | if\s+you\s+know\s+your\s+party.s\s+extension
  | please\s+(press|dial|stay\s+on\s+the\s+line|hold|continue\s+to\s+hold)
  | press\s+(one|two|three|four|five|six|seven|eight|nine|zero|\d)\b
  | (one\s+moment|just\s+a\s+moment|bear\s+with\s+me)
  | let\s+me\s+(check|pull\s+up|look|transfer|connect|verify)
  | may\s+i\s+(have|get|ask|please\s+have)\s+your
  | can\s+i\s+(have|get)\s+your\s+(name|number|date)
  | i\s+apologi[sz]e\s+for\s+(the|any)
  | to\s+better\s+(assist|serve)\s+you
  | my\s+name\s+is\s+\[?\w+.{0,24}?\b(from|with|calling\s+from)\b
  | you.ve\s+reached\s+(the|our|\[)
  | all\s+(of\s+)?our\s+(agents|representatives)\s+are
  | have\s+a\s+(great|nice|good|wonderful)\s+(day|one|rest)
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: An inter-word silence at least this long ends a turn.
#:
#: This was 500 ms, chosen off the p90 intra-word gap (320 ms). That threshold separates
#: breath groups, not turns: it produced 200 segments per call with a median of 7 words,
#: and mining them returned 11.3% coverage. At 2000 ms the median utterance is 18 words
#: and the corpus shrinks 4.1x. The gap belongs to the consumer, not the acoustics --
#: barge-in tuning would want the small value back. See ADR-0016.
DEFAULT_GAP_MS: Final = 2000
#: A record with three turns or fewer is not a dialogue.
MIN_TURNS: Final = 4


def record_id(archive: str, member: str) -> str:
    """A stable, opaque record id.

    AIxBlock file names embed the dialled number -- ``..._6149286164_4712234-all...``.
    Carrying that into ``record_id`` would put a phone number into every CallLog and
    every log line downstream, so the name is hashed and the original never stored
    (Rule 2).
    """
    digest = hashlib.sha256(f"{archive}/{member}".encode()).hexdigest()
    return f"aix-{digest[:16]}"


def segment_turns(
    words: Sequence[dict[str, Any]], gap_ms: int = DEFAULT_GAP_MS
) -> list[dict[str, Any]]:
    """Recover turns from word-level timing.

    The transcript's flat ``text`` merges both parties into one string, so the word list
    is the only structure the corpus actually provides.
    """
    turns: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict) or not str(word.get("text") or "").strip():
            continue
        if buffer and _gap(buffer[-1], word) >= gap_ms:
            turns.append(_build_turn(buffer))
            buffer = []
        buffer.append(word)
    if buffer:
        turns.append(_build_turn(buffer))
    return [turn for turn in turns if turn["text"]]


def _gap(previous: dict[str, Any], current: dict[str, Any]) -> float:
    try:
        return float(current.get("start", 0)) - float(previous.get("end", 0))
    except (TypeError, ValueError):
        return 0.0


def _build_turn(words: Sequence[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(str(w["text"]).strip() for w in words).strip()
    return {
        "speaker": "agent" if _AGENT_CUES.search(text) else "customer",
        "text": text,
        "start_ms": _as_int(words[0].get("start")),
        "end_ms": _as_int(words[-1].get("end")),
    }


def _as_int(value: Any) -> int | None:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


def read_archive(
    archive: Path, gap_ms: int = DEFAULT_GAP_MS, min_turns: int = MIN_TURNS
) -> tuple[list[str], dict[str, int]]:
    """Turn one ZIP of transcripts into JSONL lines. Blocking -- call it in a thread."""
    lines: list[str] = []
    stats = {"members": 0, "unparsable": 0, "too_short": 0, "kept": 0, "turns": 0}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.endswith(".json") or member.startswith("__MACOSX"):
                continue
            stats["members"] += 1
            record = _to_record(bundle, archive.name, member, gap_ms, min_turns, stats)
            if record is not None:
                lines.append(json.dumps(record, ensure_ascii=False))
    return lines, stats


def _to_record(
    bundle: zipfile.ZipFile,
    archive: str,
    member: str,
    gap_ms: int,
    min_turns: int,
    stats: dict[str, int],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(bundle.read(member))
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, zipfile.BadZipFile):
        stats["unparsable"] += 1
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("words"), list):
        stats["unparsable"] += 1
        return None
    turns = segment_turns(payload["words"], gap_ms)
    if len(turns) < min_turns:
        stats["too_short"] += 1
        return None
    stats["kept"] += 1
    stats["turns"] += len(turns)
    return {
        "id": record_id(archive, member),
        "turns": turns,
        "audio_duration_s": payload.get("audio_duration"),
        "asr_confidence": payload.get("confidence"),
        "diarization": "pause_segmented",
    }
