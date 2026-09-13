"""The voice module is frozen (ADR-0018) -- this asserts it stays that way.

Freezing is not deleting. The code, its tests and its budget all remain and must keep
passing under `make voice`; what stops is treating voice as a live constraint on the chat
work. The risk a freeze introduces is silent drift: someone edits the core, voice breaks,
nobody notices because voice is out of the default run. These tests are the tripwire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.security
def test_the_core_still_does_not_import_voice() -> None:
    """The freeze is only safe because the dependency runs one way.

    If the core ever imports `ccas.voice`, freezing voice would freeze the core with it,
    and a chat-only deployment would drag in LiveKit, Deepgram and Cartesia.
    """
    offenders = []
    for path in (REPO / "src" / "ccas").rglob("*.py"):
        if "voice" in path.relative_to(REPO / "src" / "ccas").parts:
            continue
        text = path.read_text()
        if "ccas.voice" in text or "from ccas import voice" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"core modules importing voice: {offenders}"


def test_voice_tests_still_exist() -> None:
    """A freeze that quietly became a deletion would pass every other check."""
    expected = (
        REPO / "tests" / "test_module_5_voice.py",
        REPO / "tests" / "unit" / "voice",
        REPO / "tests" / "latency",
        REPO / "src" / "ccas" / "voice",
    )
    missing = [str(p.relative_to(REPO)) for p in expected if not p.exists()]
    assert not missing, f"frozen, not deleted -- these must remain: {missing}"


def test_the_freeze_is_documented_where_someone_would_look() -> None:
    assert "frozen" in (REPO / "CLAUDE.md").read_text().lower()
    assert (REPO / "docs" / "adr" / "0018-freeze-voice-pivot-to-chat.md").is_file()
