from __future__ import annotations

import pytest

from ccas.voice.dtmf import DtmfCollector
from ccas.voice.transport import DtmfSignal

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice


def press(collector: DtmfCollector, digits: str, start_ms: int = 0, step_ms: int = 300):
    """Type a sequence. Returns the *first* entry it completes.

    Digits after a completion legitimately begin a new entry, so taking the last return
    value would discard the one the test is about.
    """
    completed = None
    for index, digit in enumerate(digits):
        entry = collector.push(DtmfSignal(digit, start_ms + index * step_ms))
        if entry is not None and completed is None:
            completed = entry
    return completed


def test_the_terminator_completes_an_entry() -> None:
    entry = press(DtmfCollector(), "884210#")
    assert entry is not None
    assert entry.digits == "884210"
    assert entry.terminated


def test_an_entry_is_pending_until_it_completes() -> None:
    collector = DtmfCollector()
    assert press(collector, "8842") is None
    assert collector.pending
    assert collector.length == 4


def test_star_clears_the_entry() -> None:
    """Universally understood as "I got that wrong, let me start again"."""
    collector = DtmfCollector()
    press(collector, "8842*")
    assert not collector.pending
    entry = press(collector, "1234#")
    assert entry is not None
    assert entry.digits == "1234"


def test_a_full_buffer_completes_without_a_terminator() -> None:
    collector = DtmfCollector(max_digits=4)
    entry = press(collector, "12345")
    assert entry is not None
    assert entry.digits == "1234"
    assert not entry.terminated
    # The overflow digit begins the next entry rather than being dropped.
    assert collector.length == 1


def test_an_idle_entry_expires() -> None:
    """Callers routinely type a reference and simply stop."""
    collector = DtmfCollector(inter_digit_timeout_ms=2000)
    press(collector, "884210", start_ms=0, step_ms=100)
    assert collector.expire(1000) is None
    entry = collector.expire(3000)
    assert entry is not None
    assert entry.digits == "884210"
    assert not entry.terminated


def test_expiry_with_nothing_pending_is_a_no_op() -> None:
    assert DtmfCollector().expire(99_999) is None


def test_a_lone_terminator_produces_nothing() -> None:
    assert DtmfCollector().push(DtmfSignal("#", 0)) is None


def test_the_entry_records_when_typing_started() -> None:
    entry = press(DtmfCollector(), "123#", start_ms=5000, step_ms=250)
    assert entry is not None
    assert entry.at_ms == 5000


def test_an_entry_never_renders_its_digits() -> None:
    """A four-digit run is a PIN; a traceback must not print it."""
    entry = press(DtmfCollector(), "4821#")
    assert entry is not None
    assert "4821" not in f"{entry!r} {entry!s}"


@pytest.mark.parametrize("bad", ["", "12", "x", "!"])
def test_a_signal_must_be_one_valid_digit(bad: str) -> None:
    with pytest.raises(ValueError, match="not a DTMF digit"):
        DtmfSignal(bad, 0)
