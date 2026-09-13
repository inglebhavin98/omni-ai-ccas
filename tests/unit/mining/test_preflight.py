"""The memory preflight (docs/future-scoped-work.md 9.15).

A process that cannot get memory does not fail -- it sits. Twice on this project a mining
run stalled indefinitely in uninterruptible sleep at the encoder load, on a machine that
had run the same work an hour earlier. There is no back-pressure and no diagnostic, so the
guard has to be ours and it has to run *before* the allocation.
"""

from __future__ import annotations

import pytest

from ccas.mining.preflight import (
    InsufficientMemoryError,
    available_memory_mb,
    parse_meminfo,
    parse_vm_stat,
    require_memory,
)


def test_enough_headroom_is_silent() -> None:
    require_memory(needed_mb=1000, available_mb=4000, what="encoder")


def test_too_little_headroom_raises_with_both_numbers() -> None:
    with pytest.raises(InsufficientMemoryError) as excinfo:
        require_memory(needed_mb=1300, available_mb=200, what="bge-large")
    message = str(excinfo.value)
    assert "bge-large" in message
    assert "1300" in message and "200" in message
    # It must say what to do, not just that it refused.
    assert "close" in message.lower() or "free" in message.lower()


def test_the_margin_is_applied_above_the_bare_requirement() -> None:
    """Matching the requirement exactly means thrashing, not succeeding."""
    with pytest.raises(InsufficientMemoryError):
        require_memory(needed_mb=1000, available_mb=1000, what="encoder")
    require_memory(needed_mb=1000, available_mb=1500, what="encoder")


def test_unknown_memory_does_not_block() -> None:
    """A guard against stalling must not itself become a reason a run cannot start."""
    require_memory(needed_mb=99_999, available_mb=None, what="encoder")


def test_parse_meminfo_reads_memavailable() -> None:
    text = "MemTotal:       16384000 kB\nMemFree:          512000 kB\nMemAvailable:    2048000 kB\n"
    assert parse_meminfo(text) == 2000


def test_parse_meminfo_without_memavailable_is_unknown() -> None:
    assert parse_meminfo("MemTotal: 16384000 kB\n") is None


def test_parse_vm_stat_counts_reclaimable_pages() -> None:
    """Free alone understates macOS headroom: inactive and speculative are reclaimable."""
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                   100000.\n"
        "Pages active:                                 200000.\n"
        "Pages inactive:                                50000.\n"
        "Pages speculative:                             10000.\n"
    )
    # (100000 + 50000 + 10000) pages * 16384 bytes = 2500 MiB
    assert parse_vm_stat(text) == 2500


def test_parse_vm_stat_on_junk_is_unknown() -> None:
    assert parse_vm_stat("not vm_stat output") is None


def test_available_memory_is_a_plausible_number_or_none() -> None:
    """The real reader, on whatever platform this runs."""
    value = available_memory_mb()
    assert value is None or 0 < value < 1_000_000


def test_the_encoder_declares_what_it_needs() -> None:
    """The guard is only useful if the number it checks against is real."""
    from ccas.mining.embedder import MODEL_RESIDENT_MB

    assert MODEL_RESIDENT_MB["BAAI/bge-large-en-v1.5"] >= 1000
    # An unknown model must not silently check against zero.
    assert MODEL_RESIDENT_MB.get("something/unheard-of") is None
