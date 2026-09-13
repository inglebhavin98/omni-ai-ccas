"""Refuse to start rather than thrash (docs/future-scoped-work.md 9.15).

A process that cannot get memory does not fail -- it sits. Twice on this project a mining
run stalled indefinitely in uninterruptible sleep while loading the sentence encoder, on a
machine that had run the same work an hour earlier; a background watcher was killed by the
OS before either run produced a line of output. There is no back-pressure from the
allocator and no diagnostic, so the check has to be ours and it has to run *before* the
allocation rather than after.

Deliberately best-effort. When available memory cannot be determined the run proceeds: a
guard against stalling must not itself become a reason a run cannot start. This is a
performance guard, not a safety gate -- Rule 2's fail-closed rule is about egress, and
nothing here touches egress.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from ccas.observability.logging import get_logger

__all__ = [
    "HEADROOM_MARGIN",
    "InsufficientMemoryError",
    "available_memory_mb",
    "check_headroom",
    "parse_meminfo",
    "parse_vm_stat",
    "require_memory",
]

LOG = get_logger("mining.preflight")

#: Matching the requirement exactly means swapping, not succeeding. The encoder needs room
#: for its weights *and* the batch it is encoding, and the OS needs room to not evict them.
HEADROOM_MARGIN = 1.25

#: `vm_stat` is the only portable reader on macOS and it is not on any hot path -- this
#: runs once, before a multi-minute encode.
_VM_STAT_TIMEOUT_S = 2.0


class InsufficientMemoryError(RuntimeError):
    """Refusing to start work the machine cannot hold."""


def parse_meminfo(text: str) -> int | None:
    """MiB available, from Linux ``/proc/meminfo``.

    ``MemAvailable`` rather than ``MemFree``: the kernel's own estimate of what a new
    allocation can actually get, which is the question being asked.
    """
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) // 1024
    return None


def parse_vm_stat(text: str) -> int | None:
    """MiB available, from macOS ``vm_stat``.

    Free pages alone badly understate headroom -- inactive and speculative pages are
    reclaimable on demand, and on a browser-heavy desktop they are most of what is
    reclaimable.
    """
    page_size = 4096
    header = text.splitlines()[0] if text else ""
    if "page size of" in header:
        digits = "".join(c for c in header.split("page size of")[1] if c.isdigit())
        if digits:
            page_size = int(digits)

    counts: dict[str, int] = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        digits = value.strip().rstrip(".")
        if digits.isdigit():
            counts[name.strip()] = int(digits)

    reclaimable = sum(
        counts.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative")
    )
    if reclaimable == 0:
        return None
    return reclaimable * page_size // (1024 * 1024)


def available_memory_mb() -> int | None:
    """Best-effort available memory. ``None`` means "could not tell"."""
    system = platform.system()
    try:
        if system == "Linux":
            return parse_meminfo(Path("/proc/meminfo").read_text())
        if system == "Darwin":
            binary = shutil.which("vm_stat")
            if binary is None:
                return None
            completed = subprocess.run(  # noqa: S603 - fixed binary, no shell, no user input
                [binary], capture_output=True, text=True, timeout=_VM_STAT_TIMEOUT_S, check=False
            )
            return parse_vm_stat(completed.stdout) if completed.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        # Unreadable is the same as unknown, and unknown does not block.
        return None
    return None


def require_memory(needed_mb: int, what: str, available_mb: int | None) -> None:
    """Raise rather than let the machine thrash on an allocation it cannot serve.

    ``available_mb`` is passed in rather than read here, so this stays pure and the one
    ambiguous reading -- ``None`` meaning "unknown" versus "go and look" -- cannot arise.
    Callers pair it with ``available_memory_mb()``; I/O at the edges (Rule 8).
    """
    available = available_mb
    if available is None:
        LOG.info("mining.preflight.unknown", correlation_id="preflight", needed_mb=needed_mb)
        return

    required = int(needed_mb * HEADROOM_MARGIN)
    if available >= required:
        return

    LOG.warning(
        "mining.preflight.refused",
        correlation_id="preflight",
        what=what,
        needed_mb=needed_mb,
        available_mb=available,
    )
    raise InsufficientMemoryError(
        f"{what} needs about {needed_mb} MB resident and only {available} MB is available "
        f"(want {required} MB with headroom). Starting anyway does not fail -- it stalls in "
        f"uninterruptible sleep, sometimes for hours. Close what you can to free memory, or "
        f"run this on an idle machine."
    )


def check_headroom(needed_mb: int, what: str) -> None:
    """``require_memory`` against the machine's actual state. The edge."""
    require_memory(needed_mb, what, available_memory_mb())
