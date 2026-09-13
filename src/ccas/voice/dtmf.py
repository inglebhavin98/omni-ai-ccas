"""Touch-tone collection.

Speech recognition fails on noisy lines and unfamiliar accents, and the legacy-migration
analysis names DTMF fallback as a gap that sinks IVR replacements. A slot marked
``dtmf_capturable`` can be answered with the keypad instead.

Digits are masked before storage at the normalizer (runs of four or more are a PIN), so
this collector holds them only for the moment between keypress and slot capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ccas.voice.transport import DtmfSignal

__all__ = ["DTMF_TERMINATOR", "DtmfCollector", "DtmfEntry"]

DTMF_TERMINATOR = "#"
CLEAR_DIGIT = "*"


@dataclass(frozen=True, slots=True)
class DtmfEntry:
    digits: str
    at_ms: int
    terminated: bool
    """True when the caller pressed #, false when the entry timed out or filled up."""

    def __repr__(self) -> str:
        return f"DtmfEntry(len={len(self.digits)}, at={self.at_ms}ms, term={self.terminated})"

    __str__ = __repr__


@dataclass(slots=True)
class DtmfCollector:
    """Accumulates keypresses into one entry.

    Three ways to finish: the terminator, a maximum length, or an inter-digit timeout --
    because callers frequently type a reference and then simply stop.
    """

    max_digits: int = 24
    inter_digit_timeout_ms: int = 4000
    _digits: list[str] = field(default_factory=list)
    _first_at_ms: int = 0
    _last_at_ms: int = 0

    @property
    def pending(self) -> bool:
        return bool(self._digits)

    @property
    def length(self) -> int:
        return len(self._digits)

    def reset(self) -> None:
        self._digits.clear()
        self._first_at_ms = 0
        self._last_at_ms = 0

    def push(self, signal: DtmfSignal) -> DtmfEntry | None:
        """Add a keypress. Returns an entry when the collection completes."""
        if signal.digit == CLEAR_DIGIT:
            # Universally understood as "I got that wrong, let me start again".
            self.reset()
            return None

        if not self._digits:
            self._first_at_ms = signal.at_ms
        self._last_at_ms = signal.at_ms

        if signal.digit == DTMF_TERMINATOR:
            return self._finish(terminated=True)

        self._digits.append(signal.digit)
        if len(self._digits) >= self.max_digits:
            return self._finish(terminated=False)
        return None

    def expire(self, now_ms: int) -> DtmfEntry | None:
        """Close the entry if the caller has stopped typing."""
        if not self._digits:
            return None
        if now_ms - self._last_at_ms < self.inter_digit_timeout_ms:
            return None
        return self._finish(terminated=False)

    def _finish(self, terminated: bool) -> DtmfEntry | None:
        if not self._digits:
            self.reset()
            return None
        entry = DtmfEntry("".join(self._digits), self._first_at_ms, terminated)
        self.reset()
        return entry
