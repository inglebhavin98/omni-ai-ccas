"""Audio primitives.

``AudioFrame`` carries raw caller audio, which is the most sensitive thing this system
touches -- it has not been through STT, so it has not been through redaction. It is
never a Pydantic field, never logged, and renders blind, for the same reasons
``RawRecord`` does (CLAUDE.md Rule 2).
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["DEFAULT_FORMAT", "AudioFormat", "AudioFrame", "silence", "tone"]


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2
    """Bytes per sample. 2 = PCM16, which is what every codec in the stack speaks."""

    @property
    def bytes_per_ms(self) -> float:
        return self.sample_rate * self.channels * self.sample_width / 1000

    def frame_bytes(self, duration_ms: int) -> int:
        return int(self.bytes_per_ms * duration_ms)


DEFAULT_FORMAT = AudioFormat()


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One chunk of PCM. Typically 20ms, matching what WebRTC delivers."""

    data: bytes
    timestamp_ms: int = 0
    format: AudioFormat = DEFAULT_FORMAT

    @property
    def duration_ms(self) -> int:
        return round(len(self.data) / self.format.bytes_per_ms)

    @property
    def sample_count(self) -> int:
        return len(self.data) // self.format.sample_width

    def samples(self) -> NDArray[np.int16]:
        return np.frombuffer(self.data, dtype=np.int16)

    def rms(self) -> float:
        """Normalised loudness, 0..1. The cheapest speech/silence signal there is."""
        if not self.data:
            return 0.0
        samples = self.samples().astype(np.float64)
        return float(np.sqrt(np.mean(samples**2)) / 32768.0)

    def __repr__(self) -> str:
        return f"AudioFrame({self.duration_ms}ms, {len(self.data)}B, t={self.timestamp_ms}ms)"

    __str__ = __repr__


def silence(duration_ms: int, fmt: AudioFormat = DEFAULT_FORMAT, at_ms: int = 0) -> AudioFrame:
    return AudioFrame(b"\x00" * fmt.frame_bytes(duration_ms), at_ms, fmt)


def tone(
    duration_ms: int,
    fmt: AudioFormat = DEFAULT_FORMAT,
    at_ms: int = 0,
    hz: float = 220.0,
    amplitude: float = 0.35,
) -> AudioFrame:
    """Synthetic voiced audio.

    Not speech -- it stands in for it, so a driver can produce deterministic
    speech/silence structure without shipping WAV files or a speech synthesiser.
    """
    count = int(fmt.sample_rate * duration_ms / 1000)
    peak = int(amplitude * 32767)
    samples = array.array(
        "h",
        (int(peak * math.sin(2 * math.pi * hz * (i / fmt.sample_rate))) for i in range(count)),
    )
    return AudioFrame(samples.tobytes(), at_ms, fmt)
