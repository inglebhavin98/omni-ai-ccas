"""Voice activity detection.

Two implementations. ``EnergyVad`` is arithmetic -- no model, no download, deterministic,
and good enough to drive the turn loop in CI. ``SileroVad`` is the real one (Rule 5) and
is lazily imported, so the base install does not carry a torch model it may never use.

The interesting output is not "is this speech" but *end of speech*: the moment after
which the caller has stopped and the platform may answer. That is the first 100 ms of
the latency budget, and getting it wrong is either a bot that interrupts or a bot that
leaves dead air.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccas.voice.audio import AudioFrame

__all__ = [
    "SILERO_WINDOW_SAMPLES",
    "EnergyVad",
    "SileroVad",
    "SpeechEvent",
    "VadDecision",
    "VadState",
    "VoiceActivityDetector",
]


class SpeechEvent(StrEnum):
    NONE = "none"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    """End of utterance. The turn may now be processed."""


@dataclass(frozen=True, slots=True)
class VadDecision:
    event: SpeechEvent
    is_speech: bool
    at_ms: int
    speech_duration_ms: int = 0
    silence_duration_ms: int = 0


class VadState(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"


class VoiceActivityDetector(ABC):
    """Frame-by-frame. Stateful: the whole point is the transition."""

    name: str

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def push(self, frame: AudioFrame) -> VadDecision: ...

    @abstractmethod
    def reset(self) -> None: ...


class _SpeechGate:
    """The hangover state machine, shared by every detector.

    What differs between detectors is only how a frame is judged speech or not. The
    transition logic -- how long before we believe it, how long before we call it over --
    is the same, and is the part that has to be right.
    """

    __slots__ = (
        "_pending_ms",
        "_silence_ms",
        "_speech_ms",
        "_state",
        "min_silence_ms",
        "min_speech_ms",
    )

    def __init__(self, min_speech_ms: int, min_silence_ms: int) -> None:
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._state = VadState.IDLE
        self._speech_ms = 0
        self._silence_ms = 0
        self._pending_ms = 0

    @property
    def state(self) -> VadState:
        return self._state

    def reset(self) -> None:
        self._state = VadState.IDLE
        self._speech_ms = 0
        self._silence_ms = 0
        self._pending_ms = 0

    def step(self, loud: bool, duration_ms: int, at_ms: int) -> VadDecision:
        if self._state is VadState.IDLE:
            if not loud:
                self._pending_ms = 0
                return VadDecision(SpeechEvent.NONE, False, at_ms)
            self._pending_ms += duration_ms
            if self._pending_ms < self.min_speech_ms:
                # Not yet convinced. A cough or a door closing is loud too.
                return VadDecision(SpeechEvent.NONE, False, at_ms)
            self._state = VadState.SPEAKING
            self._speech_ms = self._pending_ms
            self._silence_ms = 0
            self._pending_ms = 0
            return VadDecision(SpeechEvent.SPEECH_STARTED, True, at_ms, self._speech_ms)

        if loud:
            self._speech_ms += duration_ms
            self._silence_ms = 0
            return VadDecision(SpeechEvent.NONE, True, at_ms, self._speech_ms)

        self._silence_ms += duration_ms
        if self._silence_ms < self.min_silence_ms:
            # Inside the hangover: a pause for breath, not the end of a sentence.
            return VadDecision(SpeechEvent.NONE, True, at_ms, self._speech_ms, self._silence_ms)

        spoken, silent = self._speech_ms, self._silence_ms
        self.reset()
        return VadDecision(SpeechEvent.SPEECH_ENDED, False, at_ms, spoken, silent)


class EnergyVad(VoiceActivityDetector):
    """RMS threshold with hangover. No model, deterministic, good enough for CI.

    ``min_silence_ms`` is the tunable that matters: too short and the bot cuts in on a
    pause for breath, too long and every turn pays for it. Budgeted at 100 ms in
    configs/latency_budget.yaml.
    """

    name = "energy"

    def __init__(
        self,
        threshold: float = 0.02,
        min_speech_ms: int = 100,
        min_silence_ms: int = 100,
    ) -> None:
        self.threshold = threshold
        self._gate = _SpeechGate(min_speech_ms, min_silence_ms)

    @property
    def available(self) -> bool:
        return True

    @property
    def state(self) -> VadState:
        return self._gate.state

    def reset(self) -> None:
        self._gate.reset()

    def push(self, frame: AudioFrame) -> VadDecision:
        return self._gate.step(frame.rms() >= self.threshold, frame.duration_ms, frame.timestamp_ms)


#: Silero is trained on fixed-size windows; anything else gives meaningless probabilities.
SILERO_WINDOW_SAMPLES = 512


class SileroVad(VoiceActivityDetector):
    """The locked choice (Rule 5). Lazily loaded; unavailability is reported, not silent.

    Frames arrive at whatever size the transport delivers (20 ms = 320 samples at 16 kHz),
    so they are buffered into the model's fixed window before inference.
    """

    name = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 100,
        min_silence_ms: int = 100,
        sample_rate: int = 16_000,
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._gate = _SpeechGate(min_speech_ms, min_silence_ms)
        self._buffer: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._last_probability = 0.0
        self._model: Any = None
        self._load_error: str | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            self._load_error = "silero_vad_not_installed"
            return None
        try:
            self._model = load_silero_vad()
        except Exception as exc:  # a load failure must be reported, never silent
            self._load_error = f"silero_load_failed:{type(exc).__name__}"
            return None
        return self._model

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_model()
        return self._load_error

    @property
    def state(self) -> VadState:
        return self._gate.state

    @property
    def last_probability(self) -> float:
        return self._last_probability

    def reset(self) -> None:
        self._gate.reset()
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, frame: AudioFrame) -> VadDecision:
        model = self._ensure_model()
        if model is None:
            raise RuntimeError(
                f"silero VAD unavailable ({self._load_error}); run "
                "`uv sync --extra voice`, or configure the energy VAD explicitly"
            )
        import torch

        self._buffer = np.concatenate([self._buffer, frame.samples().astype(np.float32) / 32768.0])
        probabilities: list[float] = []
        while self._buffer.size >= SILERO_WINDOW_SAMPLES:
            window = self._buffer[:SILERO_WINDOW_SAMPLES]
            self._buffer = self._buffer[SILERO_WINDOW_SAMPLES:]
            probabilities.append(
                float(model(torch.from_numpy(window.copy()), self.sample_rate).item())
            )

        if probabilities:
            # Max over the windows in this frame: a frame is speech if any part of it is.
            self._last_probability = max(probabilities)
        return self._gate.step(
            self._last_probability >= self.threshold,
            frame.duration_ms,
            frame.timestamp_ms,
        )
