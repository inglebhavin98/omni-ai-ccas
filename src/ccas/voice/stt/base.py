"""Speech-to-text interface.

Streaming by contract: the turn loop needs partials to start speculative work and a
``final`` to know the caller has stopped. A batch-only STT would force the whole
utterance to be buffered before anything downstream could begin, which does not fit the
180 ms slice (CLAUDE.md Rule 3).

Transcripts here are **unredacted**. They are handed straight to the redaction pipeline;
nothing else may hold one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ccas.voice.audio import AudioFrame

__all__ = ["SpeechToText", "Transcript"]


@dataclass(frozen=True, slots=True)
class Transcript:
    """One STT emission. ``is_final`` marks end of utterance as the engine sees it."""

    text: str
    is_final: bool
    confidence: float = 1.0
    at_ms: int = 0
    latency_ms: int = 0

    def __repr__(self) -> str:
        kind = "final" if self.is_final else "partial"
        return f"Transcript({kind}, chars={len(self.text)}, conf={self.confidence:.2f})"

    __str__ = __repr__


class SpeechToText(ABC):
    name: str

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def stream(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        """Transcribe a live frame stream, yielding partials then a final per utterance."""

    async def aclose(self) -> None:
        return None
