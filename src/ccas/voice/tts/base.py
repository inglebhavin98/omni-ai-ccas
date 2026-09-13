"""Text-to-speech interface.

Chunked by contract. The budget allows 120 ms to *first audio*, not to complete audio,
so a synthesiser that returns one buffer at the end cannot meet it however fast it is.

``cancel`` is the barge-in half of the contract: generation must stop mid-sentence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ccas.voice.audio import AudioFrame

__all__ = ["TextToSpeech"]


class TextToSpeech(ABC):
    name: str

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        """Yield audio chunks as they are produced, first chunk as early as possible."""

    @abstractmethod
    async def cancel(self) -> None:
        """Stop generating. Called on barge-in, and must return promptly."""

    async def aclose(self) -> None:
        return None
