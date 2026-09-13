"""A deterministic TTS for tests and the mock driver.

Emits synthetic audio proportional to the text, so playback duration and barge-in
timing behave like the real thing without a synthesiser or a network call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame, tone
from ccas.voice.tts.base import TextToSpeech

__all__ = ["ScriptedTts"]

#: Roughly conversational pace. Only the ratio matters -- it sets how long a bot turn
#: lasts, and therefore how much of it a barge-in can interrupt.
MS_PER_CHARACTER = 55


class ScriptedTts(TextToSpeech):
    name = "scripted"

    def __init__(
        self,
        chunk_ms: int = 100,
        first_chunk_delay_ms: int = 0,
        fmt: AudioFormat = DEFAULT_FORMAT,
        pace_factor: float = 0.0,
    ) -> None:
        self.chunk_ms = chunk_ms
        self.first_chunk_delay_ms = first_chunk_delay_ms
        self.format = fmt
        self.pace_factor = pace_factor
        """Real seconds per simulated second of audio. Must match the transport's, or
        one side races ahead and barge-in timing becomes meaningless. Zero is instant."""
        self._cancelled = False
        self.spoken: list[str] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def cancel(self) -> None:
        self._cancelled = True

    async def _iter(self, text: str) -> AsyncIterator[AudioFrame]:
        self._cancelled = False
        self.spoken.append(text)
        total_ms = max(self.chunk_ms, len(text) * MS_PER_CHARACTER)
        if self.first_chunk_delay_ms:
            await asyncio.sleep(self.first_chunk_delay_ms / 1000)
        for offset in range(0, total_ms, self.chunk_ms):
            if self._cancelled:
                return
            yield tone(self.chunk_ms, self.format, at_ms=offset)
            # Yield control so a concurrent barge-in can land mid-utterance.
            await asyncio.sleep(self.chunk_ms * self.pace_factor / 1000)

    def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        return self._iter(text)
