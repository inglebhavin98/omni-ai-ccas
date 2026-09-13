"""A deterministic STT for tests and the mock driver.

Returns pre-scripted text per utterance rather than transcribing anything. That keeps a
voice-loop test a test of the *loop* -- VAD boundaries, redaction, graph invocation,
barge-in, the latency ledger -- instead of a test of an acoustic model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from ccas.voice.audio import AudioFrame
from ccas.voice.stt.base import SpeechToText, Transcript

__all__ = ["ScriptedStt"]


class ScriptedStt(SpeechToText):
    name = "scripted"

    def __init__(
        self,
        utterances: Sequence[str],
        partials: bool = True,
        latency_ms: int = 0,
        confidence: float = 0.95,
    ) -> None:
        self._utterances = list(utterances)
        self._partials = partials
        self._latency_ms = latency_ms
        self._confidence = confidence
        self._index = 0

    @property
    def available(self) -> bool:
        return True

    @property
    def remaining(self) -> int:
        return max(0, len(self._utterances) - self._index)

    def next_final(self, at_ms: int = 0) -> Transcript | None:
        """The next scripted utterance, or None once the script is spent."""
        if self._index >= len(self._utterances):
            return None
        text = self._utterances[self._index]
        self._index += 1
        return Transcript(
            text=text,
            is_final=True,
            confidence=self._confidence,
            at_ms=at_ms,
            latency_ms=self._latency_ms,
        )

    async def _iter(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        async for frame in frames:
            transcript = self.next_final(frame.timestamp_ms)
            if transcript is None:
                return
            if self._partials and transcript.text:
                head = transcript.text.split(" ", 1)[0]
                yield Transcript(head, False, self._confidence, frame.timestamp_ms)
            yield transcript

    def stream(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        return self._iter(frames)
