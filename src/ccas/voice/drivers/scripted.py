"""Deterministic transport for CI and the latency bench.

Emits synthetic speech/silence structure so the VAD produces real transitions, and
records everything sent back so a test can assert on playback and on barge-in truncation.
No media server, no network, no WAV files (ADR-0001).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame, silence, tone
from ccas.voice.transport import AudioTransport, DtmfSignal, TransportInfo

__all__ = ["ScriptedTransport", "ScriptedUtterance"]


@dataclass(slots=True)
class ScriptedUtterance:
    """One caller turn: how long they speak, then how long they pause."""

    speech_ms: int = 600
    trailing_silence_ms: int = 400
    dtmf: str = ""
    """Digits pressed during this turn, if any."""

    barge_in: bool = False
    """Speak while the bot is still playing. Drives the interruption path."""


@dataclass(slots=True)
class ScriptedTransport(AudioTransport):
    utterances: Sequence[ScriptedUtterance] = field(default_factory=list)
    frame_ms: int = 20
    lead_in_silence_ms: int = 200
    format: AudioFormat = DEFAULT_FORMAT
    pace_factor: float = 0.0
    """Real seconds per simulated second. 0 runs as fast as possible.

    Set it -- to the *same* value on the TTS -- when a test needs playback and listening
    to overlap in the right proportion. Without a shared clock the transport races ahead
    and the bot appears to have played 40ms of a 7s greeting by the time the caller
    interrupts, so barge-in is suppressed as a stray "mm"."""
    played: list[AudioFrame] = field(default_factory=list)
    clears: int = field(default=0)
    _dtmf_queue: asyncio.Queue[DtmfSignal] = field(default_factory=asyncio.Queue)
    _closed: bool = field(default=False)

    @property
    def info(self) -> TransportInfo:
        return TransportInfo(driver="scripted", format=self.format)

    @property
    def played_ms(self) -> int:
        return sum(frame.duration_ms for frame in self.played)

    async def _frames(self) -> AsyncIterator[AudioFrame]:
        clock = 0
        for _ in range(self.lead_in_silence_ms // self.frame_ms):
            yield silence(self.frame_ms, self.format, clock)
            clock += self.frame_ms
            await self._tick()

        for utterance in self.utterances:
            for digit in utterance.dtmf:
                await self._dtmf_queue.put(DtmfSignal(digit, clock))
            for _ in range(max(1, utterance.speech_ms // self.frame_ms)):
                yield tone(self.frame_ms, self.format, clock)
                clock += self.frame_ms
                await self._tick()
            for _ in range(max(1, utterance.trailing_silence_ms // self.frame_ms)):
                yield silence(self.frame_ms, self.format, clock)
                clock += self.frame_ms
                await self._tick()

    async def _tick(self) -> None:
        await asyncio.sleep(self.frame_ms * self.pace_factor / 1000)

    def receive(self) -> AsyncIterator[AudioFrame]:
        return self._frames()

    async def send(self, frame: AudioFrame) -> None:
        if self._closed:
            raise RuntimeError("transport is closed")
        self.played.append(frame)

    async def clear(self) -> None:
        """Barge-in. Drops queued playback, exactly as a real transport must."""
        self.clears += 1
        self.played.clear()

    async def _dtmf(self) -> AsyncIterator[DtmfSignal]:
        """Drain whatever the script has queued, then stop.

        A real transport keeps this open for the life of the call; the scripted one is
        finite by design, so the turn loop terminates without a timeout.
        """
        while True:
            try:
                yield self._dtmf_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def dtmf(self) -> AsyncIterator[DtmfSignal]:
        return self._dtmf()

    async def close(self) -> None:
        self._closed = True
