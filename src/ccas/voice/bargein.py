"""Barge-in: stopping mid-sentence when the caller starts talking.

The single most-complained-about behaviour of the systems this replaces is a bot that
keeps speaking over you. Three things must happen, in this order and fast:

1. tell the transport to drop queued playback -- the caller stops hearing us;
2. cancel TTS generation -- we stop producing more;
3. cancel the in-flight turn -- we stop reasoning about a question they abandoned.

Order matters. Cancelling generation first still leaves whatever is already queued
playing, which is exactly what the caller is complaining about.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ccas.observability.logging import get_logger
from ccas.voice.transport import AudioTransport
from ccas.voice.tts.base import TextToSpeech

__all__ = ["BargeInController", "BargeInResult"]

LOG = get_logger("voice.bargein")


@dataclass(frozen=True, slots=True)
class BargeInResult:
    interrupted: bool
    played_ms_before: int
    cancelled_turn: bool


@dataclass(slots=True)
class BargeInController:
    transport: AudioTransport
    tts: TextToSpeech
    enabled: bool = True
    min_played_ms: int = 250
    """Ignore interruptions in the first moments of a bot turn.

    Callers routinely say "yeah" or "mm" over an opening word without meaning to
    interrupt; treating that as barge-in makes the bot restart constantly.
    """

    count: int = field(default=0)
    _speaking_task: asyncio.Task[None] | None = field(default=None)
    _turn_task: asyncio.Task[object] | None = field(default=None)

    def begin(
        self,
        speaking: asyncio.Task[None] | None = None,
        turn: asyncio.Task[object] | None = None,
    ) -> None:
        self._speaking_task = speaking
        self._turn_task = turn

    def end(self) -> None:
        self._speaking_task = None
        self._turn_task = None

    async def on_caller_speech(self, played_ms: int, correlation_id: str) -> BargeInResult:
        """Called when the VAD reports speech while the bot is playing audio."""
        if not self.enabled or played_ms < self.min_played_ms:
            return BargeInResult(False, played_ms, False)

        await self.transport.clear()
        await self.tts.cancel()

        cancelled_turn = False
        for task in (self._speaking_task, self._turn_task):
            if task is not None and not task.done():
                task.cancel()
                cancelled_turn = cancelled_turn or task is self._turn_task
        self.end()

        self.count += 1
        LOG.info(
            "voice.barge_in",
            correlation_id=correlation_id,
            played_ms=played_ms,
            cancelled_turn=cancelled_turn,
            total=self.count,
        )
        return BargeInResult(True, played_ms, cancelled_turn)
