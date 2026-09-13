"""LiveKit WebRTC transport (ADR-0001).

Phase 1 ingress: browser and soft clients, no carrier, no SIP trunk. SIP lands later
behind this same ``AudioTransport`` interface, which is the whole reason the interface
exists.

**Not verified against a live server.** The LiveKit SDK is an optional extra and this
driver has not run against a real room; expect to adjust on first contact.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ccas.observability.logging import get_logger
from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame
from ccas.voice.transport import AudioTransport, DtmfSignal, TransportInfo

__all__ = ["LiveKitTransport"]

LOG = get_logger("voice.livekit")


class LiveKitTransport(AudioTransport):
    """Bridges a LiveKit room's audio tracks onto the transport contract."""

    def __init__(
        self,
        room: Any,
        participant: str = "",
        fmt: AudioFormat = DEFAULT_FORMAT,
        queue_size: int = 64,
    ) -> None:
        self._room = room
        self._participant = participant
        self.format = fmt
        self._outbound: asyncio.Queue[AudioFrame | None] = asyncio.Queue(queue_size)
        self._dtmf_queue: asyncio.Queue[DtmfSignal] = asyncio.Queue()
        self._closed = False

    @property
    def info(self) -> TransportInfo:
        return TransportInfo(
            driver="livekit",
            room=getattr(self._room, "name", None),
            participant=self._participant or None,
            format=self.format,
        )

    async def _inbound(self) -> AsyncIterator[AudioFrame]:
        try:
            from livekit import rtc
        except ImportError as exc:
            raise RuntimeError("livekit is not installed; run `uv sync --extra voice`") from exc

        track = await self._first_audio_track(rtc)
        stream = rtc.AudioStream(track, sample_rate=self.format.sample_rate, num_channels=1)
        clock = 0
        async for event in stream:
            if self._closed:
                return
            data = bytes(event.frame.data)
            frame = AudioFrame(data, clock, self.format)
            clock += frame.duration_ms
            yield frame

    async def _first_audio_track(self, rtc: Any) -> Any:
        """Wait for the caller's microphone.

        A room exists before anyone publishes to it, so a driver that assumed a track
        was present would fail on every call that connects a fraction early.
        """
        ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def on_subscribed(track: Any, *_: Any) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and not ready.done():
                ready.set_result(track)

        self._room.on("track_subscribed", on_subscribed)
        for participant in getattr(self._room, "remote_participants", {}).values():
            for publication in getattr(participant, "track_publications", {}).values():
                track = getattr(publication, "track", None)
                if track is not None and track.kind == rtc.TrackKind.KIND_AUDIO:
                    return track
        return await ready

    def receive(self) -> AsyncIterator[AudioFrame]:
        return self._inbound()

    async def send(self, frame: AudioFrame) -> None:
        if self._closed:
            return
        await self._outbound.put(frame)

    async def clear(self) -> None:
        """Drain the playback queue. The barge-in primitive."""
        dropped = 0
        while not self._outbound.empty():
            self._outbound.get_nowait()
            dropped += 1
        LOG.info("voice.livekit.cleared", frames_dropped=dropped)

    async def _dtmf(self) -> AsyncIterator[DtmfSignal]:
        while not self._closed:
            try:
                yield await asyncio.wait_for(self._dtmf_queue.get(), timeout=0.05)
            except TimeoutError:
                return

    def dtmf(self) -> AsyncIterator[DtmfSignal]:
        return self._dtmf()

    def push_dtmf(self, digit: str, at_ms: int) -> None:
        """Called by the room's SIP DTMF handler once SIP ingress lands."""
        self._dtmf_queue.put_nowait(DtmfSignal(digit, at_ms))

    async def pending_playback(self) -> AudioFrame | None:
        """Next frame to publish, or None when the call is closing."""
        return await self._outbound.get()

    async def close(self) -> None:
        self._closed = True
        await self._outbound.put(None)
