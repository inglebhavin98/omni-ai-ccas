"""Deepgram Nova-3 streaming STT (CLAUDE.md Rule 5).

Streaming WebSocket with native ``speech_final`` events, which is why it is the locked
choice: the turn loop needs to know the caller has stopped without waiting on its own
silence timer.

**Not verified against the live service.** Written to the documented protocol and
exercised against a fake socket. Expect to adjust on first contact -- the same caveat
that applies to the corpus adapters.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

from ccas.observability.logging import get_logger
from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame
from ccas.voice.stt.base import SpeechToText, Transcript

__all__ = ["DEFAULT_MODEL", "DeepgramStt", "WebSocketLike", "parse_message"]

LOG = get_logger("voice.stt.deepgram")
DEFAULT_MODEL = "nova-3"
ENDPOINT = "wss://api.deepgram.com/v1/listen"


class WebSocketLike(Protocol):
    """The slice of a websocket client this adapter uses."""

    async def send(self, data: bytes | str) -> None: ...
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...
    async def close(self) -> None: ...


def parse_message(raw: str | bytes) -> Transcript | None:
    """Turn one socket message into a transcript, or None if it carries no text.

    Pure, so the wire format is testable without a socket -- which matters for an
    integration nobody here can exercise live.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    # Valid JSON that is not an object is still not a result. One odd frame must not
    # take the transcription stream down mid-call.
    if not isinstance(payload, dict):
        return None
    if payload.get("type") not in (None, "Results"):
        return None

    channel = payload.get("channel")
    alternatives = (channel or {}).get("alternatives") if isinstance(channel, dict) else None
    if not alternatives or not isinstance(alternatives, list):
        return None
    best = alternatives[0]
    if not isinstance(best, dict):
        return None
    text = str(best.get("transcript") or "").strip()
    if not text:
        return None

    # `speech_final` means the endpointer decided the caller stopped; `is_final` only
    # means this segment will not be revised. The turn loop cares about the former.
    is_final = bool(payload.get("speech_final") or payload.get("is_final"))
    start_s = float(payload.get("start") or 0.0)
    return Transcript(
        text=text,
        is_final=is_final,
        confidence=float(best.get("confidence") or 0.0),
        at_ms=int(start_s * 1000),
    )


def query_params(model: str, fmt: AudioFormat, language: str, endpointing_ms: int) -> str:
    params = {
        "model": model,
        "encoding": "linear16",
        "sample_rate": str(fmt.sample_rate),
        "channels": str(fmt.channels),
        "language": language,
        "punctuate": "true",
        "interim_results": "true",
        "endpointing": str(endpointing_ms),
    }
    return "&".join(f"{k}={v}" for k, v in params.items())


class DeepgramStt(SpeechToText):
    name = "deepgram"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fmt: AudioFormat = DEFAULT_FORMAT,
        language: str = "en",
        endpointing_ms: int = 100,
        socket: WebSocketLike | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.format = fmt
        self.language = language
        self.endpointing_ms = endpointing_ms
        self._socket = socket

    @property
    def available(self) -> bool:
        return self._socket is not None or bool(self._api_key)

    @property
    def url(self) -> str:
        params = query_params(self.model, self.format, self.language, self.endpointing_ms)
        return f"{ENDPOINT}?{params}"

    async def _connect(self) -> WebSocketLike:
        if self._socket is not None:
            return self._socket
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is unset; cannot open a transcription socket")
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets is not installed; run `uv sync --extra voice`") from exc
        self._socket = await websockets.connect(
            self.url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )
        return self._socket

    async def _iter(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        import asyncio

        socket = await self._connect()

        async def pump() -> None:
            async for frame in frames:
                await socket.send(frame.data)

        pumping = asyncio.create_task(pump())
        try:
            async for message in socket:
                transcript = parse_message(message)
                if transcript is not None:
                    yield transcript
        finally:
            pumping.cancel()

    def stream(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        return self._iter(frames)

    async def aclose(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
