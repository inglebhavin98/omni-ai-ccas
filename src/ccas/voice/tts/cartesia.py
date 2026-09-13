"""Cartesia Sonic streaming TTS (CLAUDE.md Rule 5).

Chunked output with a low time-to-first-byte, which is the property the 120 ms slice
actually needs -- total synthesis time matters far less than when the caller first hears
something.

**Not verified against the live service.** Written to the documented protocol and
exercised against a fake transport.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ccas.observability.logging import get_logger
from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame
from ccas.voice.tts.base import TextToSpeech

__all__ = ["DEFAULT_MODEL", "CartesiaTts"]

LOG = get_logger("voice.tts.cartesia")
DEFAULT_MODEL = "sonic-2"
ENDPOINT = "https://api.cartesia.ai/tts/bytes"
API_VERSION = "2024-06-10"


class CartesiaTts(TextToSpeech):
    name = "cartesia"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "",
        model: str = DEFAULT_MODEL,
        fmt: AudioFormat = DEFAULT_FORMAT,
        chunk_ms: int = 40,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.format = fmt
        self.chunk_ms = chunk_ms
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=2.0))
        self._cancelled = False

    @property
    def available(self) -> bool:
        return bool(self._api_key and self.voice_id)

    def request_body(self, text: str) -> dict[str, object]:
        return {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "language": "en",
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.format.sample_rate,
            },
        }

    async def cancel(self) -> None:
        self._cancelled = True

    async def _iter(self, text: str) -> AsyncIterator[AudioFrame]:
        self._cancelled = False
        if not self.available:
            raise RuntimeError(
                "CARTESIA_API_KEY or the voice id is unset; cannot synthesise speech"
            )

        chunk_bytes = self.format.frame_bytes(self.chunk_ms)
        offset = 0
        buffer = bytearray()
        async with self._client.stream(
            "POST",
            ENDPOINT,
            json=self.request_body(text),
            headers={
                "X-API-Key": self._api_key or "",
                "Cartesia-Version": API_VERSION,
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise RuntimeError(f"cartesia returned {response.status_code}")
            async for piece in response.aiter_bytes():
                if self._cancelled:
                    return
                buffer.extend(piece)
                # Re-chunk to a fixed frame size: a transport expects uniform frames,
                # and the vendor's chunking is a property of its encoder, not of us.
                while len(buffer) >= chunk_bytes and not self._cancelled:
                    yield AudioFrame(bytes(buffer[:chunk_bytes]), offset, self.format)
                    del buffer[:chunk_bytes]
                    offset += self.chunk_ms
        if buffer and not self._cancelled:
            yield AudioFrame(bytes(buffer), offset, self.format)

    def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        return self._iter(text)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
