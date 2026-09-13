"""Audio transport interface.

One protocol, several drivers: LiveKit WebRTC in deployment, a deterministic scripted
driver in CI. ADR-0001 deferred SIP behind this same interface, which is why ``receive``
and ``send`` say nothing about how the audio arrived.

``clear`` is the one method that exists solely for barge-in: on detected speech the
playback queue must be dropped *now*, not after the current sentence finishes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from ccas.voice.audio import DEFAULT_FORMAT, AudioFormat, AudioFrame

__all__ = ["AudioTransport", "DtmfSignal", "TransportInfo"]


@dataclass(frozen=True, slots=True)
class DtmfSignal:
    """A touch-tone keypress as the gateway reports it (RFC 2833 on SIP)."""

    digit: str
    at_ms: int

    def __post_init__(self) -> None:
        if len(self.digit) != 1 or self.digit not in "0123456789*#ABCD":
            raise ValueError(f"not a DTMF digit: {self.digit!r}")


@dataclass(frozen=True, slots=True)
class TransportInfo:
    driver: str
    room: str | None = None
    participant: str | None = None
    format: AudioFormat = DEFAULT_FORMAT


class AudioTransport(ABC):
    """Bidirectional audio plus out-of-band DTMF."""

    @property
    @abstractmethod
    def info(self) -> TransportInfo: ...

    @abstractmethod
    def receive(self) -> AsyncIterator[AudioFrame]:
        """Inbound caller audio, in arrival order. Ends when the call does."""

    @abstractmethod
    async def send(self, frame: AudioFrame) -> None:
        """Queue a frame for playback."""

    @abstractmethod
    async def clear(self) -> None:
        """Drop everything queued for playback.

        The barge-in primitive. A transport that cannot do this cannot support
        interruption, and a caller talking over a bot that keeps speaking is the single
        most complained-about behaviour of the systems this replaces.
        """

    @abstractmethod
    def dtmf(self) -> AsyncIterator[DtmfSignal]:
        """Keypresses, if the transport carries them. May yield nothing."""

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
