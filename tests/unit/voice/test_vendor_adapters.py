"""Deepgram and Cartesia adapters.

Neither has met its live service. These tests pin the wire formats and the re-chunking
so the parts we *can* verify are verified, and so a protocol change shows up as a test
failure rather than a silent call that never transcribes.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ccas.voice.audio import DEFAULT_FORMAT
from ccas.voice.stt.deepgram import DeepgramStt, parse_message, query_params
from ccas.voice.tts.cartesia import CartesiaTts

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice


def dg(transcript: str, *, speech_final: bool = False, is_final: bool = False) -> str:
    return json.dumps(
        {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.97}]},
            "is_final": is_final,
            "speech_final": speech_final,
            "start": 1.5,
        }
    )


# ------------------------------------------------------------------- deepgram


def test_a_partial_is_not_final() -> None:
    result = parse_message(dg("where is my"))
    assert result is not None
    assert not result.is_final
    assert result.text == "where is my"


def test_speech_final_ends_the_utterance() -> None:
    """`speech_final` is the endpointer's verdict; `is_final` only means no revision."""
    result = parse_message(dg("where is my delivery", speech_final=True))
    assert result is not None
    assert result.is_final


def test_timings_are_converted_to_milliseconds() -> None:
    result = parse_message(dg("hello", speech_final=True))
    assert result is not None
    assert result.at_ms == 1500


def test_an_empty_transcript_is_not_an_event() -> None:
    assert parse_message(dg("   ")) is None
    assert parse_message(dg("")) is None


def test_metadata_messages_are_ignored() -> None:
    assert parse_message(json.dumps({"type": "Metadata", "request_id": "x"})) is None


@pytest.mark.parametrize("junk", ["not json", b"\x00\x01", "", "[]"])
def test_malformed_messages_do_not_crash_the_stream(junk: str | bytes) -> None:
    assert parse_message(junk) is None


def test_the_socket_is_configured_for_our_audio_format() -> None:
    params = query_params("nova-3", DEFAULT_FORMAT, "en", 100)
    assert "encoding=linear16" in params
    assert f"sample_rate={DEFAULT_FORMAT.sample_rate}" in params
    assert "interim_results=true" in params, "the turn loop needs partials"
    assert "endpointing=100" in params


def test_the_adapter_is_unavailable_without_a_key() -> None:
    assert not DeepgramStt().available
    assert DeepgramStt(api_key="k").available


def test_the_url_names_the_model() -> None:
    assert "model=nova-3" in DeepgramStt(api_key="k").url


# ------------------------------------------------------------------- cartesia


def tts(handler, **kw: object) -> CartesiaTts:
    return CartesiaTts(
        api_key="k",
        voice_id="v",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kw,  # type: ignore[arg-type]
    )


def test_the_request_asks_for_raw_pcm() -> None:
    """A container format would have to be decoded before it could be played."""
    body = CartesiaTts(api_key="k", voice_id="v").request_body("hello")
    output = body["output_format"]
    assert isinstance(output, dict)
    assert output["encoding"] == "pcm_s16le"
    assert output["container"] == "raw"
    assert output["sample_rate"] == DEFAULT_FORMAT.sample_rate


def test_the_adapter_is_unavailable_without_a_key_or_voice() -> None:
    assert not CartesiaTts().available
    assert not CartesiaTts(api_key="k").available
    assert CartesiaTts(api_key="k", voice_id="v").available


async def test_audio_is_rechunked_to_a_uniform_frame_size() -> None:
    """A transport expects uniform frames; the vendor's chunking is its encoder's business."""
    chunk_ms = 40
    payload = b"\x01\x02" * (DEFAULT_FORMAT.frame_bytes(chunk_ms * 3) // 2)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    frames = [f async for f in tts(handler, chunk_ms=chunk_ms).synthesize("hi")]
    assert len(frames) == 3
    assert all(f.duration_ms == chunk_ms for f in frames)
    assert [f.timestamp_ms for f in frames] == [0, 40, 80]


async def test_a_trailing_partial_chunk_is_still_emitted() -> None:
    chunk_ms = 40
    payload = b"\x01\x02" * (DEFAULT_FORMAT.frame_bytes(chunk_ms) // 2 + 10)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    frames = [f async for f in tts(handler, chunk_ms=chunk_ms).synthesize("hi")]
    assert len(frames) == 2
    assert frames[-1].duration_ms < chunk_ms


async def test_cancelling_stops_generation() -> None:
    """The barge-in half of the contract."""
    payload = b"\x01\x02" * 40_000

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    engine = tts(handler)
    produced = 0
    async for _ in engine.synthesize("a long sentence"):
        produced += 1
        if produced == 2:
            await engine.cancel()
    assert produced == 2


async def test_an_error_status_is_raised_not_silently_empty() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with pytest.raises(RuntimeError, match="cartesia returned 429"):
        async for _ in tts(handler).synthesize("hi"):
            pass


async def test_synthesising_without_credentials_is_refused() -> None:
    with pytest.raises(RuntimeError, match="unset"):
        async for _ in CartesiaTts().synthesize("hi"):
            pass
