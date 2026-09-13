from __future__ import annotations

import pytest

from ccas.voice.audio import DEFAULT_FORMAT, AudioFrame, silence, tone
from ccas.voice.vad import EnergyVad, SpeechEvent, VadState

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice

FRAME_MS = 20


def test_frame_duration_is_derived_from_the_format() -> None:
    assert silence(20).duration_ms == 20
    assert silence(100).duration_ms == 100


def test_a_frame_never_renders_its_audio() -> None:
    """Raw audio has not been through STT, so it has not been through redaction."""
    frame = tone(20)
    rendered = f"{frame!r} {frame!s}"
    assert "\\x" not in rendered
    assert str(len(frame.data)) in rendered


def test_silence_is_quiet_and_tone_is_not() -> None:
    assert silence(20).rms() == 0.0
    assert tone(20).rms() > 0.1


def test_an_empty_frame_does_not_divide_by_zero() -> None:
    assert AudioFrame(b"", 0, DEFAULT_FORMAT).rms() == 0.0


def test_sample_count_matches_the_sample_width() -> None:
    frame = tone(100)
    assert frame.sample_count == len(frame.data) // 2
    assert frame.samples().shape == (frame.sample_count,)


# --------------------------------------------------------------------- vad


def feed(vad: EnergyVad, speech_ms: int, silence_ms: int, start: int = 0) -> list:
    """Push speech then silence, returning every non-NONE decision."""
    events = []
    clock = start
    for _ in range(speech_ms // FRAME_MS):
        d = vad.push(tone(FRAME_MS, at_ms=clock))
        clock += FRAME_MS
        if d.event is not SpeechEvent.NONE:
            events.append(d)
    for _ in range(silence_ms // FRAME_MS):
        d = vad.push(silence(FRAME_MS, at_ms=clock))
        clock += FRAME_MS
        if d.event is not SpeechEvent.NONE:
            events.append(d)
    return events


def test_an_utterance_produces_a_start_then_an_end() -> None:
    events = feed(EnergyVad(min_speech_ms=60, min_silence_ms=100), 400, 200)
    assert [e.event for e in events] == [SpeechEvent.SPEECH_STARTED, SpeechEvent.SPEECH_ENDED]


def test_silence_alone_produces_nothing() -> None:
    assert feed(EnergyVad(), 0, 500) == []


def test_a_brief_noise_is_not_speech() -> None:
    """A cough or a door closing is loud too."""
    vad = EnergyVad(min_speech_ms=100, min_silence_ms=100)
    events = feed(vad, 40, 300)
    assert events == []


def test_a_pause_for_breath_does_not_end_the_turn() -> None:
    """The single most important VAD tuning: too short and the bot cuts in."""
    vad = EnergyVad(min_speech_ms=60, min_silence_ms=200)
    events = feed(vad, 300, 100)  # pause shorter than the hangover
    assert [e.event for e in events] == [SpeechEvent.SPEECH_STARTED]
    events += feed(vad, 300, 300)  # keeps talking, then really stops
    assert events[-1].event is SpeechEvent.SPEECH_ENDED


def test_the_end_event_reports_how_long_they_spoke() -> None:
    events = feed(EnergyVad(min_speech_ms=60, min_silence_ms=100), 400, 200)
    ended = events[-1]
    assert ended.speech_duration_ms >= 300
    assert ended.silence_duration_ms >= 100


def test_the_detector_resets_between_utterances() -> None:
    vad = EnergyVad(min_speech_ms=60, min_silence_ms=100)
    feed(vad, 300, 200)
    assert vad.state is VadState.IDLE
    assert [e.event for e in feed(vad, 300, 200)] == [
        SpeechEvent.SPEECH_STARTED,
        SpeechEvent.SPEECH_ENDED,
    ]


@pytest.mark.parametrize("hangover", [60, 100, 200, 400])
def test_the_hangover_is_what_it_says(hangover: int) -> None:
    vad = EnergyVad(min_speech_ms=40, min_silence_ms=hangover)
    events = feed(vad, 200, hangover - FRAME_MS)
    assert all(e.event is not SpeechEvent.SPEECH_ENDED for e in events)
    events = feed(vad, 0, FRAME_MS * 2)
    assert events[-1].event is SpeechEvent.SPEECH_ENDED


def test_the_energy_detector_needs_no_model() -> None:
    assert EnergyVad().available


def test_silero_reports_unavailability_rather_than_pretending() -> None:
    from ccas.voice.vad import SileroVad

    vad = SileroVad()
    if vad.available:
        pytest.skip("silero is installed; the unavailable path is not reachable")
    assert vad.load_error is not None
    with pytest.raises(RuntimeError, match="silero VAD unavailable"):
        vad.push(tone(20))
