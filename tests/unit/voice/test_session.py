from __future__ import annotations

import asyncio

import pytest

from ccas.schemas.common import Speaker, VerificationLevel
from ccas.voice.audio import tone
from ccas.voice.drivers.scripted import ScriptedUtterance
from tests.voice_harness import build_call, utterances_for

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice


def spoken(state) -> list[str]:
    return [t.content.text for t in state.turns if t.speaker is Speaker.BOT]


async def test_the_call_opens_before_the_caller_speaks() -> None:
    harness = build_call([])
    state = await harness.run()
    assert spoken(state)
    assert "recorded" in spoken(state)[0].lower()


async def test_a_contained_call_runs_end_to_end() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    state = await harness.run()
    assert state.escalation is None
    assert any("on its way" in text for text in spoken(state))
    assert len(harness.session.turns) == 2


async def test_the_transcript_is_redacted_before_the_graph_sees_it() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    state = await harness.run()
    rendered = state.model_dump_json()
    assert "ORD-884210" not in rendered
    assert "[ACCOUNT_REF_1]" in rendered


async def test_the_tool_still_receives_the_real_reference() -> None:
    """The vault at work across the whole stack (ADR-0010)."""
    harness = build_call(["where is my delivery", "ORD-884210"])
    await harness.run()
    assert harness.ctx.vault.resolve("[ACCOUNT_REF_1]") == "ORD-884210"


async def test_audio_is_played_back_for_every_bot_turn() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    state = await harness.run()
    assert harness.transport.played_ms > 0
    assert len(harness.session.turns) + 1 >= len(spoken(state)) - 1


async def test_every_turn_carries_a_latency_ledger() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    await harness.run()
    for record in harness.session.turns:
        assert record.ledger.budget_ms == 800
        assert record.ledger.vad_ms > 0, "end-of-speech detection is never free"


async def test_the_ledger_survives_the_round_trip_through_the_graph() -> None:
    """The voice side seeds it, the graph adds to it, the voice side finishes it."""
    harness = build_call(["where is my delivery", "ORD-884210"])
    await harness.run()
    stages = harness.session.turns[0].ledger.stages()
    assert stages["vad_ms"] > 0  # written by the voice loop
    assert "router_ms" in stages  # written by the graph
    assert "tts_ttfb_ms" in stages  # written back by the voice loop


async def test_a_regulated_intent_escalates_over_voice_too() -> None:
    harness = build_call(["I want to raise a dispute about a charge"])
    state = await harness.run()
    assert state.escalation is not None
    assert state.escalation.triggered_by == "risk"


async def _interrupt_during_playback(barge_in: bool) -> tuple[int, int]:
    """Start a call, wait until the bot is genuinely past the guard, then talk over it.

    Waiting on the *condition* rather than on the scheduler is what makes this
    deterministic. Racing a 250 ms guard against asyncio's ordering produced a test that
    passed or failed depending on machine load.
    """
    harness = build_call(
        ["where is my delivery"], utterances_for(1), pace_factor=0.05, barge_in=barge_in
    )
    session = harness.session
    await session._open()

    guard = session._bargein.min_played_ms
    for _ in range(2000):
        if session._played_ms > guard:
            break
        await asyncio.sleep(0.001)
    assert session._played_ms > guard, "playback never got past the barge-in guard"

    clock = 0
    for _ in range(10):
        await session._on_frame(tone(20, at_ms=clock))
        clock += 20
    await session._finish_speaking()
    return harness.transport.clears, session._bargein.count


async def test_talking_over_the_bot_clears_playback() -> None:
    clears, count = await _interrupt_during_playback(barge_in=True)
    assert clears == 1
    assert count == 1


async def test_barge_in_can_be_disabled() -> None:
    clears, count = await _interrupt_during_playback(barge_in=False)
    assert clears == 0
    assert count == 0


async def test_the_call_does_not_end_mid_sentence() -> None:
    """Whatever the transport does, the bot finishes what it started saying."""
    harness = build_call(["where is my delivery", "ORD-884210"], pace_factor=0.0)
    await harness.run()
    assert harness.session._playback is None


async def test_keypad_input_answers_a_dtmf_capturable_slot() -> None:
    """Speech recognition fails on noisy lines; the keypad is the fallback."""
    harness = build_call(
        ["where is my delivery"],
        [
            ScriptedUtterance(speech_ms=700, trailing_silence_ms=300),
            ScriptedUtterance(speech_ms=40, trailing_silence_ms=300, dtmf="884210#"),
        ],
    )
    state = await harness.run()
    assert any(record.from_dtmf for record in harness.session.turns)
    assert "order_reference" in state.slots


async def test_a_keypad_answer_is_redacted_wholesale() -> None:
    """Six bare digits match no pattern.

    Leaving keypad input to the scanner put the caller's reference straight into the
    transcript -- found by an end-to-end run, not by a unit test. A keypad-entered
    reference is exactly as sensitive as a spoken one.
    """
    harness = build_call(
        ["where is my delivery"],
        [
            ScriptedUtterance(speech_ms=700, trailing_silence_ms=300),
            ScriptedUtterance(speech_ms=40, trailing_silence_ms=300, dtmf="884210#"),
        ],
    )
    state = await harness.run()
    assert "884210" not in state.model_dump_json()
    assert "[ACCOUNT_REF_1]" in state.model_dump_json()
    # ...and the tool can still act on it.
    assert harness.ctx.vault.resolve("[ACCOUNT_REF_1]") == "884210"


async def test_a_short_keypad_answer_is_left_readable() -> None:
    """A single digit is a menu choice, not an identifier. Over-redaction helps nobody."""
    from ccas.schemas.taxonomy import SlotSpec, SlotType
    from ccas.voice.dtmf import DtmfEntry
    from ccas.voice.session import _entity_for

    plain = SlotSpec(name="choice", slot_type=SlotType.STRING, elicitation_prompt="?")
    assert _entity_for(plain, DtmfEntry("2", 0, True)) is None
    assert _entity_for(plain, DtmfEntry("8842", 0, True)) is not None


async def test_a_call_without_a_taxonomy_still_completes() -> None:
    harness = build_call(
        ["I need help"], domain="healthcare", verification=VerificationLevel.STRONG
    )
    state = await harness.run()
    assert state.escalation is not None
