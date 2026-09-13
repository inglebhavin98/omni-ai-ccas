from __future__ import annotations

import asyncio

import pytest

from ccas.voice.bargein import BargeInController
from ccas.voice.drivers.scripted import ScriptedTransport
from ccas.voice.tts.scripted import ScriptedTts

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice


def controller(**kw: object) -> BargeInController:
    return BargeInController(ScriptedTransport(), ScriptedTts(), **kw)  # type: ignore[arg-type]


async def test_an_interruption_clears_playback_and_cancels_generation() -> None:
    ctrl = controller()
    await ctrl.transport.send(
        next(iter([__import__("ccas.voice.audio", fromlist=["tone"]).tone(20)]))
    )
    result = await ctrl.on_caller_speech(played_ms=1000, correlation_id="c")
    assert result.interrupted
    assert ctrl.transport.clears == 1  # type: ignore[attr-defined]
    assert ctrl.tts.cancelled  # type: ignore[attr-defined]
    assert ctrl.count == 1


async def test_a_brief_overlap_is_not_an_interruption() -> None:
    """Callers say "yeah" over an opening word without meaning to interrupt."""
    ctrl = controller(min_played_ms=250)
    result = await ctrl.on_caller_speech(played_ms=100, correlation_id="c")
    assert not result.interrupted
    assert ctrl.transport.clears == 0  # type: ignore[attr-defined]
    assert ctrl.count == 0


async def test_barge_in_can_be_disabled() -> None:
    ctrl = controller(enabled=False)
    assert not (await ctrl.on_caller_speech(9999, "c")).interrupted


async def test_the_in_flight_turn_is_cancelled() -> None:
    """Reasoning about a question the caller has abandoned is wasted work."""

    async def long_turn() -> str:
        await asyncio.sleep(5)
        return "done"

    ctrl = controller()
    turn = asyncio.create_task(long_turn())
    speaking = asyncio.create_task(asyncio.sleep(5))
    ctrl.begin(speaking=speaking, turn=turn)

    result = await ctrl.on_caller_speech(1000, "c")
    await asyncio.sleep(0)
    assert result.cancelled_turn
    assert turn.cancelled() or turn.cancelling()
    assert speaking.cancelled() or speaking.cancelling()


async def test_an_already_finished_task_is_not_cancelled() -> None:
    ctrl = controller()
    done: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
    await done
    ctrl.begin(speaking=done)
    result = await ctrl.on_caller_speech(1000, "c")
    assert result.interrupted
    assert not done.cancelled()
