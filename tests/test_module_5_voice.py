"""Module gate: M5 — the real-time voice engine.

Granular tests live in ``tests/unit/voice/``; the budget gate in
``tests/latency/test_e2e_rtt.py``. This asserts the module works as a whole and that the
three things Phase 5 exists to deliver actually happen: a turn completes, the caller can
interrupt, and every stage is accounted for.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ccas.config.budget import load_budget
from ccas.config.settings import Settings
from ccas.schemas.common import Speaker
from ccas.voice.audio import tone
from ccas.voice.drivers.scripted import ScriptedUtterance
from ccas.voice.transport import AudioTransport
from ccas.voice.worker import preflight
from tests.voice_harness import build_call, utterances_for

#: Frozen channel -- excluded from the default run (ADR-0018).
pytestmark = pytest.mark.voice

REPO = Path(__file__).resolve().parents[1]
BUDGET = load_budget(REPO / "configs" / "latency_budget.yaml")


async def test_a_voice_call_completes_end_to_end() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    state = await harness.run()
    assert state.escalation is None
    assert any("on its way" in t.content.text for t in state.turns if t.speaker is Speaker.BOT)
    assert harness.transport.played_ms > 0


async def test_the_caller_can_interrupt() -> None:
    """The behaviour this module exists to get right."""
    harness = build_call(["where is my delivery"], utterances_for(1), pace_factor=0.05)
    session = harness.session
    await session._open()
    for _ in range(2000):
        if session._played_ms > session._bargein.min_played_ms:
            break
        await asyncio.sleep(0.001)
    for index in range(10):
        await session._on_frame(tone(20, at_ms=index * 20))
    await session._finish_speaking()
    assert harness.transport.clears == 1


async def test_every_turn_is_accounted_for_against_the_budget() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    await harness.run()
    assert harness.session.turns
    for record in harness.session.turns:
        assert set(record.ledger.stages()) == set(BUDGET.stages.as_dict())
        assert record.ledger.budget_ms == BUDGET.budget_ms


async def test_the_keypad_is_a_real_fallback() -> None:
    """Speech recognition fails on noisy lines and unfamiliar accents."""
    harness = build_call(
        ["where is my delivery"],
        [
            ScriptedUtterance(speech_ms=700, trailing_silence_ms=300),
            ScriptedUtterance(speech_ms=40, trailing_silence_ms=300, dtmf="884210#"),
        ],
    )
    state = await harness.run()
    assert any(r.from_dtmf for r in harness.session.turns)
    assert "order_reference" in state.slots


async def test_keypad_input_is_redacted_like_speech() -> None:
    """The fallback must not be the leak."""
    harness = build_call(
        ["where is my delivery"],
        [
            ScriptedUtterance(speech_ms=700, trailing_silence_ms=300),
            ScriptedUtterance(speech_ms=40, trailing_silence_ms=300, dtmf="884210#"),
        ],
    )
    state = await harness.run()
    assert "884210" not in state.model_dump_json()


async def test_nothing_unredacted_reaches_the_graph_over_voice() -> None:
    harness = build_call(["my card is 4111 1111 1111 1111"])
    state = await harness.run()
    rendered = state.model_dump_json()
    assert "4111 1111 1111 1111" not in rendered
    assert "[PAYMENT_CARD_1]" in rendered


async def test_the_tool_still_gets_the_real_value() -> None:
    harness = build_call(["where is my delivery", "ORD-884210"])
    await harness.run()
    assert harness.ctx.vault.resolve("[ACCOUNT_REF_1]") == "ORD-884210"


def test_the_transport_contract_admits_more_than_one_driver() -> None:
    """ADR-0001 defers SIP behind this interface; that only works if it is abstract."""
    from ccas.voice.drivers.livekit import LiveKitTransport
    from ccas.voice.drivers.scripted import ScriptedTransport

    assert issubclass(ScriptedTransport, AudioTransport)
    assert issubclass(LiveKitTransport, AudioTransport)
    for method in ("receive", "send", "clear", "dtmf"):
        assert hasattr(AudioTransport, method)


def test_the_worker_refuses_to_answer_calls_when_unready() -> None:
    """Rule 2: a worker that degraded past redaction would be worse than a silent one.

    Preflight must object with no vendor credentials configured, whatever else is ready.
    It used to be asserted via the missing-taxonomy complaint; `retail` acquired an
    adopted taxonomy in ADR-0020, so that particular objection no longer fires and the
    test now checks the thing it was always about.
    """
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        domains_dir=REPO / "domains",
        config_dir=REPO / "configs",
        redaction_policy=REPO / "configs" / "redaction_policy.yaml",
    )
    problems = preflight(settings, "retail")
    assert problems, "with no credentials configured, preflight must object"
    assert any(
        "deepgram" in p.lower() or "cartesia" in p.lower() or "livekit" in p.lower()
        for p in problems
    ), f"preflight should name the missing vendors; got {problems}"


def test_the_worker_refuses_a_domain_with_no_taxonomy() -> None:
    """The router would have nothing to classify into, so answering would be dishonest."""
    domains = REPO / "domains"
    without = [
        d.name
        for d in sorted(domains.iterdir())
        if (d / "pack.yaml").is_file() and not (d / "taxonomy.json").is_file()
    ]
    if not without:
        pytest.skip("every pack has a taxonomy")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        domains_dir=domains,
        config_dir=REPO / "configs",
        redaction_policy=REPO / "configs" / "redaction_policy.yaml",
    )
    assert any("taxonomy" in p for p in preflight(settings, without[0]))


@pytest.mark.parametrize("name", ["deepgram", "cartesia", "livekit"])
def test_vendor_adapters_report_unavailability_rather_than_failing_silently(
    name: str,
) -> None:
    from ccas.voice.stt.deepgram import DeepgramStt
    from ccas.voice.tts.cartesia import CartesiaTts

    if name == "deepgram":
        assert not DeepgramStt().available
    elif name == "cartesia":
        assert not CartesiaTts().available
    else:
        from ccas.voice.drivers.livekit import LiveKitTransport

        assert LiveKitTransport(room=object()).info.driver == "livekit"
