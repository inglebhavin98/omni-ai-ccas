"""The end-to-end round-trip gate (CLAUDE.md Rule 3).

What this proves and what it does not.

**Proves:** the turn loop accounts for every stage, the ledger arithmetic is right, and
the orchestration the platform actually owns -- redaction, routing, policies, tool
dispatch, grounding -- fits inside the budget with the vendor stages held at their
budgeted values.

**Does not prove:** that Deepgram, Cartesia or a real LLM meet their slices. Those are
measured against live services, and until then the numbers here are a floor rather than
a forecast (ADR-0001).

So the vendor stages are *injected* at their budgeted cost rather than stubbed at zero.
A gate that ran every stage at 0 ms would pass forever and mean nothing.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from ccas.config.budget import load_budget
from ccas.schemas.session import LatencyLedger
from tests.voice_harness import build_call, utterances_for

#: Asserts the voice budget specifically -- frozen with it (ADR-0018).
pytestmark = pytest.mark.voice

REPO = Path(__file__).resolve().parents[2]
BUDGET = load_budget(REPO / "configs" / "latency_budget.yaml")

pytestmark = pytest.mark.latency

SCRIPT = ["where is my delivery", "ORD-884210"]


def _with_vendor_costs(ledger: LatencyLedger) -> LatencyLedger:
    """Charge the stages this suite cannot measure at their budgeted cost.

    STT, TTS and the LLM run against scripted stand-ins here, so they are free. Leaving
    them free would let the platform's own cost grow unnoticed until it is the whole
    budget.
    """
    stages = BUDGET.stages
    return ledger.model_copy(
        update={
            "stt_ms": max(ledger.stt_ms, stages.stt_ms),
            "llm_ttft_ms": max(ledger.llm_ttft_ms, stages.llm_ttft_ms),
            "tts_ttfb_ms": max(ledger.tts_ttfb_ms, stages.tts_ttfb_ms),
        }
    )


async def _measured_ledgers(rounds: int = 6) -> list[LatencyLedger]:
    ledgers: list[LatencyLedger] = []
    for index in range(rounds):
        harness = build_call(SCRIPT, session_id=f"rtt-{index}")
        await harness.run()
        ledgers.extend(record.ledger for record in harness.session.turns)
    assert ledgers, "no turns were measured"
    return ledgers


async def test_a_turn_fits_the_budget_with_vendors_at_full_cost() -> None:
    ledgers = [_with_vendor_costs(ledger) for ledger in await _measured_ledgers()]
    rtts = sorted(ledger.total_rtt_ms for ledger in ledgers)
    p95 = rtts[min(len(rtts) - 1, int(len(rtts) * 0.95))]
    assert p95 <= BUDGET.budget_ms, (
        f"p95 round trip {p95}ms exceeds the {BUDGET.budget_ms}ms ceiling "
        f"(median {statistics.median(rtts):.0f}ms)"
    )


async def test_the_platforms_own_stages_are_a_small_share_of_the_budget() -> None:
    """Routing, policies, tools and redaction are what we control.

    If they ever approach the vendor stages, the budget stops being about the network
    and starts being about us.
    """
    ledgers = await _measured_ledgers()
    ours = [
        ledger.router_ms + ledger.tool_ms + round(ledger.redact_us / 1000) for ledger in ledgers
    ]
    assert max(ours) < BUDGET.budget_ms / 4, (
        f"platform-owned stages reached {max(ours)}ms of a {BUDGET.budget_ms}ms budget"
    )


async def test_end_of_speech_detection_is_charged_to_the_turn() -> None:
    """A loop that forgot to charge VAD would look 100ms faster than it is."""
    for ledger in await _measured_ledgers(rounds=2):
        assert ledger.vad_ms > 0


async def test_every_stage_has_somewhere_to_be_recorded() -> None:
    """A stage with no field is a stage nobody will notice growing."""
    ledger = (await _measured_ledgers(rounds=1))[0]
    assert set(ledger.stages()) == set(BUDGET.stages.as_dict())


async def test_a_keypad_turn_skips_the_vendor_stages_entirely() -> None:
    """The point of the DTMF fallback: no VAD wait, no STT, no transcription risk."""
    from ccas.voice.drivers.scripted import ScriptedUtterance

    harness = build_call(
        ["where is my delivery"],
        [
            ScriptedUtterance(speech_ms=700, trailing_silence_ms=300),
            ScriptedUtterance(speech_ms=40, trailing_silence_ms=300, dtmf="884210#"),
        ],
    )
    await harness.run()
    keypad = [r for r in harness.session.turns if r.from_dtmf]
    assert keypad, "the fixture should produce a keypad turn"
    assert keypad[0].ledger.vad_ms == 0
    assert keypad[0].ledger.stt_ms == 0


async def test_no_turn_reports_a_breach_on_the_mock_driver() -> None:
    for ledger in await _measured_ledgers():
        assert not ledger.breached


async def test_the_budget_ceiling_is_carried_on_every_turn() -> None:
    for ledger in await _measured_ledgers(rounds=2):
        assert ledger.budget_ms == BUDGET.budget_ms


async def test_slow_vendors_are_reported_as_a_breach() -> None:
    """The gate must be able to fail. A ledger that never breaches measures nothing."""
    harness = build_call(SCRIPT, utterances_for(2))
    await harness.run()
    ledger = harness.session.turns[0].ledger
    overrun = ledger.model_copy(update={"llm_ttft_ms": BUDGET.budget_ms + 1})
    assert overrun.breached
