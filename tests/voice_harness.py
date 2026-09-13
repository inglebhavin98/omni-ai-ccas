"""Builds a complete voice call from stubs.

Everything below the transport is real -- VAD, redaction, the graph, the tool executor,
the policies, the latency ledger. Only the four things that would need a network are
scripted: audio, STT, TTS and the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccas.config.budget import LatencyBudget, load_budget
from ccas.config.domain_loader import LoadedDomain, load_pack
from ccas.graph.assembly import build_graph
from ccas.graph.checkpoint import memory_checkpointer, thread_config
from ccas.graph.context import GraphContext, build_context
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.llm.bindings import load_bindings
from ccas.schemas.common import Channel, TraceContext, VerificationLevel
from ccas.schemas.session import CallerContext, SessionState
from ccas.schemas.taxonomy import IntentTaxonomy
from ccas.voice.drivers.scripted import ScriptedTransport, ScriptedUtterance
from ccas.voice.session import VoiceSession
from ccas.voice.stt.scripted import ScriptedStt
from ccas.voice.tts.scripted import ScriptedTts
from ccas.voice.vad import EnergyVad
from tests.graph_stub import StubGraphProvider

__all__ = ["VoiceHarness", "build_call", "utterances_for"]

REPO = Path(__file__).resolve().parents[1]
TAXONOMY = IntentTaxonomy.model_validate(
    json.loads((REPO / "tests" / "fixtures" / "taxonomies" / "retail.json").read_text())
)


@dataclass(slots=True)
class VoiceHarness:
    session: VoiceSession
    transport: ScriptedTransport
    tts: ScriptedTts
    ctx: GraphContext
    budget: LatencyBudget

    async def run(self) -> SessionState:
        return await self.session.run()

    @property
    def rtts(self) -> list[int]:
        return [turn.rtt_ms for turn in self.session.turns]

    @property
    def barge_ins(self) -> int:
        return (self.session.turns and 0) or 0


def utterances_for(
    count: int, speech_ms: int = 700, silence_ms: int = 300, dtmf: str = ""
) -> list[ScriptedUtterance]:
    return [
        ScriptedUtterance(
            speech_ms=speech_ms,
            trailing_silence_ms=silence_ms,
            dtmf=dtmf if index == count - 1 else "",
        )
        for index in range(count)
    ]


def build_call(
    script: list[str],
    utterances: list[ScriptedUtterance] | None = None,
    *,
    session_id: str = "voice-test",
    verification: VerificationLevel = VerificationLevel.SOFT,
    barge_in: bool = True,
    pace_factor: float = 0.0,
    stt_latency_ms: int = 0,
    tts_first_chunk_ms: int = 0,
    grounded: bool = True,
    domain: str = "retail",
) -> VoiceHarness:
    pack = load_pack(REPO / "domains", domain)
    loaded = LoadedDomain(
        pack=pack,
        root=REPO / "domains" / domain,
        taxonomy=TAXONOMY if domain == "retail" else None,
    )
    provider = StubGraphProvider(grounded=grounded)
    bindings = load_bindings(REPO / "configs" / "models.yaml")
    ctx = build_context(loaded, provider, bindings, config_dir=REPO / "configs")
    graph = build_graph(
        ctx,
        IntentRouter(provider, bindings.resolve("router"), TAXONOMY)
        if loaded.has_taxonomy
        else None,
        Responder(provider, bindings.resolve("task_agent")),
        checkpointer=memory_checkpointer(),
    )

    trace = TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id=session_id)
    transport = ScriptedTransport(
        utterances=utterances or utterances_for(len(script)), pace_factor=pace_factor
    )
    tts = ScriptedTts(chunk_ms=40, first_chunk_delay_ms=tts_first_chunk_ms, pace_factor=pace_factor)
    budget = load_budget(REPO / "configs" / "latency_budget.yaml")
    state = SessionState(
        session_id=session_id,
        trace=trace,
        domain=domain,
        channel=Channel.VOICE,
        caller=CallerContext(caller_ref="a" * 32, verification=verification),
    )
    config: dict[str, Any] = thread_config(session_id, trace)
    session = VoiceSession(
        transport,
        EnergyVad(min_speech_ms=60, min_silence_ms=100),
        ScriptedStt(script, latency_ms=stt_latency_ms),
        tts,
        graph,
        ctx,
        state,
        config,
        budget,
        barge_in=barge_in,
    )
    return VoiceHarness(session=session, transport=transport, tts=tts, ctx=ctx, budget=budget)
