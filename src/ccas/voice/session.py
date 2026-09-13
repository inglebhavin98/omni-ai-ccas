"""The voice turn loop.

Ties the whole platform together for one call:

    audio -> VAD -> STT -> redact -> graph -> TTS -> audio

Every stage writes into one ``LatencyLedger``, which is what turns the 800 ms budget
from a table in a config file into a measurement on every turn (CLAUDE.md Rule 3).

Two things make this harder than the diagram suggests. Playback runs concurrently so
that barge-in can land mid-sentence. And the ledger has to survive a round trip through
the graph, which writes its own slices -- so the voice side seeds it, the graph adds to
it, and the voice side finishes it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from ccas.config.budget import LatencyBudget
from ccas.graph.context import GraphContext
from ccas.observability.logging import get_logger
from ccas.schemas.common import Speaker
from ccas.schemas.pii import PiiEntityType
from ccas.schemas.session import LatencyLedger, SessionState, SlotValue, Turn
from ccas.schemas.taxonomy import SlotSpec
from ccas.voice.audio import AudioFrame
from ccas.voice.bargein import BargeInController
from ccas.voice.dtmf import DtmfCollector, DtmfEntry
from ccas.voice.stt.base import SpeechToText, Transcript
from ccas.voice.stt.scripted import ScriptedStt
from ccas.voice.transport import AudioTransport
from ccas.voice.tts.base import TextToSpeech
from ccas.voice.vad import SpeechEvent, VoiceActivityDetector

__all__ = ["TurnRecord", "VoiceSession"]

LOG = get_logger("voice.session")


@dataclass(slots=True)
class TurnRecord:
    """What happened in one turn. The unit the latency bench reports on."""

    index: int
    ledger: LatencyLedger
    spoken: list[str] = field(default_factory=list)
    interrupted: bool = False
    from_dtmf: bool = False

    @property
    def rtt_ms(self) -> int:
        return self.ledger.total_rtt_ms

    @property
    def breached(self) -> bool:
        return self.ledger.breached


class VoiceSession:
    """One call. Not reusable -- a second call gets a new session and a new vault."""

    def __init__(
        self,
        transport: AudioTransport,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        graph: Any,
        ctx: GraphContext,
        state: SessionState,
        config: dict[str, Any],
        budget: LatencyBudget,
        barge_in: bool = True,
    ) -> None:
        self.transport = transport
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.graph = graph
        self.ctx = ctx
        self.config = config
        self.budget = budget
        self.state: dict[str, Any] = {}
        self._initial = state
        self._dtmf = DtmfCollector()
        self._bargein = BargeInController(transport, tts, enabled=barge_in)
        self.turns: list[TurnRecord] = []
        self._speaking = False
        self._played_ms = 0
        self._playback: asyncio.Task[None] | None = None
        self._queued_utterances = 0

    # ---------------------------------------------------------------- running

    async def run(self) -> SessionState:
        """Drive the call to completion. Returns the final state."""
        await self._open()
        async for frame in self.transport.receive():
            if self._terminal:
                break
            await self._on_frame(frame)
            await self._drain_dtmf(frame.timestamp_ms)
        await self._finish_speaking()
        await self._close()
        return self.snapshot()

    def snapshot(self) -> SessionState:
        return SessionState.model_validate(self.state or self._initial.model_dump())

    @property
    def _terminal(self) -> bool:
        return bool(self.state.get("terminal"))

    async def _open(self) -> None:
        """First invocation: the greeting, before the caller has said anything."""
        self.state = await self.graph.ainvoke(self._initial, self.config)
        await self._speak_pending(TurnRecord(index=0, ledger=self.budget.new_ledger()))

    async def _close(self) -> None:
        await self.tts.aclose()
        await self.stt.aclose()
        await self.transport.close()

    # ----------------------------------------------------------------- frames

    async def _on_frame(self, frame: AudioFrame) -> None:
        started = time.perf_counter_ns()
        decision = self.vad.push(frame)

        if decision.event is SpeechEvent.SPEECH_STARTED and self._speaking:
            result = await self._bargein.on_caller_speech(
                self._played_ms, self._initial.trace.correlation_id
            )
            if result.interrupted:
                await self._finish_speaking()
                self._speaking = False
            return

        if decision.event is not SpeechEvent.SPEECH_ENDED:
            return

        ledger = self.budget.new_ledger()
        # End-of-speech detection cost: the hangover we waited out plus this decision.
        ledger.vad_ms = decision.silence_duration_ms + _ms_since(started)
        await self._handle_utterance(ledger)

    async def _handle_utterance(self, ledger: LatencyLedger) -> None:
        transcript = await self._transcribe()
        if transcript is None:
            return
        ledger.stt_ms = transcript.latency_ms or 1

        record = TurnRecord(index=len(self.turns) + 1, ledger=ledger)
        await self._run_turn(transcript.text, record)

    async def _transcribe(self) -> Transcript | None:
        """Pull one final transcript.

        The scripted STT is pull-based so a test can drive turns exactly; a streaming
        engine is consumed by the LiveKit worker, which owns its socket.
        """
        if isinstance(self.stt, ScriptedStt):
            return self.stt.next_final()
        return None

    # ------------------------------------------------------------------- turn

    async def _run_turn(
        self,
        text: str,
        record: TurnRecord,
        captured_via: str = "speech",
        as_entity: PiiEntityType | None = None,
    ) -> None:
        started = time.perf_counter_ns()
        allocator = self.ctx.redaction.new_allocator(self.ctx.vault)
        # A keypad answer to a PII slot is replaced wholesale: six bare digits match no
        # pattern, so scanning would put the caller's reference into the transcript.
        content = (
            self.ctx.redaction.redact_value(text, as_entity, allocator)
            if as_entity is not None
            else self.ctx.redaction.redact(text, allocator)
        )
        record.ledger.redact_us = content.report.elapsed_us

        if not content.egress_permitted:
            # Rule 2: a turn we cannot clean does not reach a model. Fail the turn.
            LOG.warning(
                "voice.turn.blocked",
                correlation_id=self._initial.trace.correlation_id,
                status=content.report.status.value,
                residual=list(content.report.residual_patterns),
            )
            return

        index = int(self.state.get("turn_index", 0))
        turn = Turn(index=index, speaker=Speaker.CALLER, content=content)
        self.state = await self.graph.ainvoke(
            {"turns": [turn], "turn_index": index + 1, "latency": record.ledger},
            self.config,
        )
        # The graph adds router_ms, tool_ms and llm_ttft_ms on top of what we seeded.
        record.ledger = self.snapshot().latency

        await self._speak_pending(record)
        record.from_dtmf = captured_via == "dtmf"
        self.turns.append(record)

        LOG.info(
            "voice.turn",
            correlation_id=self._initial.trace.correlation_id,
            turn=record.index,
            captured_via=captured_via,
            rtt_ms=record.rtt_ms,
            breached=record.breached,
            stages=record.ledger.stages(),
            redacted_entities=content.report.entity_counts,
            wall_ms=_ms_since(started),
        )

    # ---------------------------------------------------------------- speaking

    async def _speak_pending(self, record: TurnRecord) -> None:
        """Start playing what the graph just said, and return immediately.

        Playback runs as a task so the frame loop keeps consuming caller audio while the
        bot talks. Awaiting it here instead -- which is the obvious way to write this --
        makes barge-in structurally impossible: the loop cannot see the caller start
        speaking, because it is blocked on the speaking.
        """
        pending = self._unspoken()
        if not pending:
            return
        await self._finish_speaking()
        self._queued_utterances += len(pending)
        self._playback = asyncio.create_task(self._play(pending, record))
        self._bargein.begin(speaking=self._playback)

    async def _finish_speaking(self) -> None:
        """Wait for any in-flight playback. Cancellation is barge-in, not an error."""
        task = self._playback
        self._playback = None
        if task is None or task.done():
            return
        # Cancellation here is barge-in, which is a normal outcome, not an error.
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _play(self, texts: list[str], record: TurnRecord) -> None:
        for text in texts:
            record.spoken.append(text)
            self._speaking = True
            self._played_ms = 0
            started = time.perf_counter_ns()
            first = True
            try:
                async for chunk in self.tts.synthesize(text):
                    if first:
                        record.ledger.tts_ttfb_ms = _ms_since(started)
                        first = False
                    await self.transport.send(chunk)
                    self._played_ms += chunk.duration_ms
            except asyncio.CancelledError:
                record.interrupted = True
                raise
            finally:
                self._speaking = False

    def _unspoken(self) -> list[str]:
        """Bot turns the graph produced that have not been queued for playback yet.

        Counted rather than derived from the turn records: a barge-in truncates playback
        without un-queueing it, so "what did we play" and "what did we decide to say"
        are different questions.
        """
        bot_turns = [t.content.text for t in self.snapshot().turns if t.speaker is Speaker.BOT]
        return bot_turns[self._queued_utterances :]

    # ------------------------------------------------------------------- dtmf

    async def _drain_dtmf(self, now_ms: int) -> None:
        async for signal in self.transport.dtmf():
            entry = self._dtmf.push(signal)
            if entry is not None:
                await self._on_dtmf_entry(entry)
        expired = self._dtmf.expire(now_ms)
        if expired is not None:
            await self._on_dtmf_entry(expired)

    async def _on_dtmf_entry(self, entry: DtmfEntry) -> None:
        """Treat a completed keypad entry as the answer to the outstanding slot.

        Only for slots the taxonomy marks ``dtmf_capturable`` -- typing a reference is
        normal, typing a reason for return is not.
        """
        state = self.snapshot()
        pending = state.pending_slot
        node = self.ctx.node_for(state)
        spec = next((s for s in (node.slots if node else ()) if s.name == pending), None)
        if spec is None or not spec.dtmf_capturable:
            LOG.info(
                "voice.dtmf.ignored",
                correlation_id=self._initial.trace.correlation_id,
                digits=len(entry.digits),
                pending_slot=pending,
            )
            return

        ledger = self.budget.new_ledger()
        # Keypad input skips VAD and STT entirely -- that is the point of the fallback.
        ledger.vad_ms = 0
        ledger.stt_ms = 0
        record = TurnRecord(index=len(self.turns) + 1, ledger=ledger)
        await self._run_turn(
            entry.digits, record, captured_via="dtmf", as_entity=_entity_for(spec, entry)
        )

    def dtmf_slot_value(self, entry: DtmfEntry, name: str) -> SlotValue:
        """Build a slot from a keypad entry. The digits are redacted on the way in."""
        content = self.ctx.redaction.redact(
            entry.digits, self.ctx.redaction.new_allocator(self.ctx.vault)
        )
        return SlotValue(name=name, raw=content, parsed=None, valid=True, captured_via="dtmf")


#: A keypad run this long is a reference, a PIN or a card, whatever the slot says.
DTMF_SENSITIVE_LENGTH = 4


def _entity_for(spec: SlotSpec, entry: DtmfEntry) -> PiiEntityType | None:
    """How to redact a keypad answer.

    The slot's declared ``pii_entity`` wins. Failing that, any run of four or more
    digits is treated as sensitive -- six bare digits match no pattern, so leaving it to
    the scanner would put the caller's reference straight into the transcript.
    """
    if spec.pii_entity is not None:
        return spec.pii_entity
    if len(entry.digits) >= DTMF_SENSITIVE_LENGTH:
        return PiiEntityType.ACCOUNT_REF
    return None


def _ms_since(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
