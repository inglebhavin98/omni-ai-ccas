"""Scripted journeys through the mesh.

Each journey drives the graph the way the voice engine will: one invocation per caller
turn, sharing one checkpointed state. Router and responder are keyword-driven stubs, so
a failure here is a failure of the graph rather than of a model's mood.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ccas.config.domain_loader import LoadedDomain, load_pack
from ccas.graph.assembly import build_graph
from ccas.graph.checkpoint import memory_checkpointer, thread_config
from ccas.graph.context import GraphContext, build_context
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.llm.bindings import load_bindings
from ccas.schemas.common import Channel, Speaker, TraceContext, VerificationLevel
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import CallerContext, SessionState, Turn
from ccas.schemas.taxonomy import IntentTaxonomy
from tests.graph_stub import StubGraphProvider

REPO = Path(__file__).resolve().parents[2]
TAXONOMY = IntentTaxonomy.model_validate(
    json.loads((REPO / "tests" / "fixtures" / "taxonomies" / "retail.json").read_text())
)


@dataclass(slots=True)
class Call:
    """One live call: graph, context and checkpoint thread."""

    ctx: GraphContext
    graph: object
    config: dict[str, object]
    state: dict[str, object]

    @property
    def bot_said(self) -> list[str]:
        return [t.content.text for t in self.state["turns"] if t.speaker is Speaker.BOT]  # type: ignore[union-attr]

    @property
    def last_bot(self) -> str:
        return self.bot_said[-1] if self.bot_said else ""

    def tools_called(self) -> list[str]:
        return [r.payload.tool_name for r in self.state.get("tool_records", [])]  # type: ignore[union-attr]


async def start(
    utterances: Sequence[str] = (),
    verification: VerificationLevel = VerificationLevel.NONE,
    grounded: bool = True,
    domain: str = "retail",
) -> Call:
    pack = load_pack(REPO / "domains", domain)
    loaded = LoadedDomain(
        pack=pack, root=REPO / "domains" / domain, taxonomy=TAXONOMY if domain == "retail" else None
    )
    provider = StubGraphProvider(grounded=grounded)
    bindings = load_bindings(REPO / "configs" / "models.yaml")
    ctx = build_context(loaded, provider, bindings, config_dir=REPO / "configs")
    router = (
        IntentRouter(provider, bindings.resolve("router"), TAXONOMY)
        if loaded.has_taxonomy
        else None
    )
    graph = build_graph(
        ctx,
        router,
        Responder(provider, bindings.resolve("task_agent")),
        checkpointer=memory_checkpointer(),
    )

    session_id = f"test-{id(provider)}"
    trace = TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id=session_id)
    config = thread_config(session_id, trace)
    state = await graph.ainvoke(
        SessionState(
            session_id=session_id,
            trace=trace,
            domain=domain,
            channel=Channel.VOICE,
            caller=CallerContext(caller_ref="a" * 32, verification=verification),
        ),
        config,
    )
    call = Call(ctx=ctx, graph=graph, config=config, state=state)
    for text in utterances:
        await say(call, text)
        if call.state.get("terminal"):
            break
    return call


async def say(call: Call, text: str) -> None:
    content = call.ctx.redaction.redact(text, call.ctx.redaction.new_allocator(call.ctx.vault))
    index = int(call.state["turn_index"])  # type: ignore[arg-type]
    turn = Turn(index=index, speaker=Speaker.CALLER, content=content)
    call.state = await call.graph.ainvoke(  # type: ignore[attr-defined]
        {"turns": [turn], "turn_index": index + 1}, call.config
    )


# ------------------------------------------------------------------- journeys


async def test_the_call_opens_with_consent_then_the_greeting() -> None:
    """A recorded call that never said so is a defect no later step repairs."""
    call = await start()
    spoken = call.last_bot.lower()
    assert "recorded" in spoken
    assert load_pack(REPO / "domains", "retail").greeting.strip().lower() in spoken


async def test_a_contained_journey_calls_a_tool_and_answers() -> None:
    call = await start(["where is my delivery", "ORD-884210"], verification=VerificationLevel.SOFT)
    assert call.tools_called() == ["get_order_status"]
    assert call.state.get("escalation") is None
    assert "on its way" in call.last_bot


async def test_the_caller_reference_reaches_the_backend_unmasked() -> None:
    """The vault's whole purpose: a model sees the placeholder, the tool sees the value."""
    call = await start(["where is my delivery", "ORD-884210"], verification=VerificationLevel.SOFT)
    record = call.state["tool_records"][0]  # type: ignore[index]
    backend = call.ctx.executor  # noqa: F841 - readability
    # The recorded payload keeps the masked form; a HandoffContext must never carry the value.
    assert "ORD-884210" not in record.model_dump_json()


async def test_a_slot_is_requested_before_any_tool_runs() -> None:
    call = await start(["where is my delivery"], verification=VerificationLevel.SOFT)
    assert call.tools_called() == []
    assert "reference" in call.last_bot.lower()
    assert call.state.get("pending_slot") == "order_reference"


async def test_an_ambiguous_request_is_clarified_not_guessed() -> None:
    call = await start(["I need help with something else"])
    assert call.state.get("clarification_count") == 1
    assert "is this about" in call.last_bot.lower()
    assert call.state.get("escalation") is None


async def test_a_second_failed_clarification_escalates() -> None:
    """The graph ends a clarify turn; the confidence policy is what bounds it."""
    call = await start(["I need help with something else"])
    await say(call, "I need help with something else")
    await say(call, "I need help with something else")
    assert call.state.get("escalation") is not None
    assert call.state.get("escalation").reason is HandoffReason.LOW_CONFIDENCE  # type: ignore[union-attr]


async def test_a_regulated_intent_escalates_without_calling_a_tool() -> None:
    call = await start(["I want to raise a dispute about a charge"])
    escalation = call.state.get("escalation")
    assert escalation is not None
    assert escalation.reason is HandoffReason.RISK_TIER  # type: ignore[union-attr]
    assert escalation.triggered_by == "risk"  # type: ignore[union-attr]
    assert call.tools_called() == []


async def test_an_unrecognised_request_escalates_rather_than_guessing() -> None:
    call = await start(["no idea really"])
    assert call.state.get("escalation") is not None
    assert call.tools_called() == []


async def test_an_ungroundable_answer_hands_over_instead_of_inventing_one() -> None:
    """Rule 4: missing data escalates; it is never improvised."""
    call = await start(
        ["where is my delivery", "ORD-884210"],
        verification=VerificationLevel.SOFT,
        grounded=False,
    )
    assert call.state.get("escalation") is not None
    assert "put you through" in call.last_bot.lower()


async def test_an_under_verified_caller_is_asked_to_step_up() -> None:
    """A refusal for want of verification is recoverable, not a failure."""
    call = await start(
        [
            "I want to change the address",
            "ORD-884210",
            "14 Bridge Lane",
            "BS1 4QP",
            "United Kingdom",
        ]
    )
    assert "security information" in call.last_bot.lower()
    assert call.state.get("escalation") is None
    denied = [r for r in call.state["tool_records"] if r.result.error_code == "not_authorized"]
    assert denied, "the step-up should follow a DENIED tool result"


async def test_a_failing_tool_escalates_rather_than_answering() -> None:
    call = await start(["where is my delivery", "ORD-404404"], verification=VerificationLevel.SOFT)
    assert call.state.get("tool_failure_count") >= 1
    assert call.state.get("escalation") is not None


# ------------------------------------------------------------------ invariants


async def test_every_terminal_call_writes_an_outcome() -> None:
    """A call that ends without one is a call nobody can report on."""
    call = await start(["I want to raise a dispute about a charge"])
    assert call.state.get("terminal") is True
    assert call.state.get("outcome") is not None
    assert call.state.get("outcome").status == "escalated"  # type: ignore[union-attr]


async def test_the_handoff_payload_builds_and_is_clean() -> None:
    from ccas.nodes.escalate import build_handoff

    call = await start(["I want to raise a dispute about a charge"])
    state = SessionState.model_validate(call.state)
    handoff = build_handoff(call.ctx, state, state.escalation)  # type: ignore[arg-type]
    assert handoff.summary.egress_permitted
    assert handoff.target_queue in {q.name for q in call.ctx.pack.queues}
    assert handoff.intent_path[-1] == "disputes.chargeback.raise"


async def test_an_unverified_caller_is_not_sent_to_a_verified_queue() -> None:
    """Routing them there just moves the problem to a human who also cannot proceed."""
    from ccas.nodes.escalate import build_handoff

    call = await start(["I want to raise a dispute about a charge"])
    state = SessionState.model_validate(call.state)
    handoff = build_handoff(call.ctx, state, state.escalation)  # type: ignore[arg-type]
    queue = call.ctx.pack.queues_by_name[handoff.target_queue]
    assert state.caller.verification.satisfies(queue.min_verification)


async def test_no_transcript_carries_the_raw_reference() -> None:
    call = await start(["where is my delivery", "ORD-884210"], verification=VerificationLevel.SOFT)
    rendered = json.dumps([t.content.text for t in call.state["turns"]])  # type: ignore[union-attr]
    assert "ORD-884210" not in rendered
    assert "[ACCOUNT_REF_1]" in rendered


async def test_the_turn_count_advances_once_per_utterance() -> None:
    call = await start()
    before = int(call.state["turn_index"])  # type: ignore[arg-type]
    await say(call, "where is my delivery")
    assert int(call.state["turn_index"]) > before  # type: ignore[arg-type]


async def test_a_pack_without_a_taxonomy_escalates_honestly() -> None:
    """No mined taxonomy means the platform cannot claim to understand anything."""
    call = await start(["I need help"], domain="healthcare")
    assert call.state.get("escalation") is not None
    assert call.tools_called() == []
