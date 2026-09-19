"""Escalation: build the CTI payload and tell the caller once.

``HandoffContext`` refuses construction if any field is unredacted, so this node is the
last place a leak could be caught -- and it is caught by the contract rather than by
this code, which is the point.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.llm.prompt import authored
from ccas.nodes.base import NodeFn, speak
from ccas.policies.base import PolicyAction
from ccas.schemas.common import Urgency, utcnow
from ccas.schemas.escalation import EscalationDecision, HandoffReason
from ccas.schemas.handoff import HandoffContext, VerifiedIdentity
from ccas.schemas.session import SessionState

__all__ = ["HANDOFF_MESSAGE", "NAME", "build_handoff", "make_node"]

NAME = "escalate"

HANDOFF_MESSAGE = (
    "Let me put you through to someone who can help with that. "
    "I'll pass on everything we've covered so you don't have to repeat it."
)


def build_handoff(
    ctx: GraphContext, state: SessionState, decision: EscalationDecision
) -> HandoffContext:
    taxonomy = ctx.taxonomy
    intent_path: tuple[str, ...] = ()
    if taxonomy is not None and state.current_intent and state.current_intent.intent_id:
        try:
            intent_path = tuple(n.intent_id for n in taxonomy.path(state.current_intent.intent_id))
        except KeyError:
            intent_path = ()

    queue = _queue_for(ctx, state, decision)
    summary = ctx.redaction.redact(_summarise(state, decision))

    identity = None
    if state.caller.verification.rank > 0:
        identity = VerifiedIdentity(
            caller_ref=state.caller.caller_ref,
            level=state.caller.verification,
            method="session",
            # Verified attributes are caller-derived -- a name, a date of birth, whatever
            # the pack asked for. They travel to a third-party desktop, so they go through
            # the redactor like any other caller text rather than riding out as `str`.
            attributes={
                name: ctx.redaction.redact(value)
                for name, value in state.caller.verified_attributes.items()
            },
            verified_at=utcnow(),
        )

    return HandoffContext(
        handoff_id=f"{state.session_id}-handoff",
        session_id=state.session_id,
        trace=state.trace,
        domain=state.domain,
        reason=decision.reason,
        urgency=decision.urgency,
        target_queue=queue.name,
        required_skills=queue.skills,
        identity=identity,
        intent_path=intent_path,
        intent_confidence=state.current_intent.confidence if state.current_intent else 0.0,
        collected_slots=dict(state.slots),
        summary=summary,
        sentiment_trail=tuple(state.sentiment_trail),
        transcript=tuple(state.turns),
        tool_trace=tuple(r for r in state.tool_records if r.result.safe_for_model),
        # Authored, not caller-derived: a policy name, an enum member and a turn count.
        # Nothing here originates with the caller, which is the only reason `authored` is
        # the right wrapper rather than the redactor.
        cti_attributes={
            "triggered_by": authored(decision.triggered_by),
            "reason": authored(decision.reason.value),
            "turns": authored(str(state.turn_index)),
        },
    )


def make_node(ctx: GraphContext) -> NodeFn:
    async def escalate(state: SessionState) -> dict[str, Any]:
        decision = state.escalation or _decision_from_policy(ctx, state)
        handoff = build_handoff(ctx, state, decision)
        return {
            **speak(ctx, state, HANDOFF_MESSAGE),
            "escalation": decision,
            "active_agent": None,
            "outcome": _outcome(state, handoff),
            "terminal": True,
        }

    return escalate


def _decision_from_policy(ctx: GraphContext, state: SessionState) -> EscalationDecision:
    """Re-derive the verdict that routed us here.

    The conditional edge computes a verdict to choose this node but does not write it,
    which keeps edges pure. Re-evaluating costs nothing (policies are pure functions)
    and means a handoff always names the rule that fired instead of saying "unspecified".
    """
    verdict = ctx.policies.evaluate(state, ctx.policy_context(state))
    if verdict.action is PolicyAction.ESCALATE and verdict.reason is not None:
        return verdict.to_escalation(at_turn=state.turn_index)
    return EscalationDecision(
        reason=HandoffReason.POLICY,
        triggered_by=verdict.policy,
        at_turn=state.turn_index,
        urgency=Urgency.NORMAL,
        detail=verdict.detail,
    )


def _queue_for(
    ctx: GraphContext, state: SessionState, decision: EscalationDecision
) -> Any:  # QueueSpec
    """The intent's queue when it names one and the caller qualifies for it."""
    node = ctx.node_for(state)
    named = node.escalation.target_queue if node and node.escalation.target_queue else None
    queues = ctx.pack.queues_by_name
    candidate = queues.get(named) if named else None
    if candidate is None:
        return ctx.pack.default_queue
    if not state.caller.verification.satisfies(candidate.min_verification):
        # Routing an unverified caller to a queue that requires verification just moves
        # the problem to a human. Send them somewhere that can actually take the call.
        return ctx.pack.default_queue
    return candidate


def _summarise(state: SessionState, decision: EscalationDecision) -> str:
    """Composed from redacted turns, then redacted again on the way out."""
    intent = (
        state.current_intent.intent_id
        if state.current_intent and state.current_intent.intent_id
        else "not identified"
    )
    lines = [
        f"Caller reached a handoff after {state.turn_index} turn(s).",
        f"Intent: {intent}. Reason: {decision.reason.value} (via {decision.triggered_by}).",
    ]
    if decision.detail:
        lines.append(f"Detail: {decision.detail}")
    if state.slots:
        captured = ", ".join(sorted(name for name, s in state.slots.items() if s.valid))
        lines.append(f"Collected: {captured or 'nothing'}.")
    failures = state.failed_tool_calls()
    if failures:
        lines.append(
            "Tool failures: "
            + ", ".join(f"{r.payload.tool_name}={r.result.error_code}" for r in failures)
        )
    return " ".join(lines)


def _outcome(state: SessionState, handoff: HandoffContext) -> Any:  # SessionOutcome
    from ccas.schemas.session import SessionOutcome

    return SessionOutcome(
        status="escalated",
        resolved_intent=handoff.leaf_intent,
        turn_count=state.turn_index,
        duration_ms=0,
        disposition_code=handoff.reason.value,
    )
