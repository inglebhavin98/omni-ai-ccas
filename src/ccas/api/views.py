"""Turning session state into what the workbench displays.

Every field here is already redacted -- the API re-reads the same objects the graph
produced. Nothing reaches this layer that could not also reach a model.
"""

from __future__ import annotations

from typing import Any

from ccas.api.schemas import SessionSnapshot
from ccas.api.sessions import SessionHandle
from ccas.schemas.handoff import HandoffContext
from ccas.schemas.session import SessionState

__all__ = ["handoff_of", "snapshot_of", "tool_record_view"]


def tool_record_view(record: Any) -> dict[str, Any]:
    result = record.result
    return {
        "tool": record.payload.tool_name,
        "arguments": record.payload.arguments,
        "attempt": record.attempt,
        "status": result.status.value,
        "elapsed_ms": result.elapsed_ms,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "safe_for_model": result.safe_for_model,
        "data": result.data if result.safe_for_model else {},
        "redacted_entities": result.redaction.entity_counts,
    }


def handoff_of(handoff: HandoffContext) -> dict[str, Any]:
    """The agent-desktop view of a CTI payload.

    Serialising the model directly would also be correct -- the contract layer already
    guarantees egress-cleanliness -- but the desktop reads a few derived things (the
    identity's attributes, the queue skills) and skips the internals it has no use for,
    so the shape is named here rather than implied.
    """
    return {
        "handoff_id": handoff.handoff_id,
        "session_id": handoff.session_id,
        "domain": handoff.domain,
        "schema_version": handoff.schema_version,
        "reason": handoff.reason.value,
        "urgency": handoff.urgency.value,
        "target_queue": handoff.target_queue,
        "required_skills": list(handoff.required_skills),
        "intent_path": list(handoff.intent_path),
        "intent_confidence": handoff.intent_confidence,
        "summary": handoff.summary.text,
        "identity": (
            {
                "level": handoff.identity.level.value,
                "method": handoff.identity.method,
                "attributes": {k: v.text for k, v in handoff.identity.attributes.items()},
            }
            if handoff.identity
            else None
        ),
        "collected_slots": {name: slot.raw.text for name, slot in handoff.collected_slots.items()},
        "transcript": [
            {
                "index": turn.index,
                "speaker": turn.speaker.value,
                "text": turn.content.text,
            }
            for turn in handoff.transcript
        ],
        "tool_trace": [tool_record_view(r) for r in handoff.tool_trace],
        "cti_attributes": {k: v.text for k, v in handoff.cti_attributes.items()},
        "created_at": handoff.created_at.isoformat(),
    }


def snapshot_of(handle: SessionHandle) -> SessionSnapshot:
    state: SessionState = handle.snapshot()
    ctx = handle.ctx

    # Every verdict, not just the decisive one: seeing *why* a turn went the way it did
    # is the whole point of a workbench.
    verdicts = [
        {
            "policy": verdict.policy,
            "action": verdict.action.value,
            "reason": verdict.reason.value if verdict.reason else None,
            "detail": verdict.detail,
            "decisive": verdict.decisive,
        }
        for verdict in ctx.policies.evaluate_all(state, ctx.policy_context(state))
    ]

    intent = state.current_intent
    return SessionSnapshot(
        session_id=state.session_id,
        domain=state.domain,
        turn_index=state.turn_index,
        terminal=state.terminal,
        transcript=handle.transcript(),
        current_intent=(
            {
                "intent_id": intent.intent_id,
                "confidence": intent.confidence,
                "source": intent.source,
                "alternatives": [list(a) for a in intent.alternatives],
                "latency_ms": intent.latency_ms,
            }
            if intent
            else None
        ),
        active_agent=state.active_agent,
        slots={
            name: {
                "value": slot.raw.text,
                "valid": slot.valid,
                "attempts": slot.attempts,
                "captured_via": slot.captured_via,
            }
            for name, slot in state.slots.items()
        },
        pending_slot=state.pending_slot,
        counters={
            "clarifications": state.clarification_count,
            "no_input": state.no_input_count,
            "no_match": state.no_match_count,
            "barge_ins": state.barge_in_count,
            "tool_failures": state.tool_failure_count,
        },
        policy_verdicts=verdicts,
        tool_records=[tool_record_view(r) for r in state.tool_records],
        escalation=(
            {
                "reason": state.escalation.reason.value,
                "triggered_by": state.escalation.triggered_by,
                "urgency": state.escalation.urgency.value,
                "at_turn": state.escalation.at_turn,
                "detail": state.escalation.detail,
            }
            if state.escalation
            else None
        ),
        outcome=(
            {
                "status": state.outcome.status,
                "resolved_intent": state.outcome.resolved_intent,
                "turn_count": state.outcome.turn_count,
                "disposition_code": state.outcome.disposition_code,
            }
            if state.outcome
            else None
        ),
        latency={
            **state.latency.stages(),
            "total_rtt_ms": state.latency.total_rtt_ms,
            "budget_ms": state.latency.budget_ms,
            "breached": state.latency.breached,
        },
    )
