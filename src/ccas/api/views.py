"""Turning session state into what the workbench displays.

Every field here is already redacted -- the API re-reads the same objects the graph
produced. Nothing reaches this layer that could not also reach a model.
"""

from __future__ import annotations

from typing import Any

from ccas.api.schemas import SessionSnapshot
from ccas.api.sessions import SessionHandle
from ccas.schemas.session import SessionState

__all__ = ["snapshot_of", "tool_record_view"]


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
