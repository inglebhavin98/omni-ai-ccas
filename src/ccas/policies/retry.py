"""Exhaustion policy: the ladders that stop a call looping.

Every graph must have a turn ceiling (CLAUDE.md Rule 4), and every repair attempt must
be bounded. Without this a caller can be asked to repeat themselves indefinitely, which
is the single most-complained-about behaviour of the systems this platform replaces.
"""

from __future__ import annotations

from ccas.policies.base import PolicyAction, PolicyContext, PolicyDecision, proceed
from ccas.schemas.common import Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import SessionState

__all__ = ["NAME", "evaluate"]

NAME = "retry"


def evaluate(state: SessionState, ctx: PolicyContext) -> PolicyDecision:
    if state.turn_index >= ctx.max_turns:
        return _escalate(
            HandoffReason.MAX_TURNS,
            f"turn {state.turn_index} reached the {ctx.max_turns}-turn ceiling",
            Urgency.NORMAL,
        )

    if state.tool_failure_count >= ctx.escalation.max_tool_failures:
        return _escalate(
            HandoffReason.TOOL_FAILURE,
            f"{state.tool_failure_count} tool failure(s); the system cannot complete this",
            Urgency.HIGH,
        )

    if state.no_input_count >= ctx.max_no_input:
        return _escalate(
            HandoffReason.MAX_TURNS,
            f"{state.no_input_count} turns with no caller input",
            Urgency.NORMAL,
        )

    if state.no_match_count >= ctx.max_no_match:
        return _escalate(
            HandoffReason.UNSUPPORTED_INTENT,
            f"{state.no_match_count} turns without a recognised intent",
            Urgency.NORMAL,
        )

    return proceed(NAME)


def _escalate(reason: HandoffReason, detail: str, urgency: Urgency) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.ESCALATE,
        policy=NAME,
        reason=reason,
        urgency=urgency,
        detail=detail,
    )
