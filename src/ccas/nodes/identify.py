"""Identification: establish what verification level the caller has.

Deliberately passive on the first pass. Collecting a factor nobody has asked for yet is
both bad UX and bad data hygiene -- the caller may only want an opening-hours answer.
The level rises when something actually demands it: ``tool_exec`` returning ``DENIED``
routes back here with a step-up, which is the only path that asks for a factor.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn, speak
from ccas.schemas.common import VerificationLevel
from ccas.schemas.session import SessionState

__all__ = ["NAME", "STEP_UP_PROMPT", "make_node", "required_level"]

NAME = "identify"

STEP_UP_PROMPT = (
    "Before I can do that I need to check a couple of details with you. "
    "Can you confirm the security information on the account?"
)


def required_level(ctx: GraphContext, state: SessionState) -> VerificationLevel:
    """The strongest level any tool this intent needs would demand."""
    node = ctx.node_for(state)
    if node is None:
        return VerificationLevel.NONE
    levels = [
        spec.requires_verification
        for spec in ctx.registry.for_intent(node)
        if ctx.registry.has(spec.name)
    ]
    return max(levels, key=lambda level: level.rank, default=VerificationLevel.NONE)


def make_node(ctx: GraphContext) -> NodeFn:
    async def identify(state: SessionState) -> dict[str, Any]:
        needed = required_level(ctx, state)
        if state.caller.verification.satisfies(needed):
            # Nothing to ask for. Passing through without speaking keeps the turn count
            # honest -- a node that always talks would burn the caller's turn budget.
            return {}
        return {
            **speak(ctx, state, STEP_UP_PROMPT),
            "pending_slot": None,
        }

    return identify
