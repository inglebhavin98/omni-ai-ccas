"""Clarification: one question that separates the two likeliest intents.

One question, not a menu. A caller on a phone cannot hold a list in their head, which is
the single behaviour this platform exists to replace. Bounded by ``max_clarifications``;
the confidence policy escalates once they are spent.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn, speak
from ccas.schemas.session import SessionState

__all__ = ["FALLBACK_PROMPT", "NAME", "make_node"]

NAME = "clarify"

FALLBACK_PROMPT = "I want to make sure I get this right. Can you tell me a bit more?"


def make_node(ctx: GraphContext) -> NodeFn:
    async def clarify(state: SessionState) -> dict[str, Any]:
        question = _question_for(ctx, state)
        return {
            **speak(ctx, state, question),
            "clarification_count": state.clarification_count + 1,
        }

    return clarify


def _question_for(ctx: GraphContext, state: SessionState) -> str:
    """Name the two candidates when there are two; ask openly when there are not.

    Naming them is far more useful than a generic reprompt -- but only when the
    alternatives are genuinely close, otherwise it offers the caller a wrong option.
    """
    taxonomy = ctx.taxonomy
    prediction = state.current_intent
    if taxonomy is None or prediction is None or not prediction.alternatives:
        return FALLBACK_PROMPT

    runner_up_id, runner_up_confidence = prediction.alternatives[0]
    if prediction.intent_id is None or runner_up_confidence < prediction.confidence / 2:
        return FALLBACK_PROMPT

    try:
        first = taxonomy.resolve(prediction.intent_id)
        second = taxonomy.resolve(runner_up_id)
    except KeyError:
        return FALLBACK_PROMPT

    return (
        f"Just so I send you to the right place -- is this about "
        f"{first.label.lower()}, or {second.label.lower()}?"
    )
