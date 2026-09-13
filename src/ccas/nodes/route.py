"""Front-door routing.

Classify, then let the policies decide. The node does not choose what happens next --
it records a prediction and the graph's conditional edge reads the policy verdict. That
separation is what makes every escalation traceable to a named rule.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProviderError, ProviderRateLimitedError
from ccas.nodes.base import NodeFn, caller_history, latest_caller_text, speak
from ccas.observability.logging import get_logger
from ccas.schemas.session import IntentPrediction, SessionState

__all__ = ["NAME", "NO_INPUT_PROMPT", "make_node"]

LOG = get_logger("nodes.route")

NAME = "route"

NO_INPUT_PROMPT = "Sorry, I didn't catch that. What can I help you with?"


def make_node(ctx: GraphContext, router: IntentRouter | None) -> NodeFn:
    async def route(state: SessionState) -> dict[str, Any]:
        utterance = latest_caller_text(state)
        if utterance is None:
            return {
                **speak(ctx, state, NO_INPUT_PROMPT),
                "no_input_count": state.no_input_count + 1,
            }

        if router is None:
            # No taxonomy mined yet. Honest failure: the graph escalates rather than
            # pretending to understand.
            unresolved = IntentPrediction(confidence=0.0, source="fallback", latency_ms=0)
            return {
                "current_intent": unresolved,
                "intent_history": [unresolved],
                "no_match_count": state.no_match_count + 1,
            }

        try:
            prediction = await router.classify(utterance, caller_history(state))
        except LLMProviderError as exc:
            # Rule 4: the graph escalates, it does not raise. A provider outage that
            # propagates leaves the caller with nothing and the session unrecoverable,
            # when the platform already knows how to hand over to a human.
            LOG.warning(
                "route.router_unavailable",
                correlation_id=state.trace.correlation_id,
                reason=type(exc).__name__,
                rate_limited=isinstance(exc, ProviderRateLimitedError),
            )
            unavailable = IntentPrediction(confidence=0.0, source="unavailable", latency_ms=0)
            return {
                "current_intent": unavailable,
                "intent_history": [unavailable],
                "no_match_count": state.no_match_count + 1,
            }
        update: dict[str, Any] = {
            "current_intent": prediction,
            "intent_history": [prediction],
            "latency": state.latency.model_copy(update={"router_ms": prediction.latency_ms}),
        }
        if prediction.intent_id is None:
            update["no_match_count"] = state.no_match_count + 1
        else:
            # The L1 segment names the micro-agent that owns this intent.
            update["active_agent"] = prediction.intent_id.split(".", 1)[0]
        return update

    return route
