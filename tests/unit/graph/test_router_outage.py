"""A provider outage must escalate, not kill the turn (Rule 4).

Observed live: a free-tier 429 propagated out of the router, through the graph, and out
of the HTTP handler. The caller got a traceback and the session was unrecoverable --
when the platform already knows how to say "I can't help with that, let me find someone
who can". Rule 4 requires a reachable escalation path; an exception is not one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ccas.graph.context import GraphContext
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProviderError, ProviderRateLimitedError
from ccas.llm.bindings import load_bindings
from ccas.nodes.route import make_node
from ccas.schemas.common import Speaker
from ccas.schemas.session import SessionState, Turn
from tests.factories import redacted


@pytest.fixture
def working_router(ctx: GraphContext) -> IntentRouter:
    repo = Path(__file__).resolve().parents[3]
    binding = load_bindings(repo / "configs" / "models.yaml").resolve("router")
    return IntentRouter(ctx.provider, binding, ctx.domain.taxonomy)


class _FailingRouter:
    """Stands in for IntentRouter when the provider is down."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def classify(self, utterance: Any, history: Any = ()) -> Any:
        raise self._error


def _with_utterance(session: SessionState, text: str) -> SessionState:
    turn = Turn(index=0, speaker=Speaker.CALLER, content=redacted(text))
    return session.model_copy(update={"turns": (turn,)})


@pytest.mark.parametrize(
    "error",
    [ProviderRateLimitedError("429 free-models-per-day"), LLMProviderError("upstream 502")],
)
async def test_an_outage_escalates_rather_than_raising(
    ctx: GraphContext, session: SessionState, error: Exception
) -> None:
    node = make_node(ctx, _FailingRouter(error))  # type: ignore[arg-type]
    update = await node(_with_utterance(session, "where is my delivery"))

    assert update["current_intent"].intent_id is None
    assert update["current_intent"].confidence == 0.0
    assert update["no_match_count"] == session.no_match_count + 1


async def test_the_outage_is_recorded_so_the_reason_is_not_lost(
    ctx: GraphContext, session: SessionState
) -> None:
    """ "The model was unreachable" and "the caller was unclear" need different handling.

    Without this the escalation reads as a comprehension failure, and whoever reviews the
    transcript concludes the router is bad when the provider was simply down.
    """
    node = make_node(ctx, _FailingRouter(ProviderRateLimitedError("429")))  # type: ignore[arg-type]
    update = await node(_with_utterance(session, "track my delivery"))
    assert update["current_intent"].source == "unavailable"


async def test_a_working_router_is_untouched(
    ctx: GraphContext, session: SessionState, working_router: IntentRouter
) -> None:
    """The guard must not swallow a normal classification."""
    update = await make_node(ctx, working_router)(_with_utterance(session, "where is my delivery"))
    assert "current_intent" in update
    assert update["current_intent"].source != "unavailable"
