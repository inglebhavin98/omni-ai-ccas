"""TypeSafe ``jev`` -- a decision model rather than a text model.

**This is a spike. Nothing in `graph/` or `nodes/` imports it.** Putting a vendor on the
call path is a locked-stack change and needs an ADR (Rule 5), and Rule 6 would still
require a second variant naming a distinct model. What this exists for is to answer one
question with evidence instead of argument: does a purpose-built Choice primitive route
better, and faster, than coaxing a chat model into structured output?

Why it is worth asking. The router picks one intent from a closed set and the policies
then gate on the confidence that comes back. That is exactly a Choice, and today it is a
chat completion squeezed into a schema -- the failure ADR-0012 records, where a model
ignores ``response_format`` and the mistake surfaces far from its cause. A primitive that
returns a typed choice with a probability distribution removes that class of bug instead
of defending against it.

Deliberately **not** an ``LLMProvider``. That protocol is chat-shaped (complete / stream /
structured) and jev is not a chat model; forcing it through would misrepresent both. If
this graduates, the seam it needs is its own.

Uses ``httpx`` directly rather than ``typesafe_sdk``: a spike must not add a dependency
to decide whether to add a dependency.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ccas.llm.base import (
    LLMProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ccas.schemas.pii import RedactedText
from ccas.schemas.taxonomy import IntentTaxonomy

__all__ = [
    "DEFAULT_BASE_URL",
    "ROUTER_QUESTION",
    "AskResult",
    "RoutingChoice",
    "TypeSafeClient",
    "TypeSafeError",
    "choice_request",
    "criteria_from_taxonomy",
    "level_criteria",
    "questions_request",
    "routing_from",
]

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"

#: One question per call for the spike, so latency is comparable with the router's single
#: chat completion. Batching is where jev claims its advantage and is the obvious follow-up.
ROUTER_QUESTION = "intent"

_INSTRUCTIONS = (
    "Which intent does the caller's message express? Choose the single best match. "
    "The message has been redacted: bracketed tokens such as [ORDER_REF_1] stand in for "
    "values that were removed, and are not part of what the caller said."
)


class TypeSafeError(LLMProviderError):
    """The service answered, but not with something usable."""


@dataclass(frozen=True, slots=True)
class RoutingChoice:
    intent_id: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    path: tuple[str, ...] = ()
    """L1 -> leaf, when routed hierarchically. Empty for a flat choice."""

    calls: int = 1


@dataclass(frozen=True, slots=True)
class AskResult:
    """One answered request, before anybody interprets it.

    Deliberately not parsed: the transport should not know what a Choice or a Score means,
    so reading the answers belongs to the caller that asked the question. ``ccas.evals``
    depends on ``ccas.llm`` and not the other way round, and a judge rubric living in here
    would invert that.
    """

    body: dict[str, Any]
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0


def questions_request(
    state: Mapping[str, RedactedText],
    questions: Mapping[str, Any],
    *,
    model: str = "jev-latest",
) -> dict[str, Any]:
    """Shape a request whose state is several named fields rather than one string.

    The judge needs this shape: the caller's turns, the reply under judgement, the tool
    results behind it and the rules it had to follow are separate things, and flattening
    them into one blob asks the model to work out which is which.

    Every field goes through ``require_egress``, so a single value that lost its clearance
    stops the whole request -- one unredacted field is a leak whatever the others hold.
    """
    if not questions:
        raise ValueError("a request with no questions asks nothing")
    return {
        "state": {name: text.require_egress() for name, text in state.items()},
        "model": model,
        "questions": dict(questions),
    }


def criteria_from_taxonomy(taxonomy: IntentTaxonomy) -> dict[str, str]:
    """The choosable set, straight from the pack.

    Built rather than hand-listed so it cannot drift from the taxonomy it mirrors. The
    description is what the model actually discriminates on, so a node with none falls
    back to its label rather than offering the model an empty string.
    """
    return {
        leaf.intent_id: (leaf.description or leaf.label or leaf.intent_id).strip()
        for leaf in taxonomy.leaves()
    }


def level_criteria(taxonomy: IntentTaxonomy, parent_id: str | None) -> dict[str, str]:
    """One sibling set: the roots when ``parent_id`` is None, otherwise its children.

    The unit a hierarchical Choice is asked over. Eleven categories, then two to six
    leaves, instead of twenty-seven options of which twenty-six are irrelevant to any
    given message -- which is jagged edge 5, "accuracy falls as the state grows with
    content unrelated to the decision".
    """
    return {
        node.intent_id: (node.description or node.label or node.intent_id).strip()
        for node in taxonomy.nodes
        if node.parent_id == parent_id
    }


def choice_request(
    state: RedactedText, criteria: dict[str, str], *, model: str = "jev-latest"
) -> dict[str, Any]:
    """Shape one Choice call. ``require_egress`` is the gate, and it is the only door."""
    return {
        "state": state.require_egress(),
        "model": model,
        "questions": {
            ROUTER_QUESTION: {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }


def routing_from(
    body: dict[str, Any], allowed: set[str] | None = None
) -> tuple[str, float, dict[str, float]]:
    """Read the answer, refusing anything that is not one.

    A missing or malformed answer raises rather than returning "no intent". The two look
    identical downstream and mean opposite things: one is the model abstaining, the other
    is the integration being broken, and only the second should fail a run.
    """
    answers = body.get("answers")
    if not isinstance(answers, dict) or ROUTER_QUESTION not in answers:
        raise TypeSafeError(f"response carries no {ROUTER_QUESTION!r} answer")
    answer = answers[ROUTER_QUESTION]
    intent = answer.get("choice")
    if not isinstance(intent, str) or not intent:
        raise TypeSafeError("choice answer carries no selection")
    if allowed is not None and intent not in allowed:
        raise TypeSafeError(f"chose {intent!r}, which is not in the declared criteria")
    probabilities = answer.get("probabilities") or {}
    return intent, float(answer.get("confidence", 0.0)), dict(probabilities)


class TypeSafeClient:
    """Minimal async client. One endpoint, one verb."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout_ms: int = 10_000,
        model: str = "jev-latest",
    ) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._timeout_ms = timeout_ms
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def ask(
        self, state: Mapping[str, RedactedText], questions: Mapping[str, Any]
    ) -> AskResult:
        """Several questions against one named state, in one request.

        The batching is the point. The state is the expensive part of a judge call -- a
        whole exchange plus the tool results behind it -- and each question is scored
        independently against it, so asking three in one request sends the transcript once
        instead of three times.
        """
        return await self._send(questions_request(state, questions, model=self._model))

    async def _send(self, payload: dict[str, Any]) -> AskResult:
        """The transport, and the only place an HTTP failure becomes a provider error.

        Shaping happens before this is called, so ``require_egress`` raises while the
        request is still a dict -- a payload that lost its clearance never reaches a
        socket at all.
        """
        started = time.perf_counter_ns()
        try:
            response = await self._client.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout_ms / 1000
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"typesafe unreachable at {self._url}") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"typesafe did not respond within {self._timeout_ms}ms"
            ) from exc
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000

        if response.status_code == 429:
            raise ProviderRateLimitedError(f"typesafe rate limited: {response.text[:160]}")
        if response.status_code >= 400:
            raise TypeSafeError(f"typesafe returned {response.status_code}: {response.text[:200]}")

        body: dict[str, Any] = response.json()
        usage = body.get("usage") or {}
        return AskResult(
            body=body,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    async def choose(self, state: RedactedText, criteria: dict[str, str]) -> RoutingChoice:
        result = await self._send(choice_request(state, criteria, model=self._model))
        intent, confidence, probabilities = routing_from(result.body, allowed=set(criteria))
        return RoutingChoice(
            intent_id=intent,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def choose_hierarchical(
        self, state: RedactedText, taxonomy: IntentTaxonomy
    ) -> RoutingChoice:
        """Descend the taxonomy, one Choice per level (greedy).

        Greedy rather than beam: a beam explores K paths per level and can repair an
        ambiguous early decision, which the docs show recovering 4 of 4 against greedy's
        2 of 4 -- but it multiplies calls per level and this is a spike measuring whether
        the shape helps at all. Beam is the obvious next step if it does.

        ``confidence`` is the geometric mean of the chosen edge's probability at each
        level, which is how the docs rank beam paths. **It is not comparable to the chat
        router's confidence** and the pack's 0.82 route threshold must not be carried
        over to it -- jagged edge 8 is explicit that thresholds do not transfer between
        question formats.
        """
        parent: str | None = None
        path: list[str] = []
        edges: list[float] = []
        confidences: list[float] = []
        probabilities: dict[str, float] = {}
        tokens_in = tokens_out = calls = 0
        started = time.perf_counter_ns()

        while True:
            criteria = level_criteria(taxonomy, parent)
            if not criteria:
                break
            if len(criteria) == 1:
                # A Choice over one option spends a call to learn nothing.
                only = next(iter(criteria))
                path.append(only)
                parent = only
                continue

            result = await self.choose(state, criteria)
            calls += 1
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            path.append(result.intent_id)
            edges.append(result.probabilities.get(result.intent_id, result.confidence))
            confidences.append(result.confidence)
            probabilities = result.probabilities
            parent = result.intent_id

        latency_ms = (time.perf_counter_ns() - started) // 1_000_000
        score = 0.0
        if edges:
            product = 1.0
            for edge in edges:
                product *= edge
            score = product ** (1 / len(edges))
        elif confidences:
            score = confidences[-1]

        return RoutingChoice(
            intent_id=path[-1] if path else "",
            confidence=score,
            probabilities=probabilities,
            latency_ms=latency_ms,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            path=tuple(path),
            calls=calls,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
