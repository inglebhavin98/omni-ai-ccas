"""TypeSafe `jev` binding, over an in-process transport. No network -- Rule 7.

A spike, not an adoption: nothing in `graph/` or `nodes/` imports this. Adding a vendor to
the call path is a locked-stack change and needs an ADR (Rule 5). What these tests pin is
the part a spike gets wrong quietly -- the request shape and the egress gate.
"""

from __future__ import annotations

import httpx
import pytest

from ccas.llm.typesafe import (
    ROUTER_QUESTION,
    TypeSafeClient,
    TypeSafeError,
    choice_request,
    criteria_from_taxonomy,
    routing_from,
)
from tests.factories import redacted

CRITERIA = {
    "order.track_order": "Caller wants to track an order.",
    "refund.get_refund": "Caller wants a refund.",
}


def reply(choice: str = "order.track_order", confidence: float = 0.82) -> dict[str, object]:
    return {
        "model": "jev-latest",
        "answers": {
            ROUTER_QUESTION: {
                "type": "choice",
                "choice": choice,
                "probabilities": {"order.track_order": 0.9, "refund.get_refund": 0.1},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }


def client(handler: object, **kw: object) -> TypeSafeClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return TypeSafeClient(api_key="k", client=httpx.AsyncClient(transport=transport), **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------- pure shaping


def test_criteria_are_built_from_the_taxonomy_not_hand_written() -> None:
    """Whatever the pack declares is what the model may choose from. A hand-kept list is
    a second source of truth that silently drifts from the taxonomy it mirrors."""
    import json
    from pathlib import Path

    from ccas.schemas.taxonomy import IntentTaxonomy

    repo = Path(__file__).resolve().parents[3]
    taxonomy = IntentTaxonomy.model_validate(
        json.loads((repo / "domains" / "retail" / "taxonomy.json").read_text())
    )
    criteria = criteria_from_taxonomy(taxonomy)
    leaves = {leaf.intent_id for leaf in taxonomy.leaves()}
    assert set(criteria) == leaves
    assert all(v.strip() for v in criteria.values()), "an empty criterion tells the model nothing"


def test_the_request_carries_state_criteria_and_one_question() -> None:
    body = choice_request(redacted("where is my order"), CRITERIA, model="jev-latest")
    assert body["state"] == "where is my order"
    assert body["model"] == "jev-latest"
    question = body["questions"][ROUTER_QUESTION]  # type: ignore[index]
    assert question["type"] == "choice"
    assert question["criteria"] == CRITERIA
    assert question["instructions"]


def test_routing_is_read_out_of_the_answer() -> None:
    intent, confidence, probs = routing_from(reply())
    assert intent == "order.track_order"
    assert confidence == 0.82
    assert probs["order.track_order"] == 0.9


def test_an_answer_naming_an_unknown_intent_is_refused() -> None:
    """A choice outside the declared set is a contract violation, and routing a call into
    an intent the pack does not define would die later on a missing tool."""
    with pytest.raises(TypeSafeError, match="not in the declared"):
        routing_from(reply(choice="billing.mystery"), allowed=set(CRITERIA))


def test_a_malformed_answer_is_an_error_not_a_none_intent() -> None:
    """Silently returning "no intent" would read as an honest abstention and be scored as
    one. A broken response is a failure and has to look like one."""
    with pytest.raises(TypeSafeError):
        routing_from({"model": "jev-latest", "answers": {}})


# ----------------------------------------------------------------- behavioural


async def test_a_redacted_utterance_routes() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply())

    result = await client(handler).choose(redacted("where is my order"), CRITERIA)
    assert result.intent_id == "order.track_order"
    assert result.confidence == 0.82
    assert result.latency_ms >= 0


async def test_unredacted_text_never_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 2. TypeSafe is an external vendor like any other, so the gate is the type."""
    from ccas.schemas.pii import RedactionStatus

    sent: list[object] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, json=reply())

    dirty = redacted("card 4242 4242 4242 4242", RedactionStatus.UNVERIFIED)
    with pytest.raises(PermissionError, match="egress blocked"):
        await client(handler).choose(dirty, CRITERIA)
    assert sent == [], "nothing may be put on the wire before the gate runs"


async def test_a_429_is_reported_as_never_served() -> None:
    """The distinction ADR-0014 and ADR-0021 turn on: a case nobody served leaves the
    denominator rather than counting as a wrong answer."""
    from ccas.llm.base import ProviderRateLimitedError

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "quota"})

    with pytest.raises(ProviderRateLimitedError):
        await client(handler).choose(redacted("hello"), CRITERIA)


async def test_a_timeout_is_converted_not_leaked() -> None:
    from ccas.llm.base import ProviderTimeoutError

    async def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        await client(handler).choose(redacted("hello"), CRITERIA)


async def test_the_api_key_is_sent_as_a_bearer_token() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json=reply())

    await client(handler).choose(redacted("hello"), CRITERIA)
    assert seen == ["Bearer k"]


# ------------------------------------------------- hierarchical (the documented shape)


def _taxonomy():  # type: ignore[no-untyped-def]
    import json
    from pathlib import Path

    from ccas.schemas.taxonomy import IntentTaxonomy

    repo = Path(__file__).resolve().parents[3]
    return IntentTaxonomy.model_validate(
        json.loads((repo / "domains" / "retail" / "taxonomy.json").read_text())
    )


def test_level_criteria_returns_the_roots_then_the_children() -> None:
    from ccas.llm.typesafe import level_criteria

    taxonomy = _taxonomy()
    roots = level_criteria(taxonomy, None)
    assert "account" in roots and "order" in roots
    assert all("." not in key for key in roots), "roots are L1, not leaves"

    children = level_criteria(taxonomy, "order")
    assert set(children) == {n.intent_id for n in taxonomy.nodes if n.parent_id == "order"}
    assert all(key.startswith("order.") for key in children)


async def test_hierarchical_routing_descends_one_level_at_a_time() -> None:
    """The documented approach for a taxonomy: 27 options flat becomes 11 then 4.

    Jagged edge 5 is the reason -- "accuracy falls as the state grows with content
    unrelated to the decision", and 27 sibling criteria are mostly irrelevant to any
    given message.
    """
    asked: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        criteria = body["questions"][ROUTER_QUESTION]["criteria"]
        asked.append(criteria)
        pick = "order" if "account" in criteria else "order.track_order"
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    ROUTER_QUESTION: {
                        "type": "choice",
                        "choice": pick,
                        "probabilities": {pick: 0.9},
                        "confidence": 0.9,
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    from ccas.llm.typesafe import TypeSafeClient as C

    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    c = C(api_key="k", client=httpx.AsyncClient(transport=transport))  # type: ignore[arg-type]
    result = await c.choose_hierarchical(redacted("where is my order"), _taxonomy())

    assert result.intent_id == "order.track_order"
    assert result.path == ("order", "order.track_order")
    assert len(asked) == 2, "one call per level"
    assert len(asked[0]) == 11, "first call offers the categories"
    assert all("." not in k for k in asked[0])


async def test_a_category_with_one_child_costs_no_second_call() -> None:
    """`subscription` has exactly one leaf. Asking a Choice with one option spends a call
    to learn nothing."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    ROUTER_QUESTION: {
                        "type": "choice",
                        "choice": "subscription",
                        "probabilities": {"subscription": 0.95},
                        "confidence": 0.95,
                    }
                },
                "usage": {},
            },
        )

    from ccas.llm.typesafe import TypeSafeClient as C

    calls = {"n": 0}

    async def counting(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return await handler(request)

    transport = httpx.MockTransport(counting)  # type: ignore[arg-type]
    c = C(api_key="k", client=httpx.AsyncClient(transport=transport))  # type: ignore[arg-type]
    result = await c.choose_hierarchical(redacted("cancel my newsletter"), _taxonomy())
    assert result.intent_id == "subscription.newsletter_subscription"
    assert calls["n"] == 1
