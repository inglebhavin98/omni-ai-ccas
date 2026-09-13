"""OpenRouter binding tests over an in-process transport. No network -- Rule 7."""

from __future__ import annotations

import httpx
import pytest

from ccas.llm.base import LLMProviderError, ProviderUnavailableError
from ccas.llm.openrouter_provider import OpenRouterProvider, build_payload
from ccas.llm.prompt import authored
from ccas.schemas.llm import (
    LLMRequest,
    Message,
    ModelBinding,
    ProviderName,
    StructuredMode,
)

SCHEMA = {
    "type": "object",
    "properties": {"intent_id": {"type": "string"}},
    "required": ["intent_id"],
    "additionalProperties": False,
}


def binding(**kw: object) -> ModelBinding:
    base: dict[str, object] = {
        "node": "router",
        "provider": ProviderName.OPENROUTER,
        "model": "vendor/model:free",
    }
    base.update(kw)
    return ModelBinding(**base)  # type: ignore[arg-type]


def request(**kw: object) -> LLMRequest:
    base: dict[str, object] = {
        "binding": binding(),
        "system": authored("You classify calls."),
        "messages": (Message(role="user", content=authored("where is my delivery")),),
    }
    base.update(kw)
    return LLMRequest(**base)  # type: ignore[arg-type]


def provider(handler, **kw: object) -> OpenRouterProvider:
    return OpenRouterProvider(
        api_key="k",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kw,  # type: ignore[arg-type]
    )


def reply(content: str, finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


# ------------------------------------------------------------------- shaping


def test_native_mode_sends_a_response_format() -> None:
    payload = build_payload(
        request(
            binding=binding(structured_mode=StructuredMode.RESPONSE_FORMAT), response_schema=SCHEMA
        ),
        stream=False,
    )
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_prompt_mode_instructs_instead_of_constraining() -> None:
    """Most free models ignore response_format entirely."""
    payload = build_payload(
        request(binding=binding(structured_mode=StructuredMode.PROMPT), response_schema=SCHEMA),
        stream=False,
    )
    assert "response_format" not in payload
    system = payload["messages"][0]["content"]
    assert "OUTPUT FORMAT" in system
    assert "intent_id" in system, "the schema itself must reach the model"


def test_no_schema_means_no_json_machinery_at_all() -> None:
    payload = build_payload(request(), stream=False)
    assert "response_format" not in payload
    assert "OUTPUT FORMAT" not in payload["messages"][0]["content"]


def test_reasoning_is_requested_only_when_asked_for() -> None:
    assert "reasoning" not in build_payload(request(), stream=False)
    payload = build_payload(request(binding=binding(reasoning=True)), stream=False)
    assert payload["reasoning"] == {"enabled": True}


def test_temperature_is_forwarded() -> None:
    payload = build_payload(request(binding=binding(temperature=0.3)), stream=False)
    assert payload["temperature"] == 0.3


def test_attribution_headers_are_sent() -> None:
    """OpenRouter treats an unattributed free-tier account less generously."""
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(req.headers)
        return httpx.Response(200, json=reply("{}"))

    p = OpenRouterProvider(
        api_key="k",
        referer="https://example.test",
        title="omni-ai-ccas",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(p.complete(request()))
    assert seen["http-referer"] == "https://example.test"
    assert seen["x-title"] == "omni-ai-ccas"


# ------------------------------------------------------------------ responses


async def test_a_reply_is_mapped() -> None:
    p = provider(lambda _: httpx.Response(200, json=reply("hello")))
    response = await p.complete(request())
    assert response.text == "hello"
    assert response.usage.output_tokens == 5


async def test_reasoning_details_are_retained_for_the_next_turn() -> None:
    """They must be passed back unmodified for a model to continue a chain."""
    body = reply("answer")
    body["choices"][0]["message"]["reasoning_details"] = [{"type": "thought", "id": "x"}]  # type: ignore[index]
    p = provider(lambda _: httpx.Response(200, json=body))
    await p.complete(request())
    assert p.last_reasoning_details == [{"type": "thought", "id": "x"}]


async def test_a_200_carrying_an_error_body_is_raised() -> None:
    """Returning empty text would look like an answer."""
    p = provider(lambda _: httpx.Response(200, json={"error": {"message": "moderated"}}))
    with pytest.raises(LLMProviderError, match="openrouter error"):
        await p.complete(request())


async def test_missing_credentials_are_reported_not_attempted() -> None:
    p = OpenRouterProvider(api_key=None)
    with pytest.raises(ProviderUnavailableError, match="OPENROUTER_API_KEY"):
        await p.complete(request())
    await p.aclose()


# ----------------------------------------------------------------- structured


async def test_fenced_json_is_accepted_in_prompt_mode() -> None:
    p = provider(lambda _: httpx.Response(200, json=reply('```json\n{"intent_id": "a.b.c"}\n```')))
    response = await p.structured(
        request(binding=binding(structured_mode=StructuredMode.PROMPT), response_schema=SCHEMA)
    )
    assert response.parsed == {"intent_id": "a.b.c"}


async def test_truncation_is_named_rather_than_reported_as_missing_json() -> None:
    """Several free models reason inline whatever `reasoning` says, and spend the whole
    allowance before reaching the answer. "No JSON" sends you hunting."""
    p = provider(
        lambda _: httpx.Response(
            200, json=reply("Here's a thinking process: first I will", finish="length")
        )
    )
    with pytest.raises(LLMProviderError, match="truncated at"):
        await p.structured(request(response_schema=SCHEMA))


async def test_unparseable_output_names_the_model_and_the_mode() -> None:
    p = provider(lambda _: httpx.Response(200, json=reply("I cannot help with that.")))
    with pytest.raises(LLMProviderError, match="returned no JSON object"):
        await p.structured(request(response_schema=SCHEMA))


async def test_structured_requires_a_schema() -> None:
    p = provider(lambda _: httpx.Response(200, json=reply("{}")))
    with pytest.raises(LLMProviderError, match=r"requires request\.response_schema"):
        await p.structured(request())


# --------------------------------------------------------------------- retry


async def test_rate_limiting_is_retried() -> None:
    """Free-tier models 429 routinely; a single attempt would be unusable."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {}})
        return httpx.Response(200, json=reply("ok"))

    p = provider(handler, max_attempts=3, backoff_ms=1)
    assert (await p.complete(request())).text == "ok"
    assert p.retries == 2


async def test_a_read_timeout_never_escapes_as_an_httpx_error() -> None:
    """A raw httpx error propagating out of a graph node takes the whole call down."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    p = provider(handler, max_attempts=2, backoff_ms=1)
    with pytest.raises(LLMProviderError, match="did not respond within"):
        await p.complete(request())


async def test_a_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {}})

    p = provider(handler, max_attempts=3, backoff_ms=1)
    with pytest.raises(LLMProviderError, match="openrouter returned 400"):
        await p.complete(request())
    assert calls["n"] == 1


async def test_retry_after_is_honoured() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0.001"}, json={"error": {}})

    p = provider(handler, max_attempts=2, backoff_ms=10_000)
    with pytest.raises(LLMProviderError):
        await p.complete(request())


async def test_an_unreachable_host_is_unavailable_not_an_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError, match="unreachable"):
        await provider(handler).complete(request())
