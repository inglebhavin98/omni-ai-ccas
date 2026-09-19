"""vLLM binding tests over an in-process transport. No network -- Rule 7."""

from __future__ import annotations

import json

import httpx
import pytest

from ccas.llm.base import (
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ccas.llm.vllm_provider import VLLMProvider, build_payload
from ccas.schemas.llm import LLMRequest, Message, ModelBinding, ProviderName
from tests.factories import redacted

BASE_URL = "http://vllm.test/v1"
SCHEMA = {"type": "object", "properties": {"intent_id": {"type": "string"}}}


def binding(**kw: object) -> ModelBinding:
    base: dict[str, object] = {
        "node": "task_agent",
        "provider": ProviderName.VLLM,
        "model": "llama-3.3-70b-instruct",
    }
    base.update(kw)
    return ModelBinding(**base)  # type: ignore[arg-type]


def request(**kw: object) -> LLMRequest:
    base: dict[str, object] = {
        "binding": binding(),
        "messages": (Message(role="user", content=redacted("Where is my parcel?")),),
    }
    base.update(kw)
    return LLMRequest(**base)  # type: ignore[arg-type]


def provider(handler) -> VLLMProvider:
    return VLLMProvider(BASE_URL, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def completion(content: str = "Your parcel is out for delivery.") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }


def test_system_prompt_becomes_the_first_message() -> None:
    payload = build_payload(request(system=redacted("You are an agent.")), stream=False)
    assert payload["messages"][0] == {"role": "system", "content": "You are an agent."}
    assert payload["messages"][1]["role"] == "user"


def test_temperature_is_forwarded_when_the_binding_sets_one() -> None:
    payload = build_payload(request(binding=binding(temperature=0.3)), stream=False)
    assert payload["temperature"] == 0.3


def test_temperature_is_omitted_when_unset() -> None:
    assert "temperature" not in build_payload(request(), stream=False)


def test_response_schema_becomes_guided_decoding() -> None:
    payload = build_payload(request(response_schema=SCHEMA), stream=False)
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA


def test_stream_flag_is_explicit() -> None:
    assert build_payload(request(), stream=True)["stream"] is True
    assert build_payload(request(), stream=False)["stream"] is False


async def test_complete_maps_the_openai_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion())

    response = await provider(handler).complete(request())
    assert response.text == "Your parcel is out for delivery."
    assert response.stop_reason == "stop"
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 18


async def test_an_error_status_raises_rather_than_returning_empty_text() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not loaded")

    with pytest.raises(LLMProviderError, match="vLLM returned 500"):
        await provider(handler).complete(request())


async def test_an_unreachable_server_is_reported_not_silently_failed_over() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError, match="unreachable"):
        await provider(handler).complete(request())


async def test_streaming_yields_text_then_a_final_chunk() -> None:
    frames = [
        f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n"
        for part in ("Your ", "parcel ", "shipped.")
    ]
    frames.append("data: [DONE]\n")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="".join(frames))

    chunks = [c async for c in provider(handler).stream(request())]
    assert "".join(c.text for c in chunks) == "Your parcel shipped."
    assert chunks[-1].is_final
    assert [c.index for c in chunks] == [0, 1, 2, 3]


async def test_structured_parses_the_guided_output() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(json.dumps({"intent_id": "billing"})))

    response = await provider(handler).structured(request(response_schema=SCHEMA))
    assert response.parsed == {"intent_id": "billing"}


async def test_structured_requires_a_schema() -> None:
    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json=completion())

    with pytest.raises(LLMProviderError, match=r"requires request\.response_schema"):
        await provider(handler).structured(request())


async def test_non_json_under_guided_decoding_is_an_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("not json at all"))

    with pytest.raises(LLMProviderError, match="returned non-JSON"):
        await provider(handler).structured(request(response_schema=SCHEMA))


async def test_health_probe_reports_reachability() -> None:
    def up(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert await provider(up).healthy()
    assert not await provider(down).healthy()


async def test_a_read_timeout_never_escapes_as_an_httpx_error() -> None:
    """The same contract the OpenRouter binding holds: a raw httpx error propagating out
    of a graph node takes the whole call down, so every binding must convert it."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        await provider(handler).complete(request())
