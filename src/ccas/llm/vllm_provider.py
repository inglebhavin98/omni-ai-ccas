"""vLLM binding (OpenAI-compatible ``/v1/chat/completions``).

The second supported provider (CLAUDE.md Rule 6). Uses raw httpx rather than an SDK so
the dependency footprint stays small and any OpenAI-compatible server works.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ccas.llm.base import (
    LLMProvider,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, LLMUsage, ProviderName

__all__ = ["VLLMProvider", "build_payload"]

_DONE = "[DONE]"


def build_payload(request: LLMRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a provider-neutral request into an OpenAI-compatible body.

    Pure and synchronous, for the same reason as the Anthropic binding's shaper.
    """
    binding = request.binding
    messages: list[dict[str, str]] = []
    if request.system is not None:
        messages.append({"role": "system", "content": request.system.require_egress()})
    messages.extend({"role": m.role, "content": m.text_for_egress()} for m in request.messages)

    payload: dict[str, Any] = {
        "model": binding.model,
        "messages": messages,
        "max_tokens": binding.max_tokens,
        "stream": stream,
    }
    if binding.temperature is not None:
        payload["temperature"] = binding.temperature
    if request.tools:
        payload["tools"] = list(request.tools)
    if request.stop_sequences:
        payload["stop"] = list(request.stop_sequences)
    if request.response_schema is not None:
        # vLLM's guided decoding; the equivalent of output_config.format.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": request.response_schema},
        }
    return payload


def _usage_from(raw: dict[str, Any] | None) -> LLMUsage:
    if not raw:
        return LLMUsage()
    return LLMUsage(
        input_tokens=raw.get("prompt_tokens", 0),
        output_tokens=raw.get("completion_tokens", 0),
    )


class VLLMProvider(LLMProvider):
    name = ProviderName.VLLM

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=2.0),
        )

    async def _post(self, payload: dict[str, Any], timeout_ms: int) -> httpx.Response:
        try:
            return await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=timeout_ms / 1000,
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"vLLM unreachable at {self._base_url}") from exc
        except httpx.TimeoutException as exc:
            # There is no retry ladder here, so one deadline is the whole budget. Converted
            # for the same reason OpenRouter converts: a raw httpx error propagating out of
            # a graph node takes the whole call down.
            raise ProviderTimeoutError(f"vLLM did not respond within {timeout_ms}ms") from exc

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = build_payload(request, stream=False)
        started = time.perf_counter()
        response = await self._post(payload, request.binding.timeout_ms)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            raise LLMProviderError(f"vLLM returned {response.status_code}: {response.text[:200]}")

        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message", {})
        return LLMResponse(
            binding=request.binding,
            text=message.get("content") or "",
            tool_calls=tuple(message.get("tool_calls") or ()),
            stop_reason=choice.get("finish_reason"),
            usage=_usage_from(body.get("usage")),
            ttft_ms=elapsed_ms,
            total_ms=elapsed_ms,
        )

    async def _iter(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        payload = build_payload(request, stream=True)
        index = 0
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=request.binding.timeout_ms / 1000,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise LLMProviderError(f"vLLM returned {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == _DONE:
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield LLMChunk(index=index, text=text)
                    index += 1
        yield LLMChunk(index=index, is_final=True)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        return self._iter(request)

    async def structured(self, request: LLMRequest) -> LLMResponse:
        if request.response_schema is None:
            raise LLMProviderError("structured() requires request.response_schema")
        response = await self.complete(request)
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"node {request.binding.node!r} returned non-JSON under guided decoding"
            ) from exc
        return response.model_copy(update={"parsed": parsed})

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
