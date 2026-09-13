"""Anthropic binding.

One of the two supported providers (CLAUDE.md Rule 6). Request shaping lives in
``_build_kwargs`` so it can be unit-tested without touching the network (Rule 7).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ccas.llm.base import LLMProvider, LLMProviderError, ProviderUnavailableError
from ccas.llm.capabilities import capabilities_for
from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, LLMUsage, ProviderName

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

__all__ = ["AnthropicProvider", "build_kwargs"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Translate a provider-neutral request into Messages API parameters.

    Pure and synchronous on purpose: the shaping rules are the part most likely to
    drift, so they get tested directly rather than through a mock client.
    """
    binding = request.binding
    caps = capabilities_for(binding.model)

    kwargs: dict[str, Any] = {
        "model": binding.model,
        "max_tokens": binding.max_tokens,
        "messages": [{"role": m.role, "content": m.text_for_egress()} for m in request.messages],
    }

    if request.system is not None:
        kwargs["system"] = request.system.require_egress()

    if caps.adaptive_thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    output_config: dict[str, Any] = {}
    if caps.effort:
        output_config["effort"] = binding.effort.value
    if request.response_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": request.response_schema,
        }
    if output_config:
        kwargs["output_config"] = output_config

    if request.tools:
        kwargs["tools"] = list(request.tools)
    if request.stop_sequences:
        kwargs["stop_sequences"] = list(request.stop_sequences)

    # Cache the stable prefix (system + tools); volatile turns sit after it (Rule 3).
    if binding.cache_prefix:
        kwargs["cache_control"] = {"type": "ephemeral"}

    if caps.refusal_fallbacks:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"

    return kwargs


def _usage_from(raw: Any) -> LLMUsage:
    if raw is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


def _text_of(message: Any) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def _tool_calls_of(message: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"id": b.id, "name": b.name, "input": b.input}
        for b in message.content
        if b.type == "tool_use"
    )


class AnthropicProvider(LLMProvider):
    name = ProviderName.ANTHROPIC

    def __init__(self, api_key: str | None = None, client: AsyncAnthropic | None = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
            raise ProviderUnavailableError(
                "the anthropic SDK is not installed; run `uv sync --extra llm`"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    def _endpoint(self, kwargs: dict[str, Any]) -> Any:
        # `betas` and `fallbacks` are only accepted on the beta namespace.
        return self._client.beta.messages if "betas" in kwargs else self._client.messages

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = build_kwargs(request)
        started = time.perf_counter()
        message = await self._endpoint(kwargs).create(**kwargs)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return LLMResponse(
            binding=request.binding,
            text=_text_of(message),
            tool_calls=_tool_calls_of(message),
            stop_reason=message.stop_reason,
            usage=_usage_from(getattr(message, "usage", None)),
            ttft_ms=elapsed_ms,
            total_ms=elapsed_ms,
            refused=message.stop_reason == "refusal",
        )

    async def _iter(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        kwargs = build_kwargs(request)
        index = 0
        async with self._endpoint(kwargs).stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield LLMChunk(index=index, text=text)
                index += 1
            yield LLMChunk(index=index, is_final=True)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        return self._iter(request)

    async def structured(self, request: LLMRequest) -> LLMResponse:
        if request.response_schema is None:
            raise LLMProviderError("structured() requires request.response_schema")
        response = await self.complete(request)
        if response.refused:
            return response
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"node {request.binding.node!r} returned non-JSON under a json_schema format"
            ) from exc
        return response.model_copy(update={"parsed": parsed})

    async def healthy(self) -> bool:
        try:
            await self._client.models.list(limit=1)
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._client.close()
