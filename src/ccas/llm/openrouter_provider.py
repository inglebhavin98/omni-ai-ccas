"""OpenRouter binding: one OpenAI-compatible endpoint, many models.

Shares the wire protocol with the vLLM binding and differs in three ways that matter:

- **Attribution headers.** OpenRouter asks callers to identify themselves; a free-tier
  account without them is treated less generously.
- **Reasoning.** Models expose a ``reasoning`` block, and return ``reasoning_details``
  that must be passed back unmodified for a model to continue a chain across turns.
- **Structured output is per-model, not per-provider.** Most free models ignore
  ``response_format`` entirely, so the binding declares which mode it is in and falls
  back to asking in the prompt (see ``json_repair``).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ccas.llm.base import (
    LLMProvider,
    LLMProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from ccas.llm.json_repair import JSON_INSTRUCTION, extract_json
from ccas.schemas.common import JsonValue
from ccas.schemas.llm import (
    LLMChunk,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderName,
    StructuredMode,
)

__all__ = ["DEFAULT_BASE_URL", "OpenRouterProvider", "build_payload"]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DONE = "[DONE]"

#: Free-tier models rate-limit routinely, and an upstream provider can be briefly
#: unavailable. Both are transient and worth waiting out; a 400 is not.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def build_payload(request: LLMRequest, *, stream: bool) -> dict[str, Any]:
    """Translate a provider-neutral request into an OpenRouter body. Pure, so the
    shaping rules -- the part most likely to drift -- are tested directly."""
    binding = request.binding
    messages: list[dict[str, Any]] = []

    system = request.system.require_egress() if request.system is not None else ""
    prompt_mode = (
        request.response_schema is not None and binding.structured_mode is StructuredMode.PROMPT
    )
    if prompt_mode:
        # The server will not constrain decoding, so the instruction has to.
        system += JSON_INSTRUCTION.format(schema=json.dumps(request.response_schema, indent=2))
    if system:
        messages.append({"role": "system", "content": system})

    messages.extend({"role": m.role, "content": m.text_for_egress()} for m in request.messages)

    payload: dict[str, Any] = {
        "model": binding.model,
        "messages": messages,
        "max_tokens": binding.max_tokens,
        "stream": stream,
    }
    if binding.temperature is not None:
        payload["temperature"] = binding.temperature
    if binding.reasoning:
        payload["reasoning"] = {"enabled": True}
    if request.tools:
        payload["tools"] = list(request.tools)
    if request.stop_sequences:
        payload["stop"] = list(request.stop_sequences)
    if request.response_schema is not None and not prompt_mode:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": request.response_schema,
            },
        }
    return payload


def request_model(payload: dict[str, Any]) -> str:
    return str(payload.get("model") or "<unknown model>")


def _retry_after_ms(response: httpx.Response) -> int | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return int(float(raw) * 1000)
    except ValueError:
        return None


def _usage_from(raw: dict[str, Any] | None) -> LLMUsage:
    if not raw:
        return LLMUsage()
    return LLMUsage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
    )


class OpenRouterProvider(LLMProvider):
    name = ProviderName.OPENROUTER

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        referer: str = "",
        title: str = "",
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 3,
        backoff_ms: int = 750,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max(1, max_attempts)
        self._backoff_ms = backoff_ms
        self.retries = 0
        """Observed rate limiting, for a readiness report. A free tier that is retrying
        constantly is a capacity signal, not a bug."""
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key or ''}"}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        # Applied whether the client was built here or injected. Identity belongs to the
        # provider, not to whoever supplied the transport -- an injected client for
        # pooling or proxying would otherwise silently lose auth and attribution.
        self._client.headers.update(headers)
        self.last_reasoning_details: JsonValue = None
        """Returned verbatim by the model. Passed back unmodified to continue a chain
        across turns -- editing it invalidates the reasoning it describes."""

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _post(self, payload: dict[str, Any], timeout_ms: int) -> httpx.Response:
        """POST with backoff on the transient failures a free tier actually produces.

        Two of them, and both must be *converted* rather than allowed to escape: a raw
        httpx error propagating out of a graph node takes the whole call down.
        """
        model = str(payload.get("model") or "<unknown model>")
        last: httpx.Response | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    timeout=timeout_ms / 1000,
                )
            except httpx.ConnectError as exc:
                raise ProviderUnavailableError(
                    f"openrouter unreachable at {self._base_url}"
                ) from exc
            except httpx.TimeoutException as exc:
                # Transient and common on a free tier: the upstream provider is queueing.
                self.retries += 1
                if attempt == self._max_attempts:
                    raise LLMProviderError(
                        f"{model} did not respond within {timeout_ms}ms after "
                        f"{self._max_attempts} attempt(s)"
                    ) from exc
                await asyncio.sleep(self._backoff_ms * attempt / 1000)
                continue

            if response.status_code not in _RETRYABLE_STATUS:
                return response

            last = response
            if attempt == self._max_attempts:
                break
            self.retries += 1
            # Honour Retry-After when the server sends one; it knows better than we do.
            delay_ms = _retry_after_ms(response) or self._backoff_ms * attempt
            await asyncio.sleep(delay_ms / 1000)

        assert last is not None
        return last

    def _error_for(self, response: httpx.Response, request: LLMRequest) -> LLMProviderError:
        """Turn a non-2xx into the right exception type.

        429 becomes ``ProviderRateLimitedError`` so callers can tell "nobody served this"
        from "this failed" -- the parity harness needs that distinction to avoid recording
        a free-tier quota as a model disagreement.
        """
        detail = (
            f"openrouter returned {response.status_code} for {request.binding.model!r} "
            f"after {self._max_attempts} attempt(s): {response.text[:240]}"
        )
        if response.status_code == 429:
            return ProviderRateLimitedError(
                f"{detail}  (free-tier capacity; try another model or wait)"
            )
        return LLMProviderError(detail)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OPENROUTER_API_KEY is unset; set it in .env or the environment"
            )
        payload = build_payload(request, stream=False)
        started = time.perf_counter_ns()
        response = await self._post(payload, request.binding.timeout_ms)
        elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)

        if response.status_code >= 400:
            raise self._error_for(response, request)

        body = response.json()
        if "error" in body and not body.get("choices"):
            # A 200 with an error body: upstream model unavailable, rate limited, or
            # moderated. Surfacing it beats returning empty text that looks like an answer.
            raise LLMProviderError(f"openrouter error: {json.dumps(body['error'])[:300]}")

        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        self.last_reasoning_details = message.get("reasoning_details")

        return LLMResponse(
            binding=request.binding,
            text=str(message.get("content") or ""),
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
                raise self._error_for(response, request)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == _DONE:
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError):
                    continue
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

        parsed = extract_json(response.text)
        if parsed is not None:
            return response.model_copy(update={"parsed": parsed})

        binding = request.binding
        if response.stop_reason == "length":
            # Several free models reason inline whatever `reasoning` says, and spend the
            # whole allowance before reaching the answer. Naming that beats "no JSON".
            raise LLMProviderError(
                f"node {binding.node!r} on {binding.model!r} was truncated at "
                f"{binding.max_tokens} tokens before producing JSON "
                f"({response.usage.output_tokens} generated). Raise max_tokens, or use a "
                "model that does not reason inline on a latency-bound node."
            )
        raise LLMProviderError(
            f"node {binding.node!r} on {binding.model!r} "
            f"({binding.structured_mode.value} mode) returned no JSON object: "
            f"{response.text[:200]!r}"
        )

    async def healthy(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/models", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
