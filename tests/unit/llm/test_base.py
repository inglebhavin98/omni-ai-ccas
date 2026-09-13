from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ccas.llm.base import LLMProvider, ProviderUnavailableError
from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, ProviderName


class _Fake(LLMProvider):
    name = ProviderName.VLLM

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(binding=request.binding, text="ok", ttft_ms=10, total_ms=20)

    async def _chunks(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(index=0, text="ok", is_final=True)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        return self._chunks(request)

    async def structured(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(binding=request.binding, parsed={}, ttft_ms=10, total_ms=20)

    async def healthy(self) -> bool:
        return True


def test_the_protocol_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_a_partial_implementation_is_rejected() -> None:
    class _Partial(LLMProvider):
        name = ProviderName.VLLM

        async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(TypeError, match="abstract"):
        _Partial()  # type: ignore[abstract]


async def test_a_conforming_provider_satisfies_every_method() -> None:
    from ccas.schemas.llm import Message, ModelBinding
    from tests.factories import redacted

    request = LLMRequest(
        binding=ModelBinding(node="judge", provider=ProviderName.VLLM, model="m"),
        messages=(Message(role="user", content=redacted("hi")),),
    )
    provider = _Fake()
    assert (await provider.complete(request)).text == "ok"
    assert (await provider.structured(request)).parsed == {}
    assert await provider.healthy()
    assert [c.text async for c in provider.stream(request)] == ["ok"]
    await provider.aclose()


def test_unavailability_is_an_error_not_a_silent_failover() -> None:
    """Rule 6: an unnoticed failover would make the parity eval meaningless."""
    assert issubclass(ProviderUnavailableError, RuntimeError)
