from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    Effort,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    Message,
    ModelBinding,
    ProviderName,
    RedactionStatus,
)
from tests.factories import redacted


def binding(**kw: object) -> ModelBinding:
    base: dict[str, object] = {
        "node": "router",
        "provider": ProviderName.ANTHROPIC,
        "model": "claude-haiku-4-5",
    }
    base.update(kw)
    return ModelBinding(**base)  # type: ignore[arg-type]


def test_anthropic_binding_rejects_temperature() -> None:
    """Current-generation Anthropic models reject sampling params; effort is the lever."""
    with pytest.raises(ValidationError, match="temperature is not supported"):
        binding(temperature=0.2)


def test_vllm_binding_accepts_temperature() -> None:
    b = binding(provider=ProviderName.VLLM, model="llama-3.3-70b", temperature=0.2)
    assert b.temperature == 0.2


def test_every_supported_provider_is_representable() -> None:
    """Rule 6: no binding may be designed out of the contract."""
    assert {p.value for p in ProviderName} == {"anthropic", "vllm", "openrouter"}


def test_effort_covers_the_full_ladder() -> None:
    assert [e.value for e in Effort] == ["low", "medium", "high", "xhigh", "max"]


def test_requests_reject_an_unredacted_message() -> None:
    with pytest.raises(ValidationError, match="message 0 \\(user\\) is not egress-permitted"):
        LLMRequest(
            binding=binding(),
            messages=(Message(role="user", content=redacted("raw", RedactionStatus.UNVERIFIED)),),
        )


def test_requests_reject_an_unredacted_system_prompt() -> None:
    with pytest.raises(ValidationError, match="system prompt is not egress-permitted"):
        LLMRequest(
            binding=binding(),
            system=redacted("raw", RedactionStatus.DIRTY),
            messages=(Message(role="user", content=redacted("hi")),),
        )


def test_requests_accept_fully_redacted_content() -> None:
    request = LLMRequest(
        binding=binding(),
        system=redacted("You are a support agent."),
        messages=(Message(role="user", content=redacted("Where is [ACCOUNT_REF_1]?")),),
    )
    assert request.messages[0].text_for_egress() == "Where is [ACCOUNT_REF_1]?"


def test_requests_need_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(binding=binding(), messages=())


def test_response_budget_check_uses_the_binding() -> None:
    bound = binding(latency_budget_ms=200)
    assert LLMResponse(binding=bound, ttft_ms=180, total_ms=400).within_budget
    assert not LLMResponse(binding=bound, ttft_ms=250, total_ms=400).within_budget


def test_response_without_a_budget_is_always_within_it() -> None:
    assert LLMResponse(binding=binding(), ttft_ms=9999, total_ms=9999).within_budget


def test_usage_reports_cache_hits() -> None:
    assert LLMUsage(cache_read_tokens=512).cache_hit
    assert not LLMUsage(cache_write_tokens=512).cache_hit
