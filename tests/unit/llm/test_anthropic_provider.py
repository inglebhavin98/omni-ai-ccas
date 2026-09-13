"""Request-shaping tests. No network -- CLAUDE.md Rule 7."""

from __future__ import annotations

import pytest

from ccas.llm.anthropic_provider import FALLBACK_BETA, build_kwargs
from ccas.llm.capabilities import capabilities_for
from ccas.schemas.llm import Effort, LLMRequest, Message, ModelBinding, ProviderName
from ccas.schemas.pii import RedactionStatus
from tests.factories import redacted

SCHEMA = {
    "type": "object",
    "properties": {"intent_id": {"type": "string"}},
    "required": ["intent_id"],
    "additionalProperties": False,
}


def request(model: str = "claude-opus-5", **kw: object) -> LLMRequest:
    base: dict[str, object] = {
        "binding": ModelBinding(node="task_agent", provider=ProviderName.ANTHROPIC, model=model),
        "messages": (Message(role="user", content=redacted("Where is my parcel?")),),
    }
    base.update(kw)
    return LLMRequest(**base)  # type: ignore[arg-type]


def test_opus_gets_adaptive_thinking_and_effort() -> None:
    kwargs = build_kwargs(request("claude-opus-5"))
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "high"


def test_haiku_omits_thinking_and_effort() -> None:
    """Both parameters error on Haiku 4.5; the router must not send them."""
    kwargs = build_kwargs(request("claude-haiku-4-5"))
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


def test_effort_follows_the_binding() -> None:
    binding = ModelBinding(
        node="router",
        provider=ProviderName.ANTHROPIC,
        model="claude-opus-5",
        effort=Effort.LOW,
    )
    kwargs = build_kwargs(request(binding=binding))
    assert kwargs["output_config"]["effort"] == "low"


def test_unknown_models_get_the_conservative_shape() -> None:
    """A future model id must not be guessed at by prefix matching."""
    kwargs = build_kwargs(request("claude-something-9"))
    assert "thinking" not in kwargs
    assert "betas" not in kwargs


def test_opus_5_enables_server_side_refusal_fallbacks() -> None:
    kwargs = build_kwargs(request("claude-opus-5"))
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"


def test_haiku_does_not_request_fallbacks() -> None:
    assert "betas" not in build_kwargs(request("claude-haiku-4-5"))


def test_prefix_caching_is_on_by_default() -> None:
    assert build_kwargs(request())["cache_control"] == {"type": "ephemeral"}


def test_prefix_caching_can_be_disabled_per_binding() -> None:
    binding = ModelBinding(
        node="judge",
        provider=ProviderName.ANTHROPIC,
        model="claude-haiku-4-5",
        cache_prefix=False,
    )
    assert "cache_control" not in build_kwargs(request(binding=binding))


def test_response_schema_becomes_a_json_schema_format() -> None:
    kwargs = build_kwargs(request(response_schema=SCHEMA))
    assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}


def test_system_prompt_is_passed_through_when_redacted() -> None:
    kwargs = build_kwargs(request(system=redacted("You are a support agent.")))
    assert kwargs["system"] == "You are a support agent."


def test_shaping_cannot_smuggle_unredacted_text() -> None:
    """The egress gate sits in the contract, so the shaper never sees raw text."""
    with pytest.raises(Exception, match="egress-permitted"):
        request(system=redacted("raw", RedactionStatus.UNVERIFIED))


def test_no_prefill_is_ever_sent() -> None:
    """Assistant prefill returns a 400 on every current model."""
    kwargs = build_kwargs(request())
    assert kwargs["messages"][-1]["role"] == "user"


@pytest.mark.parametrize(
    ("model", "thinking", "effort", "fallbacks"),
    [
        ("claude-opus-5", True, True, True),
        ("claude-sonnet-5", True, True, False),
        ("claude-haiku-4-5", False, False, False),
    ],
)
def test_capability_table_matches_the_api(
    model: str, thinking: bool, effort: bool, fallbacks: bool
) -> None:
    caps = capabilities_for(model)
    assert caps.adaptive_thinking is thinking
    assert caps.effort is effort
    assert caps.refusal_fallbacks is fallbacks
