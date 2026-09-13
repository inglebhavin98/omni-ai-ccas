"""Provider-neutral LLM contracts.

Two bindings ship and stay supported (CLAUDE.md Rule 6), so nothing above ``ccas.llm``
may reference a provider SDK type. Model ids live in ``configs/models.yaml`` and reach
code only through a ``ModelBinding``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, JsonValue, Slug
from ccas.schemas.pii import RedactedText

__all__ = [
    "Effort",
    "LLMChunk",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "Message",
    "ModelBinding",
    "ProviderName",
    "Role",
]


class ProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    """One OpenAI-compatible endpoint fronting many models, including free ones.

    Treated as its own provider rather than a vLLM base-url swap because its request
    surface differs: attribution headers, a ``reasoning`` block, and per-model variation
    in whether ``response_format`` is honoured at all."""


class Effort(StrEnum):
    """Maps to Anthropic ``output_config.effort``; advisory for the vLLM binding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


Role = Literal["system", "user", "assistant"]


class StructuredMode(StrEnum):
    """How a model is asked to return JSON.

    Not every model honours ``response_format``. Three of the four free OpenRouter
    models in use ignore it, so the platform has to ask in the prompt and parse what
    comes back -- which is a different contract, and worth naming.
    """

    RESPONSE_FORMAT = "response_format"
    """Native. The server constrains decoding; the body is valid JSON by construction."""

    PROMPT = "prompt"
    """Instructed. JSON is requested in the prompt and extracted from prose."""


class Message(Frozen):
    role: Role
    content: RedactedText
    """Redacted by construction -- there is no way to build a prompt from raw text."""

    def text_for_egress(self) -> str:
        return self.content.require_egress()


class ModelBinding(Frozen):
    """Which provider and model serve one named call site (a graph node, the judge...)."""

    node: Slug
    provider: ProviderName
    model: str = Field(min_length=1)
    effort: Effort = Effort.HIGH
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = True
    cache_prefix: bool = True
    """Cache the stable system + tool prefix; volatile content goes after it (Rule 3)."""

    timeout_ms: int = Field(default=8_000, ge=100, le=600_000)
    """Transport ceiling, not a latency target. An offline labelling call against a
    reasoning model with a large context legitimately runs for minutes; call-path
    nodes are bounded separately and far more tightly by `latency_budget_ms`."""
    latency_budget_ms: int | None = Field(default=None, ge=1)
    """When set, a TTFT above this is a budget breach for that node."""

    structured_mode: StructuredMode = StructuredMode.RESPONSE_FORMAT
    """Declared per model, because it is a property of the model, not the provider.
    Getting it wrong is silent: a model that ignores ``response_format`` returns prose
    and the parse fails somewhere far away."""

    reasoning: bool = False
    """Ask the model to reason before answering. Free reasoning models are markedly
    better at classification, and markedly more likely to wrap the answer in prose --
    which is why PROMPT mode exists."""

    variant: str = ""
    """Which binding of this node this is. Defaults to the provider name; set explicitly
    when one node has two bindings on the same provider (parity between models)."""

    @model_validator(mode="after")
    def _check_temperature_support(self) -> ModelBinding:
        # Anthropic's current generation rejects sampling params alongside thinking.
        if self.provider is ProviderName.ANTHROPIC and self.temperature is not None:
            raise ValueError(
                f"binding {self.node!r}: temperature is not supported on the "
                "anthropic binding; use effort instead"
            )
        return self


class LLMUsage(Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0


class LLMRequest(Frozen):
    binding: ModelBinding
    system: RedactedText | None = None
    messages: tuple[Message, ...] = Field(min_length=1)
    tools: tuple[dict[str, JsonValue], ...] = ()
    response_schema: dict[str, JsonValue] | None = None
    """When set, the provider must return schema-valid JSON (structured output)."""

    stop_sequences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_all_redacted(self) -> LLMRequest:
        if self.system is not None and not self.system.egress_permitted:
            raise ValueError("system prompt is not egress-permitted")
        for i, message in enumerate(self.messages):
            if not message.content.egress_permitted:
                raise ValueError(f"message {i} ({message.role}) is not egress-permitted")
        return self


class LLMChunk(Frozen):
    """One streamed delta. ``index`` orders chunks within a single response."""

    index: int = Field(ge=0)
    text: str = ""
    tool_call_delta: dict[str, JsonValue] | None = None
    is_final: bool = False


class LLMResponse(Frozen):
    binding: ModelBinding
    text: str = ""
    tool_calls: tuple[dict[str, JsonValue], ...] = ()
    parsed: dict[str, JsonValue] | None = None
    stop_reason: str | None = None
    usage: LLMUsage = LLMUsage()
    ttft_ms: int = Field(ge=0)
    total_ms: int = Field(ge=0)
    refused: bool = False

    @property
    def within_budget(self) -> bool:
        budget = self.binding.latency_budget_ms
        return budget is None or self.ttft_ms <= budget
