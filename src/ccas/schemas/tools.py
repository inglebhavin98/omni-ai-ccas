"""Tool contracts (Module 4).

A tool's *shape* is declared in ``domains/<pack>/tools.yaml`` and its *body* lives in a
backend adapter. The orchestrator only ever sees a registered name and a JSON Schema,
so the model can never construct an arbitrary call (CLAUDE.md Rule 4).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ccas.schemas.common import (
    Frozen,
    JsonValue,
    RiskTier,
    Slug,
    TraceContext,
    VerificationLevel,
    utcnow,
)
from ccas.schemas.pii import RedactionReport

__all__ = [
    "RetryPolicy",
    "ToolPayload",
    "ToolRecord",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
]

RetryTrigger = Literal["timeout", "5xx", "conflict", "rate_limit"]


class RetryPolicy(Frozen):
    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_ms: int = Field(default=120, ge=0, le=5_000)
    retry_on: tuple[RetryTrigger, ...] = ("timeout", "5xx")


class ToolSpec(Frozen):
    name: Slug
    description: str = Field(min_length=1)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = None
    domain: Slug
    idempotent: bool = True
    side_effecting: bool = False
    timeout_ms: int = Field(default=1500, ge=50, le=10_000)
    """Bounded by the 150ms tool slice of the latency budget for call-path tools."""

    requires_verification: VerificationLevel = VerificationLevel.NONE
    risk_tier: RiskTier = RiskTier.LOW
    pii_output_fields: tuple[str, ...] = ()
    """Result fields redacted before the payload is shown to any model (Rule 2)."""

    retry_policy: RetryPolicy = RetryPolicy()

    @model_validator(mode="after")
    def _check_schema_is_strict(self) -> Self:
        if self.input_schema.get("type") != "object":
            raise ValueError(f"tool {self.name!r}: input_schema must be an object schema")
        if self.input_schema.get("additionalProperties") is not False:
            raise ValueError(
                f"tool {self.name!r}: input_schema must set additionalProperties=false"
            )
        return self

    @model_validator(mode="after")
    def _check_side_effects(self) -> Self:
        if self.side_effecting and self.retry_policy.max_attempts > 1 and not self.idempotent:
            raise ValueError(
                f"tool {self.name!r}: non-idempotent side-effecting tool must not auto-retry"
            )
        return self

    @property
    def requires_idempotency_key(self) -> bool:
        return self.side_effecting


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"
    NOT_FOUND = "not_found"
    INVALID_ARGS = "invalid_args"


class ToolPayload(Frozen):
    tool_call_id: str = Field(min_length=1, max_length=128)
    tool_name: Slug
    session_id: str = Field(min_length=1, max_length=128)
    intent_id: Slug | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    timeout_ms: int = Field(ge=50, le=10_000)
    verification_level: VerificationLevel = VerificationLevel.NONE
    trace: TraceContext
    issued_at: datetime = Field(default_factory=utcnow)

    def authorized_by(self, spec: ToolSpec) -> bool:
        """Both gates the executor must pass before dispatch."""
        if not self.verification_level.satisfies(spec.requires_verification):
            return False
        return not (spec.requires_idempotency_key and self.idempotency_key is None)


class ToolResult(Frozen):
    tool_call_id: str = Field(min_length=1, max_length=128)
    status: ToolStatus
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error_code: Slug | None = None
    error_message: str | None = None
    elapsed_ms: int = Field(ge=0)
    retryable: bool = False
    redaction: RedactionReport
    """Backend output is untrusted: it must be redacted before a model reads it."""

    @model_validator(mode="after")
    def _check_error_shape(self) -> Self:
        if self.status is ToolStatus.OK and (self.error_code or self.error_message):
            raise ValueError("OK result must not carry an error")
        if self.status is not ToolStatus.OK and self.error_code is None:
            raise ValueError(f"{self.status.value} result requires an error_code")
        return self

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK

    @property
    def safe_for_model(self) -> bool:
        return self.redaction.egress_permitted


class ToolRecord(Frozen):
    """One dispatch attempt, appended to ``SessionState.tool_records``."""

    payload: ToolPayload
    result: ToolResult
    attempt: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def _check_ids_match(self) -> Self:
        if self.payload.tool_call_id != self.result.tool_call_id:
            raise ValueError("payload and result tool_call_id disagree")
        return self
