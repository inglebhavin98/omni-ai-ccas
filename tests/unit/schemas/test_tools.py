from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    RetryPolicy,
    ToolPayload,
    ToolRecord,
    ToolResult,
    ToolSpec,
    ToolStatus,
    TraceContext,
    VerificationLevel,
)
from tests.factories import make_report

STRICT_SCHEMA = {
    "type": "object",
    "properties": {"reference": {"type": "string"}},
    "required": ["reference"],
    "additionalProperties": False,
}


def spec(**kw: object) -> ToolSpec:
    base: dict[str, object] = {
        "name": "get_record_status",
        "description": "d",
        "input_schema": STRICT_SCHEMA,
        "domain": "retail",
    }
    base.update(kw)
    return ToolSpec(**base)  # type: ignore[arg-type]


def payload(trace: TraceContext, **kw: object) -> ToolPayload:
    base: dict[str, object] = {
        "tool_call_id": "tc-1",
        "tool_name": "get_record_status",
        "session_id": "sess-1",
        "timeout_ms": 1500,
        "trace": trace,
    }
    base.update(kw)
    return ToolPayload(**base)  # type: ignore[arg-type]


def result(**kw: object) -> ToolResult:
    base: dict[str, object] = {
        "tool_call_id": "tc-1",
        "status": ToolStatus.OK,
        "elapsed_ms": 42,
        "redaction": make_report(),
    }
    base.update(kw)
    return ToolResult(**base)  # type: ignore[arg-type]


def test_input_schema_must_be_an_object() -> None:
    with pytest.raises(ValidationError, match="must be an object schema"):
        spec(input_schema={"type": "string"})


def test_input_schema_must_forbid_additional_properties() -> None:
    """Open schemas let a model smuggle unvalidated arguments into a backend."""
    with pytest.raises(ValidationError, match="additionalProperties=false"):
        spec(input_schema={"type": "object", "properties": {}})


def test_non_idempotent_side_effecting_tools_must_not_auto_retry() -> None:
    with pytest.raises(ValidationError, match="must not auto-retry"):
        spec(side_effecting=True, idempotent=False, retry_policy=RetryPolicy(max_attempts=3))


def test_side_effecting_tools_require_an_idempotency_key(trace: TraceContext) -> None:
    writer = spec(side_effecting=True)
    assert writer.requires_idempotency_key
    assert not payload(trace).authorized_by(writer)
    assert payload(trace, idempotency_key="idem-0001").authorized_by(writer)


def test_verification_gate_blocks_under_privileged_calls(trace: TraceContext) -> None:
    guarded = spec(requires_verification=VerificationLevel.STRONG)
    assert not payload(trace, verification_level=VerificationLevel.SOFT).authorized_by(guarded)
    assert payload(trace, verification_level=VerificationLevel.STRONG).authorized_by(guarded)
    assert payload(trace, verification_level=VerificationLevel.STEP_UP).authorized_by(guarded)


def test_ok_results_must_not_carry_an_error() -> None:
    with pytest.raises(ValidationError, match="must not carry an error"):
        result(error_code="boom")


def test_failed_results_require_an_error_code() -> None:
    with pytest.raises(ValidationError, match="requires an error_code"):
        result(status=ToolStatus.TIMEOUT)


def test_results_are_only_safe_for_the_model_when_redacted() -> None:
    from ccas.schemas import RedactionStatus

    assert result().safe_for_model
    assert not result(redaction=make_report(RedactionStatus.UNVERIFIED)).safe_for_model


def test_record_rejects_mismatched_call_ids(trace: TraceContext) -> None:
    with pytest.raises(ValidationError, match="tool_call_id disagree"):
        ToolRecord(payload=payload(trace), result=result(tool_call_id="tc-2"))


def test_record_accepts_matching_ids(trace: TraceContext) -> None:
    assert ToolRecord(payload=payload(trace), result=result()).attempt == 1
