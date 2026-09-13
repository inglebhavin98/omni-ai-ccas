"""Tool execution: the gate chain between a model's choice and a real system.

Every guarantee lives here rather than in a backend, so adding a backend cannot weaken
one. In order, and each step can terminate without dispatch:

    resolve -> validate -> authorise -> dispatch(timeout) -> retry -> redact -> record

The redaction step is the one most easily got wrong. A ``ToolSpec`` declares
``pii_output_fields``, but that trusts whoever wrote the spec to have anticipated every
field a backend might return. So declared fields are force-redacted *and* every
remaining string is passed through the redaction pipeline. A result that does not come
back clean is returned as an error with its data withheld -- the same rule ingestion
follows (CLAUDE.md Rule 2).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import jsonschema
from jsonschema import Draft202012Validator

from ccas.redaction.pipeline import RedactionPipeline
from ccas.redaction.vault import PlaceholderVault
from ccas.schemas.common import JsonValue, TraceContext, VerificationLevel
from ccas.schemas.pii import (
    PiiEntityType,
    RedactionReport,
    RedactionSpan,
    RedactionStatus,
    sha256_hex,
)
from ccas.schemas.tools import ToolPayload, ToolRecord, ToolResult, ToolSpec, ToolStatus
from ccas.tools.backend import BackendResponse, ToolBackend
from ccas.tools.registry import ToolNotRegisteredError, ToolRegistry

__all__ = ["ToolExecutor", "mint_idempotency_key"]

#: Which declared retry trigger a failure counts as.
_TRIGGER_FOR: dict[ToolStatus, str] = {
    ToolStatus.TIMEOUT: "timeout",
    ToolStatus.ERROR: "5xx",
}


def mint_idempotency_key(session_id: str, tool_name: str, arguments: dict[str, JsonValue]) -> str:
    """Deterministic in the arguments, so a retry reuses the key and a genuinely new
    request gets a different one."""
    canonical = "|".join(f"{k}={arguments[k]!r}" for k in sorted(arguments))
    digest: str = sha256_hex(f"{session_id}|{tool_name}|{canonical}")
    return digest[:32]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        backend: ToolBackend,
        redaction: RedactionPipeline,
        total_budget_ms: int | None = None,
        vault: PlaceholderVault | None = None,
    ) -> None:
        self._registry = registry
        self._backend = backend
        self._redaction = redaction
        self._vault = vault
        """Session vault. A registered backend is an internal system of record, not an
        external boundary -- it needs the reference the caller actually read out, not
        the placeholder a model sees (docs/adr/0010-placeholder-vault.md)."""
        self._total_budget_ms = total_budget_ms
        """Wall-clock ceiling across *all* attempts. Without it a two-attempt retry on a
        900ms tool spends 1.9s against a 150ms slice -- each attempt is within its own
        timeout while the turn blows its budget. Usually the `tool_ms` stage budget."""

    # ------------------------------------------------------------------ payloads

    def build_payload(
        self,
        tool_name: str,
        *,
        session_id: str,
        trace: TraceContext,
        arguments: dict[str, JsonValue],
        intent_id: str | None = None,
        verification: VerificationLevel = VerificationLevel.NONE,
        call_index: int = 0,
    ) -> ToolPayload:
        """Mint a payload, including an idempotency key where the spec requires one."""
        spec = self._registry.resolve(tool_name)
        key = (
            mint_idempotency_key(session_id, tool_name, arguments)
            if spec.requires_idempotency_key
            else None
        )
        return ToolPayload(
            tool_call_id=f"{session_id}-{call_index}-{tool_name}"[:128],
            tool_name=tool_name,
            session_id=session_id,
            intent_id=intent_id,
            arguments=arguments,
            idempotency_key=key,
            timeout_ms=spec.timeout_ms,
            verification_level=verification,
            trace=trace,
        )

    # ------------------------------------------------------------------ execution

    async def execute(self, payload: ToolPayload) -> ToolRecord:
        started = time.perf_counter_ns()

        try:
            spec = self._registry.resolve(payload.tool_name)
        except ToolNotRegisteredError as exc:
            return self._refuse(
                payload, ToolStatus.NOT_FOUND, "tool_not_registered", str(exc), started
            )

        # Restore placeholders *before* validating: a schema pattern like
        # ^[A-Z0-9-]{6,20}$ will never match "[ACCOUNT_REF_1]", so validating the masked
        # form would reject every reference the caller actually gave us.
        dispatch_payload, masked = self._detokenize(payload)
        if masked:
            return self._refuse(
                payload,
                ToolStatus.INVALID_ARGS,
                "unresolved_placeholder",
                f"arguments still masked: {', '.join(masked)}",
                started,
                attempt=1,
            )

        invalid = _validate_arguments(spec, dispatch_payload)
        if invalid is not None:
            return self._refuse(
                payload,
                ToolStatus.INVALID_ARGS,
                "schema_violation",
                invalid,
                started,
                attempt=1,
            )

        if not payload.authorized_by(spec):
            return self._refuse(
                payload,
                ToolStatus.DENIED,
                "not_authorized",
                _denial_reason(spec, payload),
                started,
                attempt=1,
            )

        response, attempts = await self._dispatch_with_retry(spec, dispatch_payload, started)
        elapsed_ms = _elapsed_ms(started)

        if not response.ok:
            return ToolRecord(
                payload=payload,
                result=ToolResult(
                    tool_call_id=payload.tool_call_id,
                    status=response.status,
                    error_code=response.error_code or "backend_error",
                    error_message=response.error_message,
                    elapsed_ms=elapsed_ms,
                    retryable=response.retryable,
                    redaction=_clean_report(elapsed_ms),
                ),
                attempt=attempts,
            )

        data, report = self._redact_output(spec, response.data, elapsed_ms)
        if not report.egress_permitted:
            # Withhold rather than emit: a half-redacted result that looks normal is the
            # worst artifact this layer could produce.
            return ToolRecord(
                payload=payload,
                result=ToolResult(
                    tool_call_id=payload.tool_call_id,
                    status=ToolStatus.ERROR,
                    error_code="redaction_failed",
                    error_message=f"tool output not egress-permitted: {report.status.value}",
                    elapsed_ms=elapsed_ms,
                    retryable=False,
                    redaction=report,
                ),
                attempt=attempts,
            )

        return ToolRecord(
            payload=payload,
            result=ToolResult(
                tool_call_id=payload.tool_call_id,
                status=ToolStatus.OK,
                data=data,
                elapsed_ms=elapsed_ms,
                redaction=report,
            ),
            attempt=attempts,
        )

    def _detokenize(self, payload: ToolPayload) -> tuple[ToolPayload, tuple[str, ...]]:
        """Restore placeholders in string arguments, or report what could not be."""
        if self._vault is None:
            return payload, ()

        masked: list[str] = []
        restored: dict[str, JsonValue] = {}
        for key, value in payload.arguments.items():
            if not isinstance(value, str):
                restored[key] = value
                continue
            masked.extend(self._vault.contains_unknown(value))
            restored[key] = self._vault.detokenize(value)

        if masked:
            return payload, tuple(sorted(set(masked)))
        if restored == payload.arguments:
            return payload, ()
        # A copy, so the recorded payload keeps the masked form: a ToolRecord ends up in
        # a HandoffContext, which must never carry the original.
        return payload.model_copy(update={"arguments": restored}), ()

    async def execute_many(self, payloads: Sequence[ToolPayload]) -> tuple[ToolRecord, ...]:
        """Independent calls run concurrently so parallel tool use fits the 150 ms slice."""
        if not payloads:
            return ()
        return tuple(await asyncio.gather(*(self.execute(p) for p in payloads)))

    # ------------------------------------------------------------------ internals

    async def _dispatch_with_retry(
        self, spec: ToolSpec, payload: ToolPayload, started_ns: int
    ) -> tuple[BackendResponse, int]:
        policy = spec.retry_policy
        response = BackendResponse(status=ToolStatus.ERROR, error_code="not_attempted")

        for attempt in range(1, policy.max_attempts + 1):
            remaining_ms = self._remaining_ms(started_ns, payload.timeout_ms)
            if remaining_ms <= 0:
                return _budget_exhausted(spec, self._total_budget_ms), attempt - 1 or 1

            response = await self._dispatch_once(spec, payload, remaining_ms)
            if response.ok:
                return response, attempt

            trigger = _TRIGGER_FOR.get(response.status)
            retryable = trigger in policy.retry_on and (
                response.retryable or response.status is ToolStatus.TIMEOUT
            )
            if not retryable or attempt == policy.max_attempts:
                return response, attempt

            backoff_ms = policy.backoff_ms * attempt
            if self._remaining_ms(started_ns, payload.timeout_ms) <= backoff_ms:
                # Not enough budget left to be worth another try.
                return response, attempt
            await asyncio.sleep(backoff_ms / 1000)
        return response, policy.max_attempts

    def _remaining_ms(self, started_ns: int, per_attempt_ms: int) -> int:
        """How long the next attempt may take: the shorter of its own timeout and
        whatever is left of the overall budget."""
        if self._total_budget_ms is None:
            return per_attempt_ms
        spent = _elapsed_ms(started_ns)
        return min(per_attempt_ms, self._total_budget_ms - spent)

    async def _dispatch_once(
        self, spec: ToolSpec, payload: ToolPayload, timeout_ms: int
    ) -> BackendResponse:
        try:
            return await asyncio.wait_for(
                self._backend.invoke(spec, payload), timeout=timeout_ms / 1000
            )
        except TimeoutError:
            return BackendResponse(
                status=ToolStatus.TIMEOUT,
                error_code="timeout",
                error_message=(
                    f"{spec.name} exceeded {timeout_ms}ms"
                    + (
                        " (shortened by the remaining tool budget)"
                        if timeout_ms < payload.timeout_ms
                        else ""
                    )
                ),
                retryable=True,
            )
        except Exception as exc:  # a backend defect must not take the call down
            return BackendResponse(
                status=ToolStatus.ERROR,
                error_code="backend_exception",
                error_message=f"{type(exc).__name__}",
                retryable=False,
            )

    def _redact_output(
        self, spec: ToolSpec, data: dict[str, JsonValue], elapsed_ms: int
    ) -> tuple[dict[str, JsonValue], RedactionReport]:
        allocator = self._redaction.new_allocator()
        spans: list[RedactionSpan] = []
        residual: list[str] = []
        declared = frozenset(spec.pii_output_fields)

        def walk(value: JsonValue, field_path: str) -> JsonValue:
            if isinstance(value, dict):
                return {k: walk(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v, field_path) for v in value]
            if not isinstance(value, str) or not value:
                return value

            if field_path in declared:
                # Declared PII: replaced wholesale, whatever it looks like. A name with
                # no lexical shape would survive a pattern scan.
                token = allocator.allocate(PiiEntityType.CUSTOM, value, custom_label=field_path)
                spans.append(
                    RedactionSpan(
                        start=0,
                        end=len(value),
                        entity_type=PiiEntityType.CUSTOM,
                        custom_label=field_path,
                        score=1.0,
                        engine="manual",
                        replacement=token,
                    )
                )
                return token

            result = self._redaction.redact(value, allocator)
            if not result.report.egress_permitted:
                residual.extend(result.report.residual_patterns or ("unredactable",))
                return ""
            spans.extend(result.report.spans)
            return result.text

        redacted = {key: walk(value, key) for key, value in data.items()}

        if residual:
            return {}, RedactionReport(
                status=RedactionStatus.DIRTY,
                spans=tuple(spans),
                engines_run=("manual", "regex"),
                policy_version=self._redaction.policy.version,
                elapsed_us=elapsed_ms * 1000,
                leak_check_passed=False,
                residual_patterns=tuple(sorted(set(residual))),
            )

        return redacted, RedactionReport.clean(
            spans=tuple(spans),
            engines_run=("manual", "regex"),
            policy_version=self._redaction.policy.version,
            elapsed_us=elapsed_ms * 1000,
        )

    def _refuse(
        self,
        payload: ToolPayload,
        status: ToolStatus,
        error_code: str,
        message: str,
        started: int,
        attempt: int = 1,
    ) -> ToolRecord:
        elapsed_ms = _elapsed_ms(started)
        return ToolRecord(
            payload=payload,
            result=ToolResult(
                tool_call_id=payload.tool_call_id,
                status=status,
                error_code=error_code,
                error_message=message,
                elapsed_ms=elapsed_ms,
                retryable=False,
                redaction=_clean_report(elapsed_ms),
            ),
            attempt=attempt,
        )


def _budget_exhausted(spec: ToolSpec, budget_ms: int | None) -> BackendResponse:
    return BackendResponse(
        status=ToolStatus.TIMEOUT,
        error_code="budget_exhausted",
        error_message=f"{spec.name}: {budget_ms}ms tool budget spent before completing",
        retryable=False,
    )


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)


def _clean_report(elapsed_ms: int) -> RedactionReport:
    """A result with no data carries nothing to redact, but still needs a report."""
    return RedactionReport.clean(
        engines_run=("manual",), policy_version="tool-executor", elapsed_us=elapsed_ms * 1000
    )


def _validate_arguments(spec: ToolSpec, payload: ToolPayload) -> str | None:
    """Return a message when the arguments violate the tool's schema."""
    try:
        Draft202012Validator(spec.input_schema).validate(payload.arguments)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(p) for p in exc.absolute_path) or "<root>"
        return f"{location}: {exc.message}"
    except jsonschema.SchemaError as exc:
        return f"tool {spec.name!r} has an invalid input_schema: {exc.message}"
    return None


def _denial_reason(spec: ToolSpec, payload: ToolPayload) -> str:
    if not payload.verification_level.satisfies(spec.requires_verification):
        return (
            f"{spec.name} requires {spec.requires_verification.value} verification, "
            f"caller is {payload.verification_level.value}"
        )
    return f"{spec.name} is side-effecting and requires an idempotency key"
