from __future__ import annotations

from pathlib import Path

from ccas.config.domain_loader import load_pack
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.schemas.common import JsonValue, TraceContext, VerificationLevel
from ccas.schemas.tools import ToolPayload, ToolStatus
from ccas.tools.backend import BackendResponse, ToolBackend
from ccas.tools.executor import ToolExecutor, mint_idempotency_key
from ccas.tools.mock_backend import MockToolBackend
from ccas.tools.registry import build_registry

REPO = Path(__file__).resolve().parents[3]
DOMAINS = REPO / "domains"
GLOBAL = REPO / "configs" / "tools.yaml"
POLICY = REPO / "configs" / "redaction_policy.yaml"

TRACE = TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id="corr-1")


def make_executor(
    backend: ToolBackend | None = None, total_budget_ms: int | None = None
) -> ToolExecutor:
    pack = load_pack(DOMAINS, "retail")
    return ToolExecutor(
        build_registry(pack, GLOBAL),
        backend or MockToolBackend.from_pack_dir(DOMAINS / "retail"),
        build_pipeline(POLICY, pack=pack, presidio=None, mode=RedactionMode.REALTIME),
        total_budget_ms=total_budget_ms,
    )


def payload(
    executor: ToolExecutor,
    tool: str,
    arguments: dict[str, JsonValue],
    verification: VerificationLevel = VerificationLevel.STRONG,
) -> ToolPayload:
    return executor.build_payload(
        tool, session_id="s1", trace=TRACE, arguments=arguments, verification=verification
    )


# ---------------------------------------------------------------- gate chain


async def test_an_unregistered_tool_never_reaches_a_backend() -> None:
    backend = MockToolBackend()
    executor = make_executor(backend)
    record = await executor.execute(
        ToolPayload(
            tool_call_id="tc-1",
            tool_name="no_such_tool",
            session_id="s1",
            timeout_ms=500,
            trace=TRACE,
        )
    )
    assert record.result.status is ToolStatus.NOT_FOUND
    assert backend.calls == []


async def test_invalid_arguments_never_reach_a_backend() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    executor = make_executor(backend)
    record = await executor.execute(payload(executor, "get_order_status", {"wrong": "x"}))
    assert record.result.status is ToolStatus.INVALID_ARGS
    assert "order_reference" in (record.result.error_message or "")
    assert backend.calls == []


async def test_a_schema_pattern_violation_is_caught() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "not a reference!"})
    )
    assert record.result.status is ToolStatus.INVALID_ARGS


async def test_under_verification_is_denied_before_dispatch() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    executor = make_executor(backend)
    record = await executor.execute(
        payload(
            executor,
            "update_delivery_address",
            {
                "order_reference": "ORD-1",
                "address_line_1": "a",
                "postal_code": "p",
                "country": "c",
            },
            verification=VerificationLevel.SOFT,
        )
    )
    assert record.result.status is ToolStatus.DENIED
    assert "requires strong verification" in (record.result.error_message or "")
    assert backend.calls == []


async def test_a_side_effecting_call_without_a_key_is_denied() -> None:
    executor = make_executor()
    record = await executor.execute(
        ToolPayload(
            tool_call_id="tc-1",
            tool_name="start_return",
            session_id="s1",
            arguments={"order_reference": "ORD-1", "line_item_id": "1", "reason_code": "damaged"},
            timeout_ms=1500,
            verification_level=VerificationLevel.STRONG,
            trace=TRACE,
        )
    )
    assert record.result.status is ToolStatus.DENIED
    assert "idempotency key" in (record.result.error_message or "")


def test_the_executor_mints_a_key_for_side_effecting_tools() -> None:
    executor = make_executor()
    built = payload(
        executor,
        "start_return",
        {"order_reference": "ORD-1", "line_item_id": "1", "reason_code": "damaged"},
    )
    assert built.idempotency_key is not None


def test_read_only_tools_get_no_key() -> None:
    executor = make_executor()
    assert (
        payload(executor, "get_order_status", {"order_reference": "ORD-1"}).idempotency_key is None
    )


def test_idempotency_keys_are_stable_in_the_arguments() -> None:
    """A retry must reuse the key; a genuinely new request must not."""
    first = mint_idempotency_key("s1", "start_return", {"a": 1, "b": 2})
    assert first == mint_idempotency_key("s1", "start_return", {"b": 2, "a": 1})
    assert first != mint_idempotency_key("s1", "start_return", {"a": 1, "b": 3})
    assert first != mint_idempotency_key("s2", "start_return", {"a": 1, "b": 2})


# ------------------------------------------------------------------- results


async def test_a_successful_call_returns_data() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-884210"})
    )
    assert record.result.ok
    assert record.result.data["status"] == "in_transit"
    assert record.attempt == 1


async def test_declared_pii_fields_are_replaced_wholesale() -> None:
    """A name has no lexical shape; a pattern scan alone would leave it."""
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-884210"})
    )
    assert record.result.data["recipient_name"] == "[RECIPIENT_NAME_1]"
    assert "Dana" not in record.result.model_dump_json()


async def test_undeclared_fields_still_pass_through_redaction() -> None:
    """The spec author cannot anticipate every field a backend returns."""
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-884210"})
    )
    note = record.result.data["contact_note"]
    assert isinstance(note, str)
    assert "415-555-0142" not in note
    assert "[PHONE_1]" in note


async def test_a_delivery_date_is_not_treated_as_a_birth_date() -> None:
    """Over-redaction makes tool output useless to the model and to the agent."""
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-884210"})
    )
    assert record.result.data["eta"] == "2026-09-18"


async def test_non_string_values_are_left_alone() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "transfer_to_human", {"queue": "tier-1", "reason": "asked"})
    )
    assert record.result.data["position"] == 3
    assert record.result.data["queued"] is True


async def test_results_are_marked_safe_for_the_model() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-884210"})
    )
    assert record.result.safe_for_model


# ------------------------------------------------------------------ failures


async def test_a_backend_error_is_surfaced_not_swallowed() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-404404"})
    )
    assert record.result.status is ToolStatus.NOT_FOUND
    assert record.result.error_code == "unknown_reference"


async def test_a_retryable_failure_is_retried() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    executor = make_executor(backend)
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-500500"})
    )
    assert record.attempt == 2
    assert len(backend.calls) == 2


async def test_a_non_retryable_failure_is_not_retried() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    executor = make_executor(backend)
    await executor.execute(payload(executor, "get_order_status", {"order_reference": "ORD-404404"}))
    assert len(backend.calls) == 1


async def test_a_slow_backend_times_out() -> None:
    executor = make_executor()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-999999"})
    )
    assert record.result.status is ToolStatus.TIMEOUT


async def test_retries_cannot_outrun_the_total_budget() -> None:
    """Each attempt within its own timeout while the turn blows its slice."""
    import time

    executor = make_executor(total_budget_ms=150)
    started = time.perf_counter()
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-999999"})
    )
    wall_ms = (time.perf_counter() - started) * 1000
    assert record.result.status is ToolStatus.TIMEOUT
    assert wall_ms < 400, f"spent {wall_ms:.0f}ms against a 150ms budget"


async def test_a_backend_exception_does_not_take_the_call_down() -> None:
    class Exploding(ToolBackend):
        name = "exploding"

        async def invoke(self, spec, payload):
            raise RuntimeError("backend defect")

    executor = make_executor(Exploding())
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-000001"})
    )
    assert record.result.status is ToolStatus.ERROR
    assert record.result.error_code == "backend_exception"


async def test_unredactable_output_is_withheld() -> None:
    """A half-redacted result that looks normal is the worst possible artifact."""

    class Leaky(ToolBackend):
        name = "leaky"

        async def invoke(self, spec, payload):
            return BackendResponse(data={"note": "reference 12345678901234567890"})

    executor = make_executor(Leaky())
    record = await executor.execute(
        payload(executor, "get_order_status", {"order_reference": "ORD-000001"})
    )
    assert record.result.status is ToolStatus.ERROR
    assert record.result.error_code == "redaction_failed"
    assert record.result.data == {}
    assert not record.result.safe_for_model


# ------------------------------------------------------------------ parallel


async def test_independent_calls_run_concurrently() -> None:
    import time

    executor = make_executor()
    payloads = [
        executor.build_payload(
            "get_order_status",
            session_id="s1",
            trace=TRACE,
            arguments={"order_reference": "ORD-999999"},
            verification=VerificationLevel.STRONG,
            call_index=i,
        )
        for i in range(3)
    ]
    started = time.perf_counter()
    records = await executor.execute_many(payloads)
    wall_ms = (time.perf_counter() - started) * 1000

    assert len(records) == 3
    # Three sequential 900ms timeouts would be ~2.7s; concurrent is ~0.9s.
    assert wall_ms < 2000, f"calls appear to have run sequentially ({wall_ms:.0f}ms)"


async def test_execute_many_on_an_empty_list_is_a_no_op() -> None:
    assert await make_executor().execute_many([]) == ()


# --------------------------------------------------------- the second pack


async def test_the_same_executor_serves_the_healthcare_pack() -> None:
    """Rule 1: swapping the pack must need no code change."""
    pack = load_pack(DOMAINS, "healthcare")
    executor = ToolExecutor(
        build_registry(pack, GLOBAL),
        MockToolBackend.from_pack_dir(DOMAINS / "healthcare"),
        build_pipeline(POLICY, pack=pack, presidio=None, mode=RedactionMode.REALTIME),
    )
    record = await executor.execute(
        executor.build_payload(
            "check_claim_status",
            session_id="s1",
            trace=TRACE,
            arguments={"claim_reference": "CLM-112233"},
            verification=VerificationLevel.STRONG,
        )
    )
    assert record.result.ok
    assert record.result.data["member_name"] == "[MEMBER_NAME_1]"
    assert record.result.data["decision"] == "approved"
