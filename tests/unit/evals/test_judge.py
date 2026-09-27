"""Unit tests for the M6b judge runtime (``ccas.evals.judge``).

No network. The provider is a stub; the point is the contract: egress-clean state in,
schema-constrained request out, ``JudgeVerdict`` out, and unmeasured-not-zero on a quota.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from ccas.evals.judge import (
    JUDGE_INSTRUCTIONS,
    RESPONSE_SCHEMA,
    SessionJudge,
    exchange_state,
)
from ccas.llm.base import (
    LLMProvider,
    LLMProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from ccas.llm.bindings import load_bindings
from ccas.schemas import RedactionStatus
from ccas.schemas.common import Channel, Speaker, TraceContext
from ccas.schemas.escalation import EscalationDecision, HandoffReason
from ccas.schemas.eval import JudgeDimension, JudgeVerdict
from ccas.schemas.llm import (
    LLMChunk,
    LLMRequest,
    LLMResponse,
    ProviderName,
)
from ccas.schemas.session import CallerContext, SessionState, Turn
from ccas.schemas.tools import ToolPayload, ToolRecord, ToolResult, ToolStatus
from tests.factories import make_report, redacted

REPO = Path(__file__).resolve().parents[3]

TRACE = TraceContext(trace_id="a" * 32, span_id="b" * 16, correlation_id="sess-judge")


def session(**kw: Any) -> SessionState:
    base: dict[str, Any] = {
        "session_id": "sess-judge",
        "trace": TRACE,
        "domain": "retail",
        "channel": Channel.CHAT,
        "caller": CallerContext(caller_ref="wb-judge000"),
        "turns": (
            Turn(
                index=0,
                speaker=Speaker.CALLER,
                content=redacted("where is order [ORDER_REF_1]"),
            ),
            Turn(
                index=1,
                speaker=Speaker.BOT,
                content=redacted("It is in transit."),
            ),
        ),
    }
    base.update(kw)
    return SessionState(**base)


def tool_record(name: str = "get_order_status", data: dict[str, Any] | None = None) -> ToolRecord:
    payload = ToolPayload(
        tool_call_id="tc-1",
        tool_name=name,
        session_id="sess-judge",
        timeout_ms=1500,
        trace=TRACE,
        arguments={"order_ref": "[ORDER_REF_1]"},
    )
    result = ToolResult(
        tool_call_id="tc-1",
        status=ToolStatus.OK,
        data=data or {"status": "in_transit"},
        elapsed_ms=12,
        redaction=make_report(),
    )
    return ToolRecord(payload=payload, result=result)


def answers_body(
    faithfulness: float,
    task_success: float = 3.0,
    policy_adherence: float = 3.0,
    **extra: Any,
) -> dict[str, Any]:
    def answer(score: float) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "score",
            "score": score,
            "probabilities": {str(int(score)): 1.0},
            "confidence": 0.9,
        }
        out.update(extra)
        return out

    return {
        "answers": {
            "faithfulness": answer(faithfulness),
            "task_success": answer(task_success),
            "policy_adherence": answer(policy_adherence),
        }
    }


class StubJudgeProvider(LLMProvider):
    """Returns a canned parsed body, or raises. Records the request it was sent."""

    name = ProviderName.VLLM

    def __init__(
        self, parsed: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.parsed = parsed
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:  # pragma: no cover
        raise NotImplementedError

    async def healthy(self) -> bool:
        return True

    async def structured(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            binding=request.binding,
            text=json.dumps(self.parsed or {}),
            parsed=self.parsed,
            ttft_ms=1,
            total_ms=1,
        )


def bindings() -> Any:
    return load_bindings(REPO / "configs" / "models.yaml")


async def verdict_for(
    provider: StubJudgeProvider, state: SessionState | None = None, **kw: Any
) -> JudgeVerdict | None:
    judge = SessionJudge(bindings(), settings=None, provider=provider, **kw)  # type: ignore[arg-type]
    return await judge.judge_session("Be concise.", state or session())


# ------------------------------------------------------------------ the state


def test_the_exchange_state_is_egress_permitted_and_carries_the_exchange() -> None:
    state = session(tool_records=(tool_record(),))
    block = exchange_state("Be concise.", state)
    assert block.egress_permitted
    text = block.require_egress()
    assert "where is order [ORDER_REF_1]" in text
    assert "It is in transit." in text
    assert "get_order_status" in text
    assert "in_transit" in text
    assert "Be concise." in text


def test_an_unredacted_turn_cannot_reach_the_judge_prompt() -> None:
    """Rule 2, at the judge: a dirty turn is refused *at the turn* -- the state renderer
    calls require_egress() per turn, so there is no path from a raw turn to the provider,
    not even into an intermediate dirty block."""
    dirty = redacted("my card is 4111", status=RedactionStatus.DIRTY)
    state = session(
        turns=(
            Turn(index=0, speaker=Speaker.CALLER, content=dirty),
            Turn(index=1, speaker=Speaker.BOT, content=redacted("ok")),
        )
    )
    with pytest.raises(PermissionError, match="egress blocked"):
        exchange_state("Be concise.", state)


def test_failed_and_unsafe_tool_results_do_not_reach_the_judge() -> None:
    failed = tool_record("cancel_order")
    failed = failed.model_copy(
        update={
            "result": failed.result.model_copy(
                update={"status": ToolStatus.TIMEOUT, "error_code": "timeout"}
            )
        }
    )
    unsafe = ToolRecord(
        payload=tool_record().payload,
        result=tool_record().result.model_copy(
            update={"redaction": make_report(status=RedactionStatus.DIRTY)}
        ),
    )
    text = exchange_state(
        "r", session(tool_records=(failed, unsafe, tool_record()))
    ).require_egress()
    assert "cancel_order" not in text
    assert "get_order_status" in text


# ---------------------------------------------------------------- the verdict


@pytest.mark.asyncio
async def test_a_judged_session_returns_a_verdict_in_the_schema() -> None:
    provider = StubJudgeProvider(parsed=answers_body(3.0, 2.0, 3.0))
    verdict = await verdict_for(provider)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.session_id == "sess-judge"
    assert verdict.judge_model  # the bound model, whatever it is
    assert set(verdict.by_dimension) == {
        JudgeDimension.FAITHFULNESS,
        JudgeDimension.TASK_SUCCESS,
        JudgeDimension.POLICY_ADHERENCE,
    }
    faith = verdict.by_dimension[JudgeDimension.FAITHFULNESS]
    assert faith.score == 1.0 and faith.passed
    task = verdict.by_dimension[JudgeDimension.TASK_SUCCESS]
    assert abs(task.score - 2 / 3) < 1e-9 and not task.passed
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_the_pass_mark_is_a_parameter_not_an_inheritance() -> None:
    lenient = await verdict_for(StubJudgeProvider(parsed=answers_body(2.0)), pass_mark=0.5)
    strict = await verdict_for(StubJudgeProvider(parsed=answers_body(2.0)), pass_mark=0.9)
    assert lenient is not None and strict is not None
    assert lenient.passed and not strict.passed


@pytest.mark.asyncio
async def test_the_request_is_schema_constrained_and_names_the_judge_node() -> None:
    provider = StubJudgeProvider(parsed=answers_body(3.0))
    await verdict_for(provider)
    (request,) = provider.requests
    assert request.response_schema == RESPONSE_SCHEMA
    assert request.binding.node == "judge"
    assert request.messages[0].content.egress_permitted
    assert "faithfulness" in request.messages[0].content.text


@pytest.mark.asyncio
async def test_the_variant_selects_its_binding() -> None:
    primary = StubJudgeProvider(parsed=answers_body(3.0))
    alt = StubJudgeProvider(parsed=answers_body(3.0))
    judge_a = SessionJudge(bindings(), None, provider=primary)  # type: ignore[arg-type]
    judge_b = SessionJudge(bindings(), None, provider=alt, variant="openrouter_alt")  # type: ignore[arg-type]
    await judge_a.judge_session("r", session())
    await judge_b.judge_session("r", session())
    models = {primary.requests[0].binding.model, alt.requests[0].binding.model}
    assert len(models) == 2, "two variants must name distinct models (Rule 6)"


# ------------------------------------------------------- unmeasured, not zero


@pytest.mark.asyncio
async def test_a_quota_is_unmeasured_and_returns_no_verdict() -> None:
    verdict = await verdict_for(StubJudgeProvider(error=ProviderRateLimitedError("429, free tier")))
    assert verdict is None


@pytest.mark.asyncio
async def test_an_unreachable_provider_is_unmeasured_too() -> None:
    verdict = await verdict_for(StubJudgeProvider(error=ProviderUnavailableError("no endpoints")))
    assert verdict is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parsed",
    [
        {},
        {"answers": {"faithfulness": {"type": "score", "score": 3.0}}},
        {"answers": {"faithfulness": "high"}},
        {"answers": {"faithfulness": {"type": "score", "score": "very good"}}},
    ],
)
async def test_a_malformed_answer_is_a_defect_not_a_zero(parsed: dict[str, Any]) -> None:
    """A body with a missing or unparseable dimension means the binding is broken.
    Scoring it as 0 would fail a turn nobody judged; dropping it would hide the defect."""
    with pytest.raises(LLMProviderError):
        await verdict_for(StubJudgeProvider(parsed=parsed))


@pytest.mark.asyncio
async def test_a_body_with_no_parsed_object_raises() -> None:
    provider = StubJudgeProvider(parsed=None)
    with pytest.raises(LLMProviderError, match="no structured body"):
        await verdict_for(provider)


# ------------------------------------------------------------- the rationale


@pytest.mark.asyncio
async def test_the_rationale_names_the_modal_rubric_level() -> None:
    verdict = await verdict_for(StubJudgeProvider(parsed=answers_body(1.0)))
    assert verdict is not None
    faith = verdict.by_dimension[JudgeDimension.FAITHFULNESS]
    assert "rubric level 1/3" in faith.rationale
    assert faith.rationale  # never empty: the schema demands text


@pytest.mark.asyncio
async def test_the_model_may_write_its_own_rationale_and_it_is_kept() -> None:
    body = answers_body(3.0, rationale="Every value in the reply appears in the results.")
    verdict = await verdict_for(StubJudgeProvider(parsed=body))
    assert verdict is not None
    assert "Every value" in verdict.by_dimension[JudgeDimension.FAITHFULNESS].rationale


@pytest.mark.asyncio
async def test_an_escalated_session_still_composes() -> None:
    """A session that ended in escalation carries an EscalationDecision on its state; the
    judge reads turns and tool records only, and the decision is pack policy that reaches
    it through the authored rules line, never through caller-derived state."""
    state = session(
        escalation=EscalationDecision(
            reason=HandoffReason.LOW_CONFIDENCE, triggered_by="confidence", at_turn=2
        )
    )
    block = exchange_state("Be concise.", state)
    assert block.egress_permitted


def test_the_system_prompt_is_authored_and_stable() -> None:
    """Stable authored text is also the prompt-cache prefix (Rule 3): the volatile
    exchange must ride in the user message, never in the system prompt."""
    assert "[ORDER_REF" not in JUDGE_INSTRUCTIONS
    assert "faithfulness" in JUDGE_INSTRUCTIONS
