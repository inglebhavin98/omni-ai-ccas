"""The Module 6b judge: one request per case, every dimension in it.

``ccas.evals.judge_rubric`` is the rubric interface; this is the runtime that puts it to
a provider through the locked ``LLMProvider`` protocol. The spike
(``scripts/spike_jev_judge.py``) proved the dimensions separate on constructed cases; a
locked stack needs the judge on the same contract every other node uses -- two variants,
the parity gate, quota semantics -- rather than on a vendor SDK beside it.

Every dimension is asked in one request against one state. The spike measured batching as
accuracy-neutral, and the state -- the exchange -- is the expensive part of the call.

The state is assembled from surfaces the platform already redacted: turns carry their
reports, tool records are filtered to ``safe_for_model``. This module performs no
redaction of its own and could not leak if it tried -- ``compose`` refuses to build the
block if any part were not egress-permitted.
"""

from __future__ import annotations

import json
from typing import Any

from ccas.evals.judge_rubric import (
    DEFAULT_PASS_MARK,
    RUBRICS,
    questions_block,
)
from ccas.llm.base import (
    LLMProvider,
    LLMProviderError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from ccas.llm.factory import build_provider
from ccas.llm.prompt import authored, compose
from ccas.observability.logging import get_logger
from ccas.schemas.eval import JudgeDimension, JudgeScore, JudgeVerdict
from ccas.schemas.llm import LLMRequest, Message, ModelBinding
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import SessionState

__all__ = [
    "JUDGE_INSTRUCTIONS",
    "RESPONSE_SCHEMA",
    "SessionJudge",
    "exchange_state",
    "judge_exchange",
]

LOG = get_logger("ccas.evals.judge")

_DIMENSIONS: tuple[JudgeDimension, ...] = (
    JudgeDimension.FAITHFULNESS,
    JudgeDimension.TASK_SUCCESS,
    JudgeDimension.POLICY_ADHERENCE,
)
#: ``pii_leakage`` is absent by decision, not omission: judging it means showing a vendor
#: text before redaction, which Rule 2 forbids. Presidio stays local (6.14).

_TOP_LEVEL = max(len(levels) - 1 for levels in RUBRICS.values())
_DIMENSION_NAMES = ", ".join(d.value for d in _DIMENSIONS)

#: The system prompt is authored text and the stable prefix of the call; the exchange
#: rides after it, which is also what keeps the prompt-cacheable prefix stable (Rule 3).
JUDGE_INSTRUCTIONS = (
    "You are the quality judge for an automated care assistant. STATE is the exchange to "
    "grade: the rules in force, the caller's turns, the assistant's reply, and the tool "
    "results the assistant actually had. "
    f"For each dimension ({_DIMENSION_NAMES}) decide where the exchange sits on that "
    "dimension's rubric. "
    'Reply with one JSON object and nothing else: {"answers": {"<dimension>": '
    '{"type": "score", "score": <number>, "probabilities": {"<level>": <weight>}, '
    '"confidence": <number>}}} where score is on a 0..3 scale against the rubric levels '
    "given in the question definitions, level is a rubric index 0..3, and probabilities "
    "describes how the levels split. Nothing else belongs in the object."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "object",
            "properties": {
                d.value: {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "score": {"type": "number"},
                        "probabilities": {"type": "object"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["type", "score"],
                }
                for d in _DIMENSIONS
            },
            "required": [d.value for d in _DIMENSIONS],
        }
    },
    "required": ["answers"],
}

_QUESTION_PREAMBLE = (
    "Grade the STATE below on every dimension. Each question names its rubric levels, "
    "low end first."
)

_TEMPLATE = (
    "{preamble}\n\n"
    "{questions}\n\n"
    "STATE\n"
    "rules: {rules}\n"
    "caller turns:\n{turns}\n"
    "assistant reply: {reply}\n"
    "tool results:\n{tools}"
)


def _turn_lines(state: SessionState) -> str:
    """Caller turns, in order. Empty rather than padded: absence is information."""
    lines = [
        f"- {turn.content.require_egress()}"
        for turn in state.turns
        if turn.speaker.value == "caller" and turn.content.text.strip()
    ]
    return "\n".join(lines) if lines else "- (none)"


def _reply_text(state: SessionState) -> str:
    for turn in reversed(state.turns):
        if turn.speaker.value == "bot":
            return turn.content.require_egress()
    return "(no assistant reply)"


def _tool_lines(state: SessionState) -> str:
    """What the assistant actually had. Only ``safe_for_model`` results -- the same
    filter the responder prompt applies, so the judge sees what the reply saw."""
    lines: list[str] = []
    for record in state.tool_records:
        if not (record.result.ok and record.result.safe_for_model):
            continue
        lines.append(f"- {record.payload.tool_name}({record.payload.arguments})")
        lines.append(f"  -> {json.dumps(record.result.data, sort_keys=True, default=str)}")
    return "\n".join(lines) if lines else "- (no tools were called)"


def exchange_state(rules: str, state: SessionState) -> RedactedText:
    """Render the exchange as one egress-permitted block for the judge.

    Every part already carries its own report -- turns from the redactor, tool results
    from the executor's redaction step, the rules authored pack data -- so this composes
    rather than re-redacts. ``compose`` inherits the weakest status among the parts, which
    means a leak upstream cannot be laundered into a judge prompt here.
    """
    return compose(
        _TEMPLATE,
        {
            "preamble": authored(_QUESTION_PREAMBLE),
            "questions": authored(questions_block(_DIMENSIONS)),
            "rules": authored(rules),
            "turns": authored(_turn_lines(state)),
            "reply": authored(_reply_text(state)),
            "tools": authored(_tool_lines(state)),
        },
    )


def _rationale(answer: dict[str, Any], levels: tuple[str, ...], normalised: float) -> str:
    """The rubric's own words for the level the model actually believes.

    The *modal* level rather than the rounded score: the score averages over the
    distribution and can land between two levels, describing neither. Without
    probabilities the top rubric level is inferred from the score band, which is the
    nearest defensible reading. Chat models may write their own ``rationale``; it is used
    when present because it says *why* rather than *what*, and truncated to fit the
    schema.
    """
    probabilities = answer.get("probabilities")
    index: int | None = None
    if isinstance(probabilities, dict) and probabilities:
        try:
            index = int(max(probabilities, key=lambda k: float(probabilities[k])))
        except (TypeError, ValueError):
            index = None
    if index is None:
        index = min(max(round(normalised * (len(levels) - 1)), 0), len(levels) - 1)
    index = min(max(index, 0), len(levels) - 1)

    confidence = answer.get("confidence")
    confidence_note = (
        f", confidence {float(confidence):.2f}" if isinstance(confidence, int | float) else ""
    )
    own = answer.get("rationale")
    own_note = f" judge: {str(own)[:512]}" if isinstance(own, str) and own.strip() else ""
    return (
        f"{levels[index]} [score {normalised:.2f}, rubric level "
        f"{index}/{len(levels) - 1}{confidence_note}]{own_note}"
    )[:2048]


def _scores_from(body: dict[str, Any], pass_mark: float) -> tuple[JudgeScore, ...]:
    """``judge_scores`` with the rationale policy above. A missing dimension raises:
    the absence of a verdict is not a zero, and scoring it as one would fail a turn
    nobody judged."""
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise LLMProviderError("judge response carries no answers object")
    scores: list[JudgeScore] = []
    for dimension in _DIMENSIONS:
        answer = answers.get(dimension.value)
        if not isinstance(answer, dict):
            raise LLMProviderError(f"judge response carries no {dimension.value!r} answer")
        try:
            raw = float(answer.get("score", 0.0))
        except (TypeError, ValueError) as exc:
            raise LLMProviderError(f"judge score for {dimension.value!r} is not a number") from exc
        normalised = min(1.0, max(0.0, raw / _TOP_LEVEL))
        scores.append(
            JudgeScore(
                dimension=dimension,
                score=normalised,
                rationale=_rationale(answer, RUBRICS[dimension], normalised),
                passed=normalised >= pass_mark,
            )
        )
    return tuple(scores)


class SessionJudge:
    """Grades a session's exchange on the ``judge`` node's bindings (Rule 6).

    Two variants are declared in ``configs/models.yaml`` and the parity gate covers this
    node like any other; ``variant`` selects which one grades.
    """

    def __init__(
        self,
        bindings: Any,
        settings: Any,
        *,
        variant: str | None = None,
        provider: LLMProvider | None = None,
        pass_mark: float = DEFAULT_PASS_MARK,
    ) -> None:
        self._binding: ModelBinding = bindings.resolve("judge", variant)
        self._provider = provider or build_provider(self._binding, settings)
        self._pass_mark = pass_mark

    async def aclose(self) -> None:
        await self._provider.aclose()

    @property
    def judge_provider(self) -> str:
        return self._binding.provider.value

    @property
    def judge_model(self) -> str:
        return self._binding.model

    async def judge_session(
        self, rules: str, state: SessionState, *, sampled: bool = True
    ) -> JudgeVerdict | None:
        """Grade one session. ``None`` means unmeasured, not failed.

        A quota (``ProviderRateLimitedError``) or an unreachable provider leaves the case
        out of the denominator -- the ADR-0014 rule, applied to the judge: the 5% sample
        may under-shoot, but a run must never record a verdict against a case the vendor
        never served. Any other provider error propagates: a malformed answer is a defect
        in the binding, and hiding it would be parity theatre.
        """
        request = LLMRequest(
            binding=self._binding,
            system=authored(JUDGE_INSTRUCTIONS),
            messages=(Message(role="user", content=exchange_state(rules, state)),),
            response_schema=RESPONSE_SCHEMA,
        )
        try:
            response = await self._provider.structured(request)
        except ProviderRateLimitedError:
            LOG.info(
                "judge.unmeasured",
                correlation_id=state.session_id,
                reason="rate_limited",
                judge_provider=self.judge_provider,
                judge_model=self.judge_model,
            )
            return None
        except ProviderUnavailableError as exc:
            LOG.info(
                "judge.unmeasured",
                correlation_id=state.session_id,
                reason="unavailable",
                detail=str(exc)[:160],
                judge_provider=self.judge_provider,
                judge_model=self.judge_model,
            )
            return None

        return _verdict_from(
            response.parsed,
            session_id=state.session_id,
            binding=self._binding,
            pass_mark=self._pass_mark,
            sampled=sampled,
        )


def _verdict_from(
    parsed: dict[str, Any] | None,
    *,
    session_id: str,
    binding: ModelBinding,
    pass_mark: float,
    sampled: bool,
) -> JudgeVerdict | None:
    """Assemble the verdict from a served response. Shared by the class and the
    case-level ``judge_exchange`` so both produce records that agree."""
    if parsed is None:
        raise LLMProviderError(f"judge binding {binding.model!r} returned no structured body")
    try:
        scores = _scores_from(parsed, pass_mark)
    except LLMProviderError:
        raise
    except (TypeError, ValueError) as exc:
        raise LLMProviderError(f"judge response was not usable: {exc}") from exc

    verdict = JudgeVerdict(
        session_id=session_id,
        judge_model=binding.model,
        judge_provider=binding.provider,
        scores=scores,
        sampled=sampled,
    )
    LOG.info(
        "judge.verdict",
        correlation_id=session_id,
        judge_provider=binding.provider.value,
        judge_model=binding.model,
        passed=verdict.passed,
        scores={s.dimension.value: round(s.score, 3) for s in scores},
    )
    return verdict


async def judge_exchange(
    provider: LLMProvider,
    binding: ModelBinding,
    rules: str,
    state: SessionState,
    *,
    pass_mark: float = DEFAULT_PASS_MARK,
    sampled: bool = True,
) -> JudgeVerdict | None:
    """One case through one prepared provider. The parts-level entry a fixture suite
    uses: the caller owns the provider's lifecycle and names the binding, so a suite can
    grade every case on the binding it is testing without re-reading settings per case.

    Unmeasured semantics are the class's: a quota or an unreachable provider returns
    ``None``; a malformed answer raises.
    """
    request = LLMRequest(
        binding=binding,
        system=authored(JUDGE_INSTRUCTIONS),
        messages=(Message(role="user", content=exchange_state(rules, state)),),
        response_schema=RESPONSE_SCHEMA,
    )
    try:
        response = await provider.structured(request)
    except ProviderRateLimitedError:
        LOG.info(
            "judge.unmeasured",
            correlation_id=state.session_id,
            reason="rate_limited",
            judge_provider=binding.provider.value,
            judge_model=binding.model,
        )
        return None
    except ProviderUnavailableError as exc:
        LOG.info(
            "judge.unmeasured",
            correlation_id=state.session_id,
            reason="unavailable",
            detail=str(exc)[:160],
            judge_provider=binding.provider.value,
            judge_model=binding.model,
        )
        return None
    return _verdict_from(
        response.parsed,
        session_id=state.session_id,
        binding=binding,
        pass_mark=pass_mark,
        sampled=sampled,
    )
