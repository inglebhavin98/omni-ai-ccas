from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    EvalReport,
    JudgeDimension,
    JudgeScore,
    JudgeVerdict,
    MetricValue,
    ProviderName,
    ProviderParityResult,
)


def score(dimension: JudgeDimension, passed: bool = True) -> JudgeScore:
    return JudgeScore(
        dimension=dimension, score=1.0 if passed else 0.0, rationale="r", passed=passed
    )


def verdict(*scores: JudgeScore) -> JudgeVerdict:
    return JudgeVerdict(
        session_id="sess-1",
        judge_model="claude-haiku-4-5",
        judge_provider=ProviderName.ANTHROPIC,
        scores=scores or (score(JudgeDimension.TASK_SUCCESS),),
    )


def test_duplicate_dimensions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate judge dimension"):
        verdict(score(JudgeDimension.TASK_SUCCESS), score(JudgeDimension.TASK_SUCCESS))


def test_verdict_passes_only_when_every_dimension_passes() -> None:
    assert verdict(score(JudgeDimension.FAITHFULNESS)).passed
    assert not verdict(
        score(JudgeDimension.FAITHFULNESS), score(JudgeDimension.TASK_SUCCESS, passed=False)
    ).passed


def test_pii_leakage_is_surfaced_separately() -> None:
    leaky = verdict(score(JudgeDimension.PII_LEAKAGE, passed=False))
    assert leaky.leaked_pii
    assert not verdict(score(JudgeDimension.PII_LEAKAGE)).leaked_pii


def test_metric_direction_is_respected() -> None:
    assert MetricValue(name="containment", value=0.8, threshold=0.7).passed
    assert not MetricValue(name="containment", value=0.6, threshold=0.7).passed
    assert MetricValue(name="wer", value=0.1, threshold=0.2, higher_is_better=False).passed
    assert not MetricValue(name="wer", value=0.3, threshold=0.2, higher_is_better=False).passed


def test_metric_without_a_threshold_always_passes() -> None:
    assert MetricValue(name="turns", value=42.0).passed


def parity(agreement: float, **kw: object) -> ProviderParityResult:
    fields: dict[str, object] = {
        "node": "router",
        "baseline": "anthropic",
        "candidate": "vllm",
        "baseline_model": "claude-haiku-4-5",
        "candidate_model": "qwen3-8b",
        "agreement": agreement,
        "baseline_p95_ttft_ms": 180,
        "candidate_p95_ttft_ms": 260,
        "sample_size": 200,
    }
    fields.update(kw)
    return ProviderParityResult(**fields)  # type: ignore[arg-type]


def test_parity_gate_fails_on_divergence() -> None:
    assert parity(0.9).passed
    assert not parity(0.6).passed


def test_two_variants_of_one_provider_are_expressible() -> None:
    """Parity moved from vendor-vs-vendor to model-vs-model when both live behind one
    gateway (ADR-0012). The contract has to be able to say so."""
    result = parity(
        0.9,
        baseline="openrouter",
        candidate="openrouter_alt",
        baseline_model="a/model-one:free",
        candidate_model="b/model-two:free",
    )
    assert result.passed
    assert result.baseline_model != result.candidate_model


def test_two_variants_naming_one_model_is_rejected() -> None:
    """Comparing a model with itself always agrees. That is not parity (Rule 6)."""
    with pytest.raises(ValidationError, match="distinct"):
        parity(0.9, candidate_model="claude-haiku-4-5")


def test_a_run_that_measured_nothing_does_not_pass() -> None:
    """ADR-0014: throttling drops cases. Zero measurements is not 100% agreement."""
    assert not parity(1.0, sample_size=0, unmeasured=8).passed


def test_a_run_below_the_minimum_sample_does_not_pass() -> None:
    assert not parity(1.0, sample_size=3, min_sample_size=5, unmeasured=5).passed
    assert parity(1.0, sample_size=5, min_sample_size=5, unmeasured=3).passed


def test_report_fails_if_any_component_fails() -> None:
    passing = EvalReport(
        report_id="r1",
        domain="retail",
        suite="regression",
        metrics=(MetricValue(name="containment", value=0.8, threshold=0.7),),
        verdicts=(verdict(),),
        parity=(parity(0.9),),
    )
    assert passing.passed

    failing = passing.model_copy(update={"parity": (parity(0.5),)})
    assert not failing.passed


def test_report_surfaces_pii_leaks() -> None:
    report = EvalReport(
        report_id="r1",
        domain="retail",
        suite="regression",
        verdicts=(
            verdict(score(JudgeDimension.PII_LEAKAGE, passed=False)),
            verdict(score(JudgeDimension.PII_LEAKAGE)),
        ),
    )
    assert len(report.pii_leaks) == 1
    assert not report.passed
