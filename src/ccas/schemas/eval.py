"""Evaluation contracts (Module 6b)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, JsonValue, SchemaVersion, Slug, utcnow
from ccas.schemas.llm import ProviderName

__all__ = [
    "EvalCase",
    "EvalReport",
    "JudgeDimension",
    "JudgeScore",
    "JudgeVerdict",
    "MetricValue",
    "ProviderParityResult",
]


class JudgeDimension(StrEnum):
    FAITHFULNESS = "faithfulness"
    """Is every assertion grounded in a tool result or retrieved passage?"""

    PII_LEAKAGE = "pii_leakage"
    """Did any unmasked identifier reach the transcript? Score 1.0 means clean."""

    TASK_SUCCESS = "task_success"
    POLICY_ADHERENCE = "policy_adherence"


class JudgeScore(Frozen):
    dimension: JudgeDimension
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2048)
    passed: bool


class JudgeVerdict(Frozen):
    session_id: str = Field(min_length=1, max_length=128)
    judge_model: str = Field(min_length=1)
    judge_provider: ProviderName
    scores: tuple[JudgeScore, ...] = Field(min_length=1)
    sampled: bool = True
    judged_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check_dimensions_unique(self) -> JudgeVerdict:
        dims = [s.dimension for s in self.scores]
        if len(dims) != len(set(dims)):
            raise ValueError("duplicate judge dimension")
        return self

    @property
    def by_dimension(self) -> dict[JudgeDimension, JudgeScore]:
        return {s.dimension: s for s in self.scores}

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)

    @property
    def leaked_pii(self) -> bool:
        score = self.by_dimension.get(JudgeDimension.PII_LEAKAGE)
        return score is not None and not score.passed


class EvalCase(Frozen):
    case_id: Slug
    domain: Slug
    description: str = Field(min_length=1)
    turns: tuple[str, ...] = Field(min_length=1)
    """Scripted caller utterances, already redacted."""

    expected_intent: Slug | None = None
    expected_tools: tuple[Slug, ...] = ()
    expected_outcome: str | None = None
    must_escalate: bool = False
    tags: tuple[Slug, ...] = ()


class MetricValue(Frozen):
    name: Slug
    value: float
    threshold: float | None = None
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        if self.threshold is None:
            return True
        return (
            self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold
        )


class ProviderParityResult(Frozen):
    """Guards Rule 6: no node may depend on a single model.

    ``baseline`` and ``candidate`` are *variant* names, not vendors. When every model
    lives behind one gateway the two variants share a provider and differ only in model,
    which is why the models are named separately and required to differ (ADR-0012).

    ``sample_size`` counts cases that were actually *measured*. A case nobody served --
    a quota, a cold model -- lands in ``unmeasured`` and is excluded from ``agreement``,
    because scoring it as disagreement would invent a defect. ``passed`` therefore also
    requires ``min_sample_size`` measurements: a run that measured nothing agrees on
    nothing, however its rate reads (ADR-0014).
    """

    node: Slug
    baseline: str = Field(min_length=1, max_length=64)
    candidate: str = Field(min_length=1, max_length=64)
    baseline_model: str = Field(min_length=1, max_length=128)
    candidate_model: str = Field(min_length=1, max_length=128)
    agreement: float = Field(ge=0.0, le=1.0)
    agreement_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    baseline_p95_ttft_ms: int = Field(ge=0)
    candidate_p95_ttft_ms: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    unmeasured: int = Field(default=0, ge=0)
    min_sample_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_variants_are_distinct(self) -> Self:
        if self.baseline_model == self.candidate_model:
            raise ValueError(
                f"parity for node {self.node!r} names one model twice "
                f"({self.baseline_model!r}); two variants must be distinct models"
            )
        return self

    @property
    def measured(self) -> bool:
        return self.sample_size >= self.min_sample_size

    @property
    def inconclusive(self) -> bool:
        """Too few measurements, and throttling is why. Not a pass, not a defect."""
        return not self.measured and self.unmeasured > 0

    @property
    def passed(self) -> bool:
        return self.measured and self.agreement >= self.agreement_threshold


class EvalReport(Frozen):
    #: 1.1 -- ProviderParityResult moved from vendor-vs-vendor to variant-vs-model and
    #: gained unmeasured/min_sample_size. See docs/tech-spec.md 1.10 for the migration.
    schema_version: SchemaVersion = "1.1"
    report_id: str = Field(min_length=1, max_length=128)
    domain: Slug
    suite: Slug
    metrics: tuple[MetricValue, ...] = ()
    verdicts: tuple[JudgeVerdict, ...] = ()
    parity: tuple[ProviderParityResult, ...] = ()
    context: dict[str, JsonValue] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utcnow)

    @property
    def passed(self) -> bool:
        return (
            all(m.passed for m in self.metrics)
            and all(v.passed for v in self.verdicts)
            and all(p.passed for p in self.parity)
        )

    @property
    def pii_leaks(self) -> tuple[JudgeVerdict, ...]:
        return tuple(v for v in self.verdicts if v.leaked_pii)
