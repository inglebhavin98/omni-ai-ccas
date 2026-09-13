"""Cross-variant agreement, the mechanism behind CLAUDE.md Rule 6.

Rule 6 says no node may depend on a single model. Declaring two variants in
``models.yaml`` makes that *possible*; this module makes it *checked*.

Parity is agreement on the **decision**, never on the words. Two models asked the same
question return different prose and different confidence numbers even when they agree
completely, so comparing text would fail on every run and teach everyone to ignore it.
What must match is the thing the graph acts on: the intent id, the tool name, the
verdict.

Everything here is pure. Calling the providers is the caller's job, which is what lets
the same report be produced from live calls, from cassettes, or from stubs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ccas.schemas.eval import ProviderParityResult

__all__ = [
    "Divergence",
    "ParityCase",
    "ParityReport",
    "VariantOutcome",
    "compare_variants",
]


@dataclass(frozen=True, slots=True)
class ParityCase:
    """One question put to every variant."""

    case_id: str
    utterance: str
    expected: str | None = None


@dataclass(frozen=True, slots=True)
class VariantOutcome:
    """What one variant decided, or why it could not.

    ``decision is None`` means the variant failed. That is a divergence, not a skip: a
    binding that times out has not agreed with anything, and counting it as a pass is how
    a provider outage would slip through the gate that exists to catch it.

    ``unavailable`` is the one exception, and it means something different: the request
    was never served -- a 429, a cold model, a quota. Nobody answered, so there is nothing
    to compare. Scoring that as disagreement would invent a defect that does not exist.
    The case is dropped from the denominator instead, and ``meets`` refuses to pass a run
    that dropped too many.
    """

    decision: str | None = None
    confidence: float | None = None
    error: str | None = None
    latency_ms: int = 0
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        return self.decision is not None and self.error is None


@dataclass(frozen=True, slots=True)
class Divergence:
    case_id: str
    utterance: str
    left: str | None
    right: str | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ParityReport:
    node: str
    left: str
    right: str
    cases: int
    agreements: int
    divergences: tuple[Divergence, ...]
    max_confidence_gap: float
    skipped: int = 0
    left_p95_ms: int = 0
    right_p95_ms: int = 0

    @property
    def agreement_rate(self) -> float:
        # Zero cases is zero agreement, not perfect agreement. A harness that could not
        # reach either provider must not report a green Rule 6.
        return self.agreements / self.cases if self.cases else 0.0

    def meets(self, threshold: float, min_cases: int = 1) -> bool:
        """Did this run clear the bar, and did it measure enough to say so?"""
        return self.cases >= max(1, min_cases) and self.agreement_rate >= threshold

    def inconclusive(self, min_cases: int = 1) -> bool:
        """Too few measurements, and throttling is why.

        Separated from a plain shortfall so a caller can skip a quota-starved run while
        still failing a harness that simply forgot to feed it cases. Either way the run
        does not pass -- ``meets`` is false in both.
        """
        return self.cases < max(1, min_cases) and self.skipped > 0

    def to_contract(
        self,
        baseline_model: str,
        candidate_model: str,
        threshold: float = 0.85,
        min_sample_size: int = 1,
    ) -> ProviderParityResult:
        """Freeze this run into the contract that an ``EvalReport`` carries.

        The models are passed in rather than carried through every outcome: the harness
        compares *variants*, and which model a variant resolves to is a fact about
        ``models.yaml`` at the moment of the run.

        The contract's ``*_p95_ttft_ms`` fields receive total response latency here. A
        parity case is a structured, non-streaming call, so there is no time-to-first-token
        distinct from time-to-answer.
        """
        return ProviderParityResult(
            node=self.node,
            baseline=self.left,
            candidate=self.right,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            agreement=self.agreement_rate,
            agreement_threshold=threshold,
            baseline_p95_ttft_ms=self.left_p95_ms,
            candidate_p95_ttft_ms=self.right_p95_ms,
            sample_size=self.cases,
            unmeasured=self.skipped,
            min_sample_size=min_sample_size,
        )

    def summary(self) -> str:
        return (
            f"{self.node}: {self.left} vs {self.right} -- "
            f"{self.agreements}/{self.cases} agree ({self.agreement_rate:.0%}), "
            f"{len(self.divergences)} divergent, {self.skipped} unmeasured, "
            f"confidence gap <= {self.max_confidence_gap:.2f}"
        )


Outcomes = Sequence[tuple[ParityCase, VariantOutcome, VariantOutcome]]


def compare_variants(node: str, left: str, right: str, outcomes: Outcomes) -> ParityReport:
    """Fold per-case outcomes into one report."""
    agreements = 0
    measured = 0
    skipped = 0
    divergences: list[Divergence] = []
    gap = 0.0
    left_ms: list[int] = []
    right_ms: list[int] = []

    for case, a, b in outcomes:
        if a.unavailable or b.unavailable:
            skipped += 1
            continue
        measured += 1
        left_ms.append(a.latency_ms)
        right_ms.append(b.latency_ms)
        if a.ok and b.ok and a.decision == b.decision:
            agreements += 1
            if a.confidence is not None and b.confidence is not None:
                gap = max(gap, abs(a.confidence - b.confidence))
            continue
        divergences.append(
            Divergence(
                case_id=case.case_id,
                utterance=case.utterance,
                left=a.decision,
                right=b.decision,
                detail=_detail(a, b),
            )
        )

    return ParityReport(
        node=node,
        left=left,
        right=right,
        cases=measured,
        agreements=agreements,
        divergences=tuple(divergences),
        max_confidence_gap=gap,
        skipped=skipped,
        left_p95_ms=_p95(left_ms),
        right_p95_ms=_p95(right_ms),
    )


def _detail(a: VariantOutcome, b: VariantOutcome) -> str | None:
    errors = [e for e in (a.error, b.error) if e]
    return "; ".join(errors) if errors else None


def _p95(values: list[int]) -> int:
    """Nearest-rank p95. With eight cases this is the slowest one, which is the point:
    a parity run is small, and the tail is what breaches a budget."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]
