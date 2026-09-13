"""The parity harness itself, offline.

Rule 6 is only meaningful if the thing that checks it can fail. These tests feed the
report known-divergent outcomes and assert it says so.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.evals.parity import ParityCase, VariantOutcome, compare_variants


def _case(name: str) -> ParityCase:
    return ParityCase(case_id=name, utterance=f"utterance {name}")


def test_identical_decisions_agree() -> None:
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (
                _case("c1"),
                VariantOutcome(decision="billing.status", confidence=0.9),
                VariantOutcome(decision="billing.status", confidence=0.8),
            )
        ],
    )
    assert report.agreement_rate == 1.0
    assert report.divergences == ()


def test_a_different_decision_is_a_divergence() -> None:
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (
                _case("c1"),
                VariantOutcome(decision="billing.status", confidence=0.9),
                VariantOutcome(decision="account.close", confidence=0.9),
            ),
            (
                _case("c2"),
                VariantOutcome(decision="billing.status", confidence=0.9),
                VariantOutcome(decision="billing.status", confidence=0.9),
            ),
        ],
    )
    assert report.agreement_rate == 0.5
    assert len(report.divergences) == 1
    assert report.divergences[0].case_id == "c1"


def test_a_failed_variant_is_a_divergence_not_a_skip() -> None:
    """A binding that errors has not agreed. Counting it as a pass would hide an outage."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (_case("c1"), VariantOutcome(decision=None, error="timeout"), VariantOutcome("x", 0.9))
        ],
    )
    assert report.agreement_rate == 0.0
    assert report.divergences[0].right == "x"
    assert "timeout" in (report.divergences[0].detail or "")


def test_confidence_drift_within_tolerance_still_agrees() -> None:
    """Two models never produce the same number. Parity is about the decision."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[(_case("c1"), VariantOutcome("x", 0.95), VariantOutcome("x", 0.41))],
    )
    assert report.agreement_rate == 1.0
    assert report.max_confidence_gap == pytest.approx(0.54)


def test_an_empty_run_is_not_a_pass() -> None:
    """Zero cases must not report 100% agreement -- that is how a broken gate goes green."""
    report = compare_variants(node="router", left="a", right="b", outcomes=[])
    assert report.agreement_rate == 0.0
    assert not report.meets(0.0)


def test_meets_compares_against_the_threshold() -> None:
    outcomes = [
        (_case(f"c{i}"), VariantOutcome("x", 0.9), VariantOutcome("x" if i else "y", 0.9))
        for i in range(5)
    ]
    report = compare_variants(node="router", left="a", right="b", outcomes=outcomes)
    assert report.agreement_rate == 0.8
    assert report.meets(0.8)
    assert not report.meets(0.85)


def test_a_throttled_variant_is_unmeasured_not_divergent() -> None:
    """429 means nobody answered. Scoring it as disagreement invents a defect."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (
                _case("c1"),
                VariantOutcome(error="rate limited", unavailable=True),
                VariantOutcome("x", 0.9),
            ),
            (_case("c2"), VariantOutcome("x", 0.9), VariantOutcome("x", 0.9)),
        ],
    )
    assert report.cases == 1
    assert report.skipped == 1
    assert report.agreement_rate == 1.0
    assert report.divergences == ()


def test_too_few_measurements_is_not_a_pass() -> None:
    """An all-throttled run must read as inconclusive, never as green."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (
                _case("c1"),
                VariantOutcome(error="rate limited", unavailable=True),
                VariantOutcome("x", 0.9),
            ),
            (_case("c2"), VariantOutcome("x", 0.9), VariantOutcome("x", 0.9)),
        ],
    )
    assert report.agreement_rate == 1.0
    assert not report.meets(0.75, min_cases=2)
    assert report.meets(0.75, min_cases=1)


def test_a_hard_provider_error_is_still_a_divergence() -> None:
    """An outage is not a free pass -- only throttling is excused."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[(_case("c1"), VariantOutcome(error="500 upstream"), VariantOutcome("x", 0.9))],
    )
    assert report.cases == 1
    assert report.skipped == 0
    assert report.agreement_rate == 0.0


def test_an_all_throttled_run_is_inconclusive_not_failed() -> None:
    """Nobody answered anything. That is a quota, not a divergence."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (_case(f"c{i}"), VariantOutcome(unavailable=True), VariantOutcome(unavailable=True))
            for i in range(4)
        ],
    )
    assert report.inconclusive(min_cases=2)
    assert not report.meets(0.0, min_cases=2)


def test_a_measured_run_is_never_inconclusive() -> None:
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[
            (_case(f"c{i}"), VariantOutcome("x", 0.9), VariantOutcome("x", 0.9)) for i in range(4)
        ],
    )
    assert not report.inconclusive(min_cases=2)


def test_too_few_cases_with_nothing_throttled_is_a_failure_not_a_skip() -> None:
    """A harness that only fed two cases is broken. Do not excuse it as a quota."""
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[(_case("c1"), VariantOutcome("x", 0.9), VariantOutcome("x", 0.9))],
    )
    assert not report.inconclusive(min_cases=5)
    assert not report.meets(0.75, min_cases=5)


def test_a_report_converts_to_the_eval_contract() -> None:
    """The harness is a dataclass; what crosses a boundary is the frozen contract."""
    report = compare_variants(
        node="router",
        left="openrouter",
        right="openrouter_alt",
        outcomes=[
            (
                _case(f"c{i}"),
                VariantOutcome("x", 0.9, latency_ms=120),
                VariantOutcome("x", 0.8, latency_ms=300),
            )
            for i in range(4)
        ]
        + [(_case("c9"), VariantOutcome(unavailable=True), VariantOutcome("x", 0.9))],
    )
    result = report.to_contract(
        baseline_model="a/one:free", candidate_model="b/two:free", threshold=0.75
    )
    assert result.node == "router"
    assert result.baseline == "openrouter"
    assert result.sample_size == 4
    assert result.unmeasured == 1
    assert result.agreement == 1.0
    assert result.passed
    assert result.baseline_p95_ttft_ms == 120  # every case took 120 ms


def test_the_contract_refuses_a_report_that_names_one_model_twice() -> None:
    report = compare_variants(
        node="router",
        left="a",
        right="b",
        outcomes=[(_case("c1"), VariantOutcome("x", 0.9), VariantOutcome("x", 0.9))],
    )
    with pytest.raises(ValidationError, match="distinct"):
        report.to_contract(baseline_model="same", candidate_model="same")
