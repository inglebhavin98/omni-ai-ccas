"""Where does a confidence number stop being worth acting on? (Rule 7: test first.)

The pack gates routing on ``intent_confidence >= 0.82``. That number was chosen for a
chat model's self-reported confidence, and jev's confidence is a different statistic --
a shape of a distribution, not a top probability -- so it cannot simply be carried over.
What can be carried over is the *method*: sweep the cutoff against held-out rows with
known answers and read what each one buys.

Pure by construction, so the same sweep reads a live run, a cassette, or these fakes.
"""

from __future__ import annotations

import pytest

from ccas.evals.confidence import ConfidenceBand, recommend, sweep
from ccas.evals.router_accuracy import RouterCase, RouterOutcome


def row(
    expected: str, predicted: str, confidence: float | None
) -> tuple[RouterCase, RouterOutcome]:
    case = RouterCase(
        row_id=f"holdout:{expected}:{predicted}", utterance="x", expected_intent=expected
    )
    return case, RouterOutcome(predicted=predicted, confidence=confidence, latency_ms=10)


def test_a_band_splits_the_run_at_its_cutoff() -> None:
    outcomes = [
        row("billing.refund_status", "billing.refund_status", 0.95),
        row("billing.refund_status", "order.track", 0.40),
    ]
    (band,) = sweep(outcomes, thresholds=(0.8,))
    assert (band.auto, band.auto_exact) == (1, 1)
    assert (band.deferred, band.deferred_exact) == (1, 0)


def test_the_cutoff_is_inclusive_at_the_boundary() -> None:
    """A row exactly at the threshold is automated. Off-by-one here would move every
    number in the study by a row and nobody would see it."""
    (band,) = sweep([row("a.one", "a.one", 0.82)], thresholds=(0.82,))
    assert band.auto == 1 and band.deferred == 0


def test_the_deferred_slice_is_also_scored_at_the_category_level() -> None:
    """The point of the study. A row the router is unsure of may still have the right L1,
    and sending it to the category's queue is a cheaper answer than escalating -- so the
    sweep has to show whether that is true before anyone builds it."""
    outcomes = [row("billing.refund_status", "billing.get_invoice", 0.3)]
    (band,) = sweep(outcomes, thresholds=(0.8,))
    assert band.deferred_exact == 0
    assert band.deferred_category == 1
    assert band.deferred_category_accuracy == 1.0


def test_a_row_with_no_confidence_cannot_be_gated_and_is_counted_apart() -> None:
    """Some providers return no confidence at all. Folding those into either slice would
    invent a decision the number cannot support; they are reported as ungated instead."""
    outcomes = [row("a.one", "a.one", None), row("a.one", "a.one", 0.9)]
    (band,) = sweep(outcomes, thresholds=(0.8,))
    assert band.ungated == 1
    assert band.gated == 1


def test_an_unserved_row_never_enters_the_sweep() -> None:
    """Rule 6 and ADR-0014: a 429 measured nothing, so it cannot argue for a threshold."""
    case = RouterCase(row_id="holdout:0", utterance="x", expected_intent="a.one")
    outcomes = [(case, RouterOutcome(unavailable=True, error="429"))]
    (band,) = sweep(outcomes, thresholds=(0.8,))
    assert band.gated == 0 and band.ungated == 0


def test_coverage_and_accuracy_move_against_each_other() -> None:
    outcomes = [
        row("a.one", "a.one", 0.95),
        row("a.two", "a.two", 0.85),
        row("a.three", "b.nine", 0.60),
    ]
    low, high = sweep(outcomes, thresholds=(0.5, 0.9))
    assert low.coverage == 1.0
    assert low.auto_accuracy == pytest.approx(2 / 3)
    assert high.coverage == pytest.approx(1 / 3)
    assert high.auto_accuracy == 1.0


def test_recommend_takes_the_lowest_cutoff_that_clears_the_floor() -> None:
    """Lowest, not highest: every point of threshold above what the floor needs is
    coverage given away, and a deferred row costs a human."""
    outcomes = [row(f"a.{i}", f"a.{i}", 0.9) for i in range(6)]
    outcomes += [row(f"b.{i}", "z.wrong", 0.4) for i in range(4)]
    bands = sweep(outcomes, thresholds=(0.3, 0.5, 0.85))
    picked = recommend(bands, min_auto_accuracy=0.95, min_auto=5)
    assert picked is not None and picked.threshold == 0.5


def test_recommend_refuses_a_cutoff_backed_by_too_few_rows() -> None:
    """A 100% slice of two rows is not evidence for automating anything -- the same
    argument ADR-0014 makes about a parity run that measured nothing."""
    outcomes = [row("a.one", "a.one", 0.99), row("a.two", "a.two", 0.99)]
    outcomes += [row(f"b.{i}", "z.wrong", 0.2) for i in range(8)]
    bands = sweep(outcomes, thresholds=(0.9,))
    assert recommend(bands, min_auto_accuracy=0.9, min_auto=5) is None


def test_an_empty_band_reads_as_zero_rather_than_perfect() -> None:
    band = ConfidenceBand(
        threshold=0.9, auto=0, auto_exact=0, deferred=0, deferred_exact=0, deferred_category=0
    )
    assert band.auto_accuracy == 0.0
    assert band.coverage == 0.0


def test_the_table_has_a_line_per_band_and_a_header() -> None:
    """Shared with eval_router.py so the two studies can be compared by eye."""
    from ccas.evals.confidence import sweep_table

    bands = sweep([row("a.one", "a.one", 0.9)], thresholds=(0.5, 0.95))
    lines = sweep_table(bands).splitlines()
    assert len(lines) == 3
    assert "cutoff" in lines[0] and "def cat" in lines[0]
    assert lines[1].startswith("   0.50")
