"""The router accuracy scorer, offline (9.18).

Two things this must get right, and both have bitten this project already.

A row that *defined* the taxonomy cannot grade it. CLAUDE.md forbids grading a taxonomy
against the corpus it was mined from, and adoption opens the same trap from the other
side: the labels came from Bitext, so scoring against Bitext rows that produced them
measures nothing. The scorer refuses them rather than trusting the caller to filter.

A row nobody routed is unmeasured, not wrong. The same distinction ADR-0014 forced on the
parity gate -- a 429 is not a misclassification, and counting it as one would report the
router as broken when the provider was simply out of quota.
"""

from __future__ import annotations

import pytest

from ccas.evals.router_accuracy import (
    RouterCase,
    RouterOutcome,
    score_router,
)
from ccas.mining.adopt import DERIVATION_SPLIT


def _case(row_id: str, expected: str) -> RouterCase:
    return RouterCase(row_id=row_id, utterance=f"utterance {row_id}", expected_intent=expected)


def _holdout(n: int, expected: str = "order.track_order") -> list[RouterCase]:
    """Row ids that hash into the holdout half, so the scorer accepts them."""
    from ccas.mining.adopt import split_of

    out, i = [], 0
    while len(out) < n:
        row_id = f"row:{i}"
        if split_of(row_id) != DERIVATION_SPLIT:
            out.append(_case(row_id, expected))
        i += 1
    return out


def test_a_correct_prediction_scores() -> None:
    cases = _holdout(4)
    report = score_router([(c, RouterOutcome(predicted=c.expected_intent)) for c in cases])
    assert report.exact_accuracy == 1.0
    assert report.measured == 4


def test_a_wrong_prediction_does_not() -> None:
    cases = _holdout(4)
    report = score_router([(c, RouterOutcome(predicted="account.close_account")) for c in cases])
    assert report.exact_accuracy == 0.0
    assert report.measured == 4


def test_the_right_category_with_the_wrong_intent_is_partial_credit() -> None:
    """Overall accuracy hides the difference between "lost" and "nearly right"."""
    cases = _holdout(4, expected="order.track_order")
    report = score_router([(c, RouterOutcome(predicted="order.cancel_order")) for c in cases])
    assert report.exact_accuracy == 0.0
    assert report.category_accuracy == 1.0


def test_a_rate_limited_row_is_unmeasured_not_wrong() -> None:
    cases = _holdout(4)
    outcomes = [
        (cases[0], RouterOutcome(unavailable=True, error="429")),
        *[(c, RouterOutcome(predicted=c.expected_intent)) for c in cases[1:]],
    ]
    report = score_router(outcomes)
    assert report.measured == 3
    assert report.unmeasured == 1
    assert report.exact_accuracy == 1.0


def test_scoring_nothing_is_not_perfect_accuracy() -> None:
    """An all-throttled run must not read as a green router."""
    cases = _holdout(3)
    report = score_router([(c, RouterOutcome(unavailable=True)) for c in cases])
    assert report.measured == 0
    assert report.exact_accuracy == 0.0
    assert not report.meets(0.0, min_cases=1)


def test_a_row_that_defined_the_taxonomy_is_refused() -> None:
    """The circularity guard. Grading on a derivation row measures the label set, not the
    router."""
    from ccas.mining.adopt import split_of

    derivation = next(f"row:{i}" for i in range(500) if split_of(f"row:{i}") == DERIVATION_SPLIT)
    with pytest.raises(ValueError, match=r"derivation|held out"):
        score_router([(_case(derivation, "order.track_order"), RouterOutcome(predicted="x"))])


def test_per_intent_accuracy_exposes_a_collapsed_intent() -> None:
    """One intent routing to nothing is invisible in an overall number."""
    good = _holdout(6, expected="order.track_order")
    bad = [
        c
        for c in _holdout(20, expected="refund.get_refund")
        if c.row_id not in {g.row_id for g in good}
    ][:2]
    outcomes = [(c, RouterOutcome(predicted=c.expected_intent)) for c in good]
    outcomes += [(c, RouterOutcome(predicted="order.track_order")) for c in bad]

    report = score_router(outcomes)
    by_intent = dict(report.per_intent)
    assert by_intent["order.track_order"].accuracy == 1.0
    assert by_intent["refund.get_refund"].accuracy == 0.0
    assert report.worst_intents(1)[0][0] == "refund.get_refund"


def test_template_placeholders_are_removed_from_the_utterance() -> None:
    """A quarter of Bitext's caller lines carry `{{Order Number}}` template syntax.

    Nobody types that, so grading on it measures how a model reacts to synthetic markup
    rather than whether it understands a request. The braces go; the words stay, because
    substituting a plausible value would invent data the corpus does not contain.
    """
    from ccas.evals.router_accuracy import naturalise

    assert naturalise("cancel order {{Order Number}}") == "cancel order order number"
    assert naturalise("help with {{Account Type}} account") == "help with account type account"


def test_naturalise_leaves_ordinary_text_alone() -> None:
    from ccas.evals.router_accuracy import naturalise

    assert naturalise("where is my delivery") == "where is my delivery"


def test_naturalise_does_not_leave_double_spaces() -> None:
    from ccas.evals.router_accuracy import naturalise

    assert "  " not in naturalise("i need {{Order Number}} and {{Invoice Number}} please")
