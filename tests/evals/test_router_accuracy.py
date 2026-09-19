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
    outcome_for_exception,
    score_router,
)
from ccas.llm.base import (
    LLMProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
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


def test_a_timed_out_row_is_unmeasured_not_wrong() -> None:
    """A provider that never answered served nothing, whether it said 429 or said nothing.

    ADR-0014 drew this line for the quota case only. The first live run (52.5% exact) spent
    19 of 40 rows exhausting a 3x30s ladder and every one was scored as a misclassification
    attributed to `<none>`, which reads as a broken router rather than an absent provider.
    """
    cases = _holdout(4)
    timed_out = outcome_for_exception(ProviderTimeoutError("no response after 3 attempt(s)"))
    outcomes = [
        (cases[0], timed_out),
        *[(c, RouterOutcome(predicted=c.expected_intent)) for c in cases[1:]],
    ]
    report = score_router(outcomes)
    assert report.measured == 3
    assert report.unmeasured == 1
    assert report.exact_accuracy == 1.0
    assert not report.confusions, "an unserved row is not a confusion"


def test_a_timeout_is_divergent_when_the_model_never_answered() -> None:
    """The loophole ADR-0014 guarded against: a model that never answers must not vanish.

    Dropping every timed-out row unconditionally would let a wholly dead binding report
    "nothing to see" instead of failing, which is exactly how a silent single-model
    dependency would hide from Rule 6.
    """
    cases = _holdout(4)
    timed_out = outcome_for_exception(ProviderTimeoutError("no response"))
    report = score_router([(c, timed_out) for c in cases])
    assert report.unmeasured == 0, "nothing excuses a model that answered nothing"
    assert report.measured == 4
    assert report.exact_accuracy == 0.0


def test_a_timeout_is_unserved_once_the_model_has_answered_elsewhere() -> None:
    """The same binding answering other rows in the same run is the evidence that the
    timeout was capacity, not incapacity."""
    cases = _holdout(4)
    outcomes = [
        (cases[0], outcome_for_exception(ProviderTimeoutError("no response"))),
        *[(c, RouterOutcome(predicted=c.expected_intent)) for c in cases[1:]],
    ]
    report = score_router(outcomes)
    assert report.measured == 3
    assert report.unmeasured == 1


def test_a_rate_limit_is_unmeasured_even_when_nothing_answered() -> None:
    """A 429 is self-describing -- the provider said it never served the request -- so it
    needs no corroboration from the rest of the run. Unchanged from ADR-0014."""
    cases = _holdout(3)
    report = score_router(
        [(c, outcome_for_exception(ProviderRateLimitedError("429"))) for c in cases]
    )
    assert report.measured == 0
    assert report.unmeasured == 3
    assert report.exact_accuracy == 0.0


def test_a_rate_limit_is_still_unmeasured() -> None:
    assert outcome_for_exception(ProviderRateLimitedError("429")).unavailable


def test_a_served_but_unusable_answer_is_divergent_not_unmeasured() -> None:
    """Rule 6: a case that *failed* is divergent. The response arrived and was unusable,
    which is a real defect in the prompt or the schema and must stay in the denominator."""
    cases = _holdout(2)
    broken = outcome_for_exception(ValueError("intent_id missing from response"))
    report = score_router([(cases[0], broken), (cases[1], RouterOutcome(predicted="x.y"))])
    assert not broken.unavailable
    assert report.measured == 2
    assert report.unmeasured == 0
    assert report.exact_accuracy == 0.0


def test_a_generic_provider_error_stays_divergent() -> None:
    """Only the never-served cases leave the denominator; everything else is a failure."""
    assert not outcome_for_exception(LLMProviderError("malformed payload")).unavailable


def test_a_row_that_never_answered_contributes_no_latency() -> None:
    """A failed row has no latency, and feeding its 0 ms into the distribution understates
    p95. The first live run put 19 zeros into a 40-sample p95."""
    cases = _holdout(4)
    outcomes = [
        (cases[0], RouterOutcome(predicted=cases[0].expected_intent, latency_ms=900)),
        (cases[1], RouterOutcome(predicted=cases[1].expected_intent, latency_ms=800)),
        (cases[2], outcome_for_exception(ValueError("unusable"))),
        (cases[3], outcome_for_exception(ValueError("unusable"))),
    ]
    report = score_router(outcomes)
    assert report.measured == 4, "a served-and-unusable row stays in the denominator"
    assert report.p95_latency_ms == 900, "the two failures must not enter the latency sample"


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
