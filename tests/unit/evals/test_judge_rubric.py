"""The judge rubric: what a Score question has to be before it is worth asking.

Module 6b's judge dimensions are `JudgeDimension` values and its output is a
`JudgeScore`. A System One model answers a rubric rather than writing an opinion, so the
mapping between the two is code -- which means it is testable, and these are the parts
that would otherwise be wrong quietly: the direction of the scale, the normalisation, and
the rationale, which the model cannot write because it cannot write anything.
"""

from __future__ import annotations

import pytest

from ccas.evals.judge_rubric import (
    RUBRICS,
    judge_questions,
    judge_scores,
    score_question,
)
from ccas.schemas.eval import JudgeDimension

DIMS = (JudgeDimension.FAITHFULNESS, JudgeDimension.TASK_SUCCESS)


def answer(
    score: float, probabilities: dict[str, float], confidence: float = 0.9
) -> dict[str, object]:
    return {
        "type": "score",
        "score": score,
        "probabilities": probabilities,
        "confidence": confidence,
        "legend": {str(i): text for i, text in enumerate(RUBRICS[JudgeDimension.FAITHFULNESS])},
    }


def body(**answers: object) -> dict[str, object]:
    return {"answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}}


def test_every_dimension_has_a_rubric_with_a_low_and_a_high_end() -> None:
    """A Score is an *ordered* scale. One level, or an unordered set, is a Choice wearing
    the wrong type, and the score it returns would not mean what the caller reads."""
    for dimension, levels in RUBRICS.items():
        assert len(levels) >= 2, dimension
        assert len(levels) <= 10, dimension  # the API's ceiling
        assert all(level.strip() for level in levels)


def test_pii_leakage_has_no_rubric_and_that_is_deliberate() -> None:
    """Rule 2: judging leakage means sending pre-redaction text to a vendor. There is no
    rubric because there is no question that can be asked without breaking the rule."""
    assert JudgeDimension.PII_LEAKAGE not in RUBRICS


def test_a_question_is_shaped_as_the_api_documents_a_score() -> None:
    q = score_question(JudgeDimension.FAITHFULNESS)
    assert q["type"] == "score"
    assert isinstance(q["instructions"], str) and q["instructions"]
    assert q["criteria"] == list(RUBRICS[JudgeDimension.FAITHFULNESS])


def test_all_dimensions_ride_in_one_request() -> None:
    """The state is the whole exchange and it is the expensive part. Asking three
    questions against it in three calls sends it three times for no gain -- the vendor's
    own measurement is 12x cheaper batched, and each question is scored independently."""
    questions = judge_questions(DIMS)
    assert set(questions) == {"faithfulness", "task_success"}


def test_the_score_is_normalised_onto_the_schema_s_zero_to_one() -> None:
    """`JudgeScore.score` is 0..1. jev returns a position on the level scale, so the top
    level of a four-level rubric is 3.0 and has to become 1.0."""
    top = len(RUBRICS[JudgeDimension.FAITHFULNESS]) - 1
    (result,) = judge_scores(
        body(faithfulness=answer(float(top), {str(top): 1.0})),
        (JudgeDimension.FAITHFULNESS,),
    )
    assert result.score == pytest.approx(1.0)
    assert result.passed


def test_a_bottom_scoring_reply_fails() -> None:
    (result,) = judge_scores(
        body(faithfulness=answer(0.0, {"0": 1.0})), (JudgeDimension.FAITHFULNESS,)
    )
    assert result.score == pytest.approx(0.0)
    assert not result.passed


def test_the_rationale_is_written_by_code_from_the_level_the_model_chose() -> None:
    """jev is "not trained to generate text", so the rationale cannot come back from the
    model. Authoring it from the legend is strictly better than a generated one: it is
    the rubric's own words, so it cannot describe a level the model did not pick."""
    levels = RUBRICS[JudgeDimension.FAITHFULNESS]
    (result,) = judge_scores(
        body(faithfulness=answer(1.2, {"1": 0.8, "2": 0.2})), (JudgeDimension.FAITHFULNESS,)
    )
    assert levels[1][:40] in result.rationale
    assert "confidence" in result.rationale


def test_a_missing_answer_raises_rather_than_scoring_zero() -> None:
    """Zero is "the reply was groundless". A dimension the service did not answer is not
    that, and recording it as that would fail a turn that was never judged."""
    with pytest.raises(ValueError, match="task_success"):
        judge_scores(body(faithfulness=answer(3.0, {"3": 1.0})), DIMS)


def test_the_pass_mark_is_a_parameter_because_it_cannot_be_inherited() -> None:
    """The same argument as the router's 0.82: a pass mark belongs to a rubric and a
    model, and carrying one across either is guesswork wearing a number."""
    middling = body(faithfulness=answer(2.0, {"2": 1.0}))
    (lenient,) = judge_scores(middling, (JudgeDimension.FAITHFULNESS,), pass_at=0.6)
    (strict,) = judge_scores(middling, (JudgeDimension.FAITHFULNESS,), pass_at=0.9)
    assert lenient.passed and not strict.passed


def test_every_judge_case_fails_at_most_the_dimension_it_plants() -> None:
    """The fixture is the grader, so its invariant is part of the measurement.

    A case that breaks two dimensions at once lets a judge score badly for being right --
    which is exactly what happened on the first run, and it read as a model error until
    the fixture was re-read. Pinning it here means the next person to add a case finds
    out immediately.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    cases = json.loads((repo / "tests" / "fixtures" / "judge_cases.json").read_text())["cases"]
    assert len(cases) >= 8
    assert {c["case_id"] for c in cases}.__len__() == len(cases)
    for case in cases:
        failing = [name for name, ok in case["expect"].items() if not ok]
        assert len(failing) <= 1, case["case_id"]
        assert (failing[0] if failing else None) == case["planted"], case["case_id"]
        assert set(case["expect"]) == {d.value for d in RUBRICS}, case["case_id"]
        assert all(case[field].strip() for field in ("caller", "reply", "tools", "rules"))
