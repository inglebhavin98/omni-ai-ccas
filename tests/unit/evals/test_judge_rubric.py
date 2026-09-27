"""The rubric is the measurement instrument; these are its calibration pins."""

from __future__ import annotations

import pytest

from ccas.evals.judge_rubric import (
    DEFAULT_PASS_MARK,
    INSTRUCTIONS,
    RUBRICS,
    judge_questions,
    judge_scores,
    questions_block,
    score_question,
)
from ccas.schemas.eval import JudgeDimension

DIMS = (JudgeDimension.FAITHFULNESS, JudgeDimension.TASK_SUCCESS)


def body(
    faithfulness: dict[str, object], task_success: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "answers": {
            "faithfulness": faithfulness,
            "task_success": task_success or {"score": 3, "probabilities": {"3": 1.0}},
        }
    }


def answer(score: float, probabilities: dict[str, float]) -> dict[str, object]:
    return {"score": score, "probabilities": probabilities}


# --------------------------------------------------------------------- shape


def test_every_rubric_has_distinct_nonempty_levels() -> None:
    for dimension, levels in RUBRICS.items():
        assert len(levels) >= 2, dimension
        assert len(levels) <= 10, dimension  # the API's ceiling
        assert all(level.strip() for level in levels)


def test_pii_leakage_has_no_rubric_by_design() -> None:
    """Judging it would mean showing a vendor pre-redaction text -- Rule 2 (6.14)."""
    assert JudgeDimension.PII_LEAKAGE not in RUBRICS


def test_score_question_shape() -> None:
    q = score_question(JudgeDimension.FAITHFULNESS)
    assert q["type"] == "score"
    assert isinstance(q["instructions"], str) and q["instructions"]
    assert q["criteria"] == list(RUBRICS[JudgeDimension.FAITHFULNESS])


def test_judge_questions_cover_every_requested_dimension() -> None:
    questions = judge_questions(DIMS)
    assert set(questions) == {"faithfulness", "task_success"}


def test_questions_block_renders_instructions_over_numbered_criteria() -> None:
    block = questions_block(DIMS)
    assert "  faithfulness: " in block
    assert "\n    0. " in block


# --------------------------------------------------------------- the anchor


def test_faithfulness_top_level_names_assertion_free_replies() -> None:
    """The 6.12 failure: a grounded refusal -- a reply that asserts little because the
    record supports little -- landed mid-scale and read as unfaithful. The top level
    must name that case explicitly, or the model has nowhere honest to put it."""
    top = RUBRICS[JudgeDimension.FAITHFULNESS][-1].lower()
    for word in ("refusal", "assert"):
        assert word in top, top
    # And the instruction must stop completeness from dragging the score down.
    instructions = INSTRUCTIONS[JudgeDimension.FAITHFULNESS].lower()
    assert "not completeness" in instructions


# ------------------------------------------------------------------ scoring


def test_top_level_score_normalises_to_one() -> None:
    top = len(RUBRICS[JudgeDimension.FAITHFULNESS]) - 1
    (result,) = judge_scores(
        body(faithfulness=answer(top, {str(top): 1.0})), (JudgeDimension.FAITHFULNESS,)
    )
    assert result.score == pytest.approx(1.0)
    assert result.passed


def test_bottom_level_score_normalises_to_zero() -> None:
    (result,) = judge_scores(
        body(faithfulness=answer(0.0, {"0": 1.0})), (JudgeDimension.FAITHFULNESS,)
    )
    assert result.score == pytest.approx(0.0)
    assert not result.passed


def test_rationale_quotes_the_level_the_model_chose() -> None:
    levels = RUBRICS[JudgeDimension.FAITHFULNESS]
    (result,) = judge_scores(
        body(faithfulness=answer(1.2, {"1": 0.8, "2": 0.2})), (JudgeDimension.FAITHFULNESS,)
    )
    assert levels[1][:40] in result.rationale
    assert "confidence" in result.rationale


def test_pass_mark_is_the_boundary_of_passed() -> None:
    middling = body(faithfulness=answer(2.0, {"2": 1.0}))
    (lenient,) = judge_scores(middling, (JudgeDimension.FAITHFULNESS,), pass_at=0.6)
    (strict,) = judge_scores(middling, (JudgeDimension.FAITHFULNESS,), pass_at=0.9)
    assert lenient.passed and not strict.passed
    assert DEFAULT_PASS_MARK == 0.75


# ---------------------------------------------------------------- fixture


def test_judge_cases_fixture_is_one_defect_per_case() -> None:
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
    assert len(cases) >= 30
    assert {c["case_id"] for c in cases}.__len__() == len(cases)
    for case in cases:
        failing = [name for name, ok in case["expect"].items() if not ok]
        assert len(failing) <= 1, case["case_id"]
        assert (failing[0] if failing else None) == case["planted"], case["case_id"]
        assert set(case["expect"]) == {d.value for d in RUBRICS}, case["case_id"]
        assert all(case[field].strip() for field in ("caller", "reply", "tools", "rules"))
        assert case["domain"]
    # More than one pack, or the rubric is only shown to work on the vertical it was
    # written against -- which is the failure Rule 1 exists to prevent, and it would
    # still produce numbers.
    assert len({c["domain"] for c in cases}) >= 2


def test_fixture_replies_carry_no_assertion_beyond_their_tools() -> None:
    """The relabel audit, made mechanical (6.12).

    Every case labelled faithfulness=true must not cite a concrete value -- a date, an
    amount, a named destination -- that its tool block does not contain. The two labels
    corrected on 2026-09-27 both failed this: "signed for at the door" and "member
    portal" appear in no tool result. A named-value check cannot catch every possible
    ungrounded assertion, but the concrete-value kind is the one a judge can be shown.
    """
    import json
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    cases = json.loads((repo / "tests" / "fixtures" / "judge_cases.json").read_text())["cases"]
    for case in cases:
        if not case["expect"]["faithfulness"]:
            continue
        reply, tools = case["reply"], case["tools"]
        # Amounts and dates in the reply must appear in the tool block.
        for value in re.findall(r"(?:£|\$|€)\s?\d[\d,.]*|\d{1,2} \w+ \d{4}", reply):
            assert value in tools, f"{case['case_id']}: {value!r} cited but never returned"
        # A named destination must be a tool name or a field value, not an invention.
        for value in re.findall(r"\b(?:portal|app|website|email)\b", reply, re.IGNORECASE):
            assert value.lower() in tools.lower() or value.lower() in case["rules"].lower(), (
                f"{case['case_id']}: {value!r} offered with nothing behind it"
            )
