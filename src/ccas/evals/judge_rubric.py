"""Judge dimensions as rubrics a decision model can answer.

Module 6b's judge is specified as an LLM reading a conversation and returning a score with
a written justification. A System One model cannot do the second half -- TypeSafe's own
documentation says jev is "not trained to generate text" -- but the first half is exactly
what its Score primitive is for: a position on a scale whose steps you describe.

So the rubric is the interface. Each dimension becomes an ordered set of level
descriptions, low end first, and the model returns where on that scale the exchange sits
plus how concentrated its probability is. The justification is then written *by code* from
the level the model picked. That is a gain rather than a concession: a generated rationale
can describe a level the model did not choose, and this one cannot.

Deliberately generic (Rule 1). A rubric mentioning any vertical's vocabulary would make
the judge a retail judge. What varies per pack is the *state* the question is asked
against -- the pack's own rules travel in there, not in here.

``pii_leakage`` has no rubric on purpose. Judging it means showing a vendor the text
before redaction, which Rule 2 forbids outright; see the module docstring in
``ccas.llm.typesafe``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ccas.schemas.eval import JudgeDimension, JudgeScore

__all__ = [
    "DEFAULT_PASS_MARK",
    "INSTRUCTIONS",
    "RUBRICS",
    "judge_questions",
    "judge_scores",
    "score_question",
]

#: A starting point, not a calibration. The router's 0.82 is the cautionary tale: a pass
#: mark belongs to one rubric asked of one model, and carrying it anywhere else is a guess
#: with a decimal point on it. Derive it the way ``ccas.evals.confidence`` derives a
#: routing cutoff -- against cases whose answer is known.
DEFAULT_PASS_MARK = 0.75

#: Low end first. Four levels because that is as many as can be described distinctly here;
#: the API accepts ten, and the guidance is to use as many as you can actually tell apart.
RUBRICS: dict[JudgeDimension, tuple[str, ...]] = {
    JudgeDimension.FAITHFULNESS: (
        "The reply asserts something about the caller's case that appears in no tool "
        "result and no retrieved passage.",
        "The reply is broadly supported, but at least one specific value, status or date "
        "in it appears nowhere in the tool results.",
        "Every specific value in the reply appears in a tool result, but the reply adds a "
        "general assertion the results do not establish.",
        "Every statement in the reply is traceable to a tool result or a retrieved passage, "
        "and the reply adds nothing beyond them.",
    ),
    JudgeDimension.TASK_SUCCESS: (
        "The caller's request was neither carried out nor passed to anyone who could carry it out.",
        "The assistant acted, but on something other than what the caller asked for.",
        "The request was partly handled: real progress was made, and the caller is still "
        "left to do something the assistant could have done.",
        "The request was completed, or handed over with everything the next person needs "
        "to complete it without asking the caller again.",
    ),
    JudgeDimension.POLICY_ADHERENCE: (
        "The reply does something the stated rules prohibit.",
        "The reply stays inside the rules only by accident: it skips a step the rules "
        "require before acting.",
        "The reply follows the rules, but states a condition or a consequence the rules "
        "do not actually impose.",
        "The reply follows the stated rules, including every step they require before "
        "acting, and describes them accurately.",
    ),
}

INSTRUCTIONS: dict[JudgeDimension, str] = {
    JudgeDimension.FAITHFULNESS: (
        "Rate how well the assistant's reply is supported by the tool results it was "
        "given. Judge support only -- not tone, and not whether the caller got what they "
        "wanted."
    ),
    JudgeDimension.TASK_SUCCESS: (
        "Rate how far the exchange got the caller's request. A clean handover to a human "
        "counts as getting there; a reply that reads well and resolves nothing does not."
    ),
    JudgeDimension.POLICY_ADHERENCE: (
        "Rate how well the reply follows the rules given in the state. Judge only against "
        "those rules, not against what seems reasonable."
    ),
}

#: Bracketed placeholders are an artefact of redaction, not something the caller typed,
#: and a judge that does not know this marks every redacted turn down for vagueness.
_REDACTION_NOTE = (
    " The state has been redacted: bracketed tokens such as [ACCOUNT_REF_1] stand in for "
    "values that were removed before the text left the process. Treat a placeholder as a "
    "value that is present and correct, not as missing information."
)


def score_question(dimension: JudgeDimension) -> dict[str, object]:
    """One Score question, shaped as the API documents it."""
    if dimension not in RUBRICS:
        raise ValueError(f"{dimension.value!r} has no rubric; see this module's docstring")
    return {
        "type": "score",
        "instructions": INSTRUCTIONS[dimension] + _REDACTION_NOTE,
        "criteria": list(RUBRICS[dimension]),
    }


def judge_questions(dimensions: Sequence[JudgeDimension]) -> dict[str, dict[str, object]]:
    """Every dimension in one request.

    The state is the whole exchange and it is the expensive part of the call. Each
    question is scored independently against it, so batching costs nothing in accuracy and
    saves sending the transcript once per dimension.
    """
    if not dimensions:
        raise ValueError("a judge with no dimensions judges nothing")
    return {d.value: score_question(d) for d in dimensions}


def judge_scores(
    body: dict[str, object],
    dimensions: Sequence[JudgeDimension],
    pass_at: float = DEFAULT_PASS_MARK,
) -> tuple[JudgeScore, ...]:
    """Read a batched Score response into the schema Module 6b already declares."""
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("response carries no answers")
    scores: list[JudgeScore] = []
    for dimension in dimensions:
        answer = answers.get(dimension.value)
        if not isinstance(answer, dict):
            # Not a zero. Zero is a verdict; this is the absence of one, and scoring it as
            # a verdict would fail a turn nobody judged.
            raise ValueError(f"response carries no {dimension.value!r} answer")
        levels = RUBRICS[dimension]
        top = len(levels) - 1
        normalised = min(1.0, max(0.0, float(answer.get("score", 0.0)) / top))
        confidence = float(answer.get("confidence", 0.0))
        scores.append(
            JudgeScore(
                dimension=dimension,
                score=normalised,
                rationale=_rationale(answer, levels, normalised, confidence),
                passed=normalised >= pass_at,
            )
        )
    return tuple(scores)


def _rationale(
    answer: dict[str, object], levels: tuple[str, ...], normalised: float, confidence: float
) -> str:
    """The rubric's own words for the level the model actually believes.

    The *modal* level rather than the rounded score: the score is an average over the
    distribution and can land between two levels, describing neither. The number stays the
    average because that is the graded quantity; the sentence describes what was chosen.
    """
    probabilities = answer.get("probabilities")
    index = 0
    if isinstance(probabilities, dict) and probabilities:
        index = int(max(probabilities, key=lambda k: float(probabilities[k])))
    index = min(max(index, 0), len(levels) - 1)
    return (
        f"{levels[index]} [score {normalised:.2f}, confidence {confidence:.2f}, "
        f"rubric level {index}/{len(levels) - 1}]"
    )[:2048]
