"""Can a decision model do Module 6b's judging? (spike)

The router spike answered "is jev faster and as accurate at picking an intent". This asks
the question that fits it better. M6b's judge has no 800 ms budget, runs on a 5% sample
offline, and its whole job is to rate an exchange against a rubric -- which is what the
Score primitive is, rather than something a Score has to be coaxed into.

There is no labelled judge corpus. Bitext and NATCS label intents, not reply quality, and
Rule 9 forbids pressing either into a role it does not have. So the grader is twelve
constructed exchanges in `tests/fixtures/judge_cases.json`, six sound and six with exactly
one planted defect -- the method the vendor's own citation-check recipe uses. Twelve cases
prove separation or its absence; they do not calibrate a pass mark, and this script says so
rather than implying otherwise.

    uv run python scripts/spike_jev_judge.py

Nothing here is wired into anything. Adoption is a locked-stack change needing an ADR
(Rule 5), and two schema changes this script demonstrates the need for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccas.config.domain_loader import load_domain
from ccas.config.settings import Settings
from ccas.evals.judge_rubric import DEFAULT_PASS_MARK, judge_questions, judge_scores
from ccas.llm.base import LLMProviderError
from ccas.llm.typesafe import TypeSafeClient
from ccas.observability.logging import configure_logging, get_logger
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.redaction.placeholder import PlaceholderVault
from ccas.schemas.eval import JudgeDimension, JudgeScore

LOG = get_logger("scripts.spike_jev_judge")

DIMENSIONS = (
    JudgeDimension.FAITHFULNESS,
    JudgeDimension.TASK_SUCCESS,
    JudgeDimension.POLICY_ADHERENCE,
)
#: pii_leakage is absent by decision, not omission: judging it means showing a vendor the
#: text before redaction, which Rule 2 forbids. Presidio stays local.

PASS_MARKS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)


@dataclass(frozen=True, slots=True)
class Judged:
    case_id: str
    planted: str | None
    expect: dict[str, bool]
    scores: tuple[JudgeScore, ...]
    latency_ms: int
    input_tokens: int
    output_tokens: int

    def score_for(self, dimension: JudgeDimension) -> float:
        return next(s.score for s in self.scores if s.dimension is dimension)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spike_jev_judge", description=__doc__)
    p.add_argument("--domain", default="retail")
    p.add_argument("--domains", type=Path, default=Path("domains"))
    p.add_argument("--cases", type=Path, default=Path("tests/fixtures/judge_cases.json"))
    p.add_argument("--out", type=Path, default=None)
    return p


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.typesafe_api_key is None:
        print("error: TYPESAFE_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    loaded = load_domain(args.domains, args.domain)
    redaction = build_pipeline(
        Path("configs") / "redaction_policy.yaml", pack=loaded.pack, mode=RedactionMode.BATCH
    )
    questions = judge_questions(DIMENSIONS)
    client = TypeSafeClient(
        api_key=settings.typesafe_api_key.get_secret_value(),
        base_url=settings.typesafe_base_url,
        model=settings.typesafe_model,
    )

    planted = sum(1 for c in cases if c["planted"])
    print(f"  cases        {len(cases)} constructed exchanges, {planted} with a planted defect")
    print(f"  dimensions   {', '.join(d.value for d in DIMENSIONS)}")
    print(f"  asking       one request per case, {len(questions)} questions each")
    print(f"  model        typesafe:{settings.typesafe_model}\n")

    judged: list[Judged] = []
    errors: dict[str, int] = {}
    wall = time.perf_counter()
    try:
        for case in cases:
            # One vault per case, shared by every field. The same reference has to become
            # the same placeholder in the caller's turn and in the tool result -- two
            # tokens for one value would read to the judge as a reply citing a value that
            # is not in the results, and it would be right to mark it down for that.
            allocator = redaction.new_allocator(PlaceholderVault())
            state = {
                name: redaction.redact(case[name], allocator)
                for name in ("caller", "reply", "tools", "rules")
            }
            try:
                result = await client.ask(state, questions)
                judged.append(
                    Judged(
                        case_id=case["case_id"],
                        planted=case["planted"],
                        expect=case["expect"],
                        scores=judge_scores(result.body, DIMENSIONS),
                        latency_ms=result.latency_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                )
                print(".", end="", flush=True)
            except (LLMProviderError, ValueError) as exc:
                errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
                print("x", end="", flush=True)
    finally:
        await client.aclose()
    elapsed = time.perf_counter() - wall
    print("\n")

    if not judged:
        print("  nothing was judged")
        for name, count in errors.items():
            print(f"    {count}x {name}")
        return 3

    _report(judged, elapsed, errors)
    LOG.info(
        "spike.jev_judge",
        correlation_id="spike-judge",
        judged=len(judged),
        errors=sum(errors.values()),
        model=settings.typesafe_model,
    )
    if args.out:
        args.out.write_text(json.dumps(_as_json(judged, elapsed, errors), indent=2) + "\n")
        print(f"\n  wrote {args.out}")
    return 0


def _report(judged: list[Judged], elapsed: float, errors: dict[str, int]) -> None:
    tokens = sum(j.input_tokens + j.output_tokens for j in judged)
    latencies = sorted(j.latency_ms for j in judged)
    p95 = latencies[max(0, -(-95 * len(latencies) // 100) - 1)]

    print("  scores (bold = the dimension this case was built to fail)")
    header = "  ".join(f"{d.value[:9]:>9}" for d in DIMENSIONS)
    print(f"    {'case':<30}  {header}")
    for j in judged:
        cells = []
        for d in DIMENSIONS:
            mark = "*" if j.planted == d.value else " "
            cells.append(f"{j.score_for(d):>8.2f}{mark}")
        print(f"    {j.case_id:<30}  {'  '.join(cells)}")

    print("\n  separation per dimension (does a score gap exist at all?)")
    print(f"    {'dimension':<18}  {'sound (min)':>12}  {'planted (max)':>14}  {'gap':>7}")
    separable: dict[str, float] = {}
    for d in DIMENSIONS:
        sound = [j.score_for(d) for j in judged if j.expect[d.value]]
        planted = [j.score_for(d) for j in judged if not j.expect[d.value]]
        if not sound or not planted:
            continue
        gap = min(sound) - max(planted)
        separable[d.value] = gap
        print(f"    {d.value:<18}  {min(sound):>12.2f}  {max(planted):>14.2f}  {gap:>+7.2f}")

    print("\n  the pass mark each dimension would need (a gap is an interval, not a number)")
    print(f"    {'dimension':<18}  {'works for marks in':>22}")
    for d in DIMENSIONS:
        sound = [j.score_for(d) for j in judged if j.expect[d.value]]
        planted = [j.score_for(d) for j in judged if not j.expect[d.value]]
        if not sound or not planted:
            continue
        low, high = max(planted), min(sound)
        window = f"({low:.2f}, {high:.2f}]" if high > low else "none -- the two overlap"
        print(f"    {d.value:<18}  {window:>22}")

    print("\n  pass-mark sweep (correct verdicts out of every case x dimension)")
    print(f"    {'mark':>5}  {'correct':>8}  {'missed defects':>15}  {'false alarms':>13}")
    for mark in PASS_MARKS:
        correct = missed = false_alarm = 0
        for j in judged:
            for d in DIMENSIONS:
                verdict = j.score_for(d) >= mark
                if verdict == j.expect[d.value]:
                    correct += 1
                elif verdict:
                    missed += 1  # said fine, was built broken
                else:
                    false_alarm += 1
        total = len(judged) * len(DIMENSIONS)
        flag = "  <- default" if abs(mark - DEFAULT_PASS_MARK) < 1e-9 else ""
        print(f"    {mark:>5.2f}  {correct:>4}/{total:<3}  {missed:>15}  {false_alarm:>13}{flag}")

    print(f"\n  wall {elapsed:.1f}s, p95 {p95} ms per case, {tokens} tokens, {len(judged)} calls")
    print(f"  cost  ${tokens / 1_000_000 * 0.042:.4f} at $0.042/M input tokens")
    print(
        "  for comparison, one chat-model judge call on the same state costs roughly "
        "3,500 ms (the router baseline's p95) and must either be asked three times or "
        "trusted to hold three rubrics at once"
    )
    if errors:
        print("\n  failures by cause")
        for name, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3}x  {name}")
    if any(gap <= 0 for gap in separable.values()):
        print(
            "\n  at least one dimension does not separate: a sound reply scored no higher "
            "than a broken one, so no pass mark fixes it"
        )


def _as_json(judged: list[Judged], elapsed: float, errors: dict[str, int]) -> dict[str, object]:
    return {
        "model": "jev",
        "judged": len(judged),
        "wall_seconds": round(elapsed, 1),
        "tokens": sum(j.input_tokens + j.output_tokens for j in judged),
        "errors": errors,
        "cases": [
            {
                "case_id": j.case_id,
                "planted": j.planted,
                "expect": j.expect,
                "latency_ms": j.latency_ms,
                "scores": {
                    s.dimension.value: {
                        "score": round(s.score, 4),
                        "passed": s.passed,
                        "rationale": s.rationale,
                    }
                    for s in j.scores
                },
            }
            for j in judged
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
