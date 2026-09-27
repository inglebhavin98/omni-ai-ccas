"""Run the offline judge suite: every constructed case through the live judge node.

``make evals`` runs the parity gate; this runs the other half -- the M6b judge against
``tests/fixtures/judge_cases.json``, the constructed exchanges whose right answer is
known because it was built (six-plus defects planted, one per case). Separation and the
pass-mark sweep print exactly as the spike printed them, so the numbers line up.

    uv run python scripts/run_evals.py                 # primary judge variant
    uv run python scripts/run_evals.py --variant openrouter_alt
    uv run python scripts/run_evals.py --out data/interim/judge_suite.json

One request per case. 30 cases is 30 calls against the daily cap -- know your budget
before running the full fixture.

Nothing here grades on a corpus the judge was mined from or against (Rule 9): the
fixture is synthetic, constructed, and labelled by construction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccas.config.settings import Settings
from ccas.evals.judge import DEFAULT_PASS_MARK, judge_exchange
from ccas.llm.base import LLMProviderError
from ccas.llm.bindings import load_bindings
from ccas.llm.factory import build_provider
from ccas.observability.logging import configure_logging, get_logger
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.redaction.placeholder import PlaceholderVault
from ccas.schemas.common import Channel, Speaker
from ccas.schemas.session import CallerContext, LatencyLedger, SessionState, Turn

LOG = get_logger("scripts.run_evals")

PASS_MARKS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
DIMENSIONS = ("faithfulness", "task_success", "policy_adherence")


@dataclass(frozen=True, slots=True)
class Graded:
    case_id: str
    domain: str
    planted: str | None
    expect: dict[str, bool]
    scores: dict[str, float]
    latency_ms: int


def fixture_state(case: dict[str, object], pack_dir: Path, domain: str) -> SessionState:
    """The constructed exchange as a SessionState, redacted through the pack pipeline.

    One vault per case: the same reference has to become the same placeholder everywhere
    it appears, exactly as the spike argued -- two tokens for one value would read to the
    judge as a reply citing a value that is not in the results.
    """
    redaction = build_pipeline(
        Path("configs") / "redaction_policy.yaml",
        pack=load_pack(pack_dir, domain),
        mode=RedactionMode.BATCH,
    )
    allocator = redaction.new_allocator(PlaceholderVault())
    caller = redaction.redact(str(case["caller"]), allocator)
    reply = redaction.redact(str(case["reply"]), allocator)
    rules = redaction.redact(str(case["rules"]), allocator)
    tools = redaction.redact(str(case["tools"]), allocator)
    correlation = f"judge-{case['case_id']}"

    # Tool results are data the executor would have redacted on the way out; the fixture
    # lines are redacted as one block and carried as the bot's side of the exchange.
    return SessionState(
        session_id=correlation,
        trace=trace_for(correlation),
        domain=domain,
        channel=Channel.CHAT,
        caller=CallerContext(caller_ref=correlation.ljust(8, "0")[:32]),
        turns=(
            Turn(index=0, speaker=Speaker.CALLER, content=caller),
            Turn(index=1, speaker=Speaker.BOT, content=reply),
            Turn(index=2, speaker=Speaker.SYSTEM, content=rules),
            Turn(index=3, speaker=Speaker.SYSTEM, content=tools),
        ),
        latency=LatencyLedger(),
    )


def load_pack(pack_dir: Path, domain: str) -> object:
    from ccas.config.domain_loader import load_domain

    return load_domain(pack_dir, domain).pack


def trace_for(correlation: str) -> object:
    from ccas.observability.tracing import current_trace_context

    return current_trace_context(correlation_id=correlation)


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.openrouter_api_key is None:
        print("error: OPENROUTER_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    bindings = load_bindings(settings.models_config)
    binding = bindings.resolve("judge", args.variant)
    provider = build_provider(binding, settings)

    print(f"  cases        {len(cases)} constructed exchanges")
    print(f"  judge        openrouter:{binding.model} ({binding.structured_mode.value} mode)")
    print(f"  pass mark    {args.pass_mark} (uncalibrated, 6.12)\n")

    graded: list[Graded] = []
    unmeasured = 0
    errors: dict[str, int] = {}
    wall = time.perf_counter()
    try:
        for case in cases:
            domain = str(case.get("domain", "retail"))
            state = fixture_state(case, settings.domains_dir, domain)
            try:
                verdict = await judge_exchange(
                    provider, binding, str(case["rules"]), state, pass_mark=args.pass_mark
                )
            except LLMProviderError as exc:
                errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
                print("x", end="", flush=True)
                continue
            if verdict is None:
                unmeasured += 1
                print("-", end="", flush=True)
                continue
            graded.append(
                Graded(
                    case_id=str(case["case_id"]),
                    domain=domain,
                    planted=case.get("planted"),  # type: ignore[arg-type]
                    expect=case["expect"],  # type: ignore[arg-type]
                    scores={s.dimension.value: s.score for s in verdict.scores},
                    latency_ms=0,
                )
            )
            print(".", end="", flush=True)
    finally:
        await provider.aclose()
    elapsed = time.perf_counter() - wall
    print("\n")

    if not graded:
        print(f"  nothing measured ({unmeasured} unmeasured, {sum(errors.values())} errors)")
        return 3

    _report(graded, unmeasured, errors, elapsed, args.pass_mark)
    _write(args, graded, unmeasured, errors, elapsed, binding.model)
    LOG.info(
        "evals.judge_suite",
        correlation_id="run-evals",
        measured=len(graded),
        unmeasured=unmeasured,
        model=binding.model,
    )
    return 0


def _score(graded: list[Graded], dim: str, expected: bool) -> list[float]:
    return [g.scores[dim] for g in graded if g.expect[dim] is expected]


def _report(
    graded: list[Graded], unmeasured: int, errors: dict[str, int], elapsed: float, mark: float
) -> None:
    print(f"  measured {len(graded)}, unmeasured {unmeasured}, wall {elapsed:.1f}s\n")

    print("  separation per dimension (min sound vs max planted)")
    for dim in DIMENSIONS:
        sound, planted = _score(graded, dim, True), _score(graded, dim, False)
        if sound and planted:
            gap = min(sound) - max(planted)
            print(f"    {dim:<18} {min(sound):>5.2f} vs {max(planted):>5.2f}  gap {gap:>+6.2f}")

    print("\n  pass-mark sweep (correct verdicts out of every case x dimension)")
    total = len(graded) * len(DIMENSIONS)
    for candidate in PASS_MARKS:
        correct = missed = false_alarm = 0
        for g in graded:
            for dim in DIMENSIONS:
                verdict = g.scores[dim] >= candidate
                if verdict == g.expect[dim]:
                    correct += 1
                elif verdict:
                    missed += 1
                else:
                    false_alarm += 1
        flag = "  <- used" if abs(candidate - mark) < 1e-9 else ""
        print(
            f"    {candidate:>5.2f}  {correct:>4}/{total:<3}"
            f"  missed {missed:<3}  false {false_alarm:<3}{flag}"
        )

    if errors:
        print("\n  failures by cause")
        for name, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3}x {name}")
    if unmeasured:
        print(f"\n  {unmeasured} case(s) were never served -- unmeasured, not wrong (ADR-0014)")


def _write(
    args: argparse.Namespace,
    graded: list[Graded],
    unmeasured: int,
    errors: dict[str, int],
    elapsed: float,
    model: str,
) -> None:
    if args.out is None:
        return
    args.out.write_text(
        json.dumps(
            {
                "model": model,
                "pass_mark": args.pass_mark,
                "measured": len(graded),
                "unmeasured": unmeasured,
                "errors": errors,
                "wall_seconds": round(elapsed, 1),
                "cases": [asdict(g) for g in graded],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n  wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_evals", description=__doc__)
    p.add_argument("--cases", type=Path, default=Path("tests/fixtures/judge_cases.json"))
    p.add_argument("--variant", default=None, help="judge binding variant (default: primary)")
    p.add_argument("--pass-mark", type=float, default=DEFAULT_PASS_MARK)
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
