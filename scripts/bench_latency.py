"""Per-stage latency report for the voice turn loop.

    uv run python scripts/bench_latency.py
    uv run python scripts/bench_latency.py --calls 50 --vendors measured

Runs scripted calls through the real loop -- VAD, redaction, routing, policies, tools,
grounding -- and reports the distribution per stage against configs/latency_budget.yaml.

By default the vendor stages (STT, LLM, TTS) are charged at their *budgeted* cost rather
than the stand-in's near-zero cost, because a report showing 3ms turns would be
flattering and useless. Pass --vendors measured to see the raw numbers instead.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.voice_harness import build_call

from ccas.config.budget import LatencyBudget, load_budget
from ccas.schemas.session import LatencyLedger

SCRIPT = ["where is my delivery", "ORD-884210"]


def charge_vendors(ledger: LatencyLedger, budget: LatencyBudget) -> LatencyLedger:
    stages = budget.stages
    return ledger.model_copy(
        update={
            "stt_ms": max(ledger.stt_ms, stages.stt_ms),
            "llm_ttft_ms": max(ledger.llm_ttft_ms, stages.llm_ttft_ms),
            "tts_ttfb_ms": max(ledger.tts_ttfb_ms, stages.tts_ttfb_ms),
        }
    )


async def collect(calls: int, budget: LatencyBudget, charge: bool) -> list[LatencyLedger]:
    ledgers: list[LatencyLedger] = []
    for index in range(calls):
        harness = build_call(SCRIPT, session_id=f"bench-{index}")
        await harness.run()
        for record in harness.session.turns:
            ledgers.append(charge_vendors(record.ledger, budget) if charge else record.ledger)
    return ledgers


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def report(ledgers: list[LatencyLedger], budget: LatencyBudget, charged: bool) -> int:
    stage_budgets = budget.stages.as_dict()
    print(f"\n  turns measured   {len(ledgers)}")
    print(f"  vendor stages    {'charged at budget' if charged else 'measured (stand-ins)'}")
    print(f"\n  {'stage':<14}{'p50':>7}{'p95':>7}{'p99':>7}{'budget':>9}   share")
    print(f"  {'-' * 52}")

    for stage, allowance in stage_budgets.items():
        values = [ledger.stages()[stage] for ledger in ledgers]
        p50, p95, p99 = (percentile(values, f) for f in (0.5, 0.95, 0.99))
        share = p95 / allowance if allowance else 0.0
        bar = "#" * min(20, round(share * 20))
        flag = "  OVER" if p95 > allowance else ""
        print(f"  {stage:<14}{p50:>7}{p95:>7}{p99:>7}{allowance:>9}   {bar}{flag}")

    totals = [ledger.total_rtt_ms for ledger in ledgers]
    p50, p95 = percentile(totals, 0.5), percentile(totals, 0.95)
    breaches = sum(1 for ledger in ledgers if ledger.breached)
    print(f"  {'-' * 52}")
    print(f"  {'TOTAL':<14}{p50:>7}{p95:>7}{percentile(totals, 0.99):>7}{budget.budget_ms:>9}")
    print(
        f"\n  median {statistics.median(totals):.0f}ms · p95 {p95}ms "
        f"· ceiling {budget.budget_ms}ms · breaches {breaches}/{len(ledgers)}"
    )
    print(
        "\n  Vendor stages are not measured against live services. Treat these as a "
        "floor,\n  not a forecast -- see docs/adr/0001-livekit-webrtc.md.\n"
    )
    return 1 if p95 > budget.budget_ms else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench_latency", description=__doc__)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument(
        "--vendors",
        choices=["budgeted", "measured"],
        default="budgeted",
        help="charge STT/LLM/TTS at their budget, or report the stand-ins' real cost",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/latency_budget.yaml"))
    args = parser.parse_args(argv)

    budget = load_budget(args.config)
    ledgers = asyncio.run(collect(args.calls, budget, args.vendors == "budgeted"))
    if not ledgers:
        print("error: no turns were measured", file=sys.stderr)
        return 2
    return report(ledgers, budget, args.vendors == "budgeted")


if __name__ == "__main__":
    raise SystemExit(main())
