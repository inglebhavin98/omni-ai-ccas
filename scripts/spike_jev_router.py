"""Does a purpose-built Choice primitive route better than a chat model? (spike)

Runs the *same* held-out rows through TypeSafe `jev` that `eval_router.py` runs through
the configured chat model, and scores them with the *same* `score_router`, so the two
numbers are comparable rather than merely both numbers.

    uv run python scripts/spike_jev_router.py --domain retail --limit 30

Baseline to beat, measured 2026-09-19 on `openrouter_alt`
(`inclusionai/ling-3.0-flash-fin:free`), 30 rows:

    23/26 exact (88.5%), category 100.0%, 4 unmeasured, p95 3496 ms

Nothing here is wired into the graph. Adoption is a locked-stack change and needs an ADR
(Rule 5); Rule 6 would then still require a second variant naming a distinct model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccas.config.domain_loader import load_domain
from ccas.config.settings import Settings
from ccas.evals.router_accuracy import (
    RouterCase,
    RouterOutcome,
    holdout_cases,
    outcome_for_exception,
    score_router,
)
from ccas.ingestion.datasets import DatasetRole, DatasetSource, require_role
from ccas.llm.base import LLMProviderError
from ccas.llm.typesafe import TypeSafeClient, criteria_from_taxonomy
from ccas.observability.logging import configure_logging, get_logger
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.redaction.placeholder import PlaceholderVault

LOG = get_logger("scripts.spike_jev_router")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spike_jev_router", description=__doc__)
    p.add_argument("--domain", default="retail")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--domains", type=Path, default=Path("domains"))
    p.add_argument("--limit", type=int, default=30, help="rows to route. One call each")
    p.add_argument("--seed", type=int, default=17, help="must match eval_router.py to compare")
    p.add_argument(
        "--mode",
        choices=["flat", "hierarchical"],
        default="hierarchical",
        help="flat = one Choice over every leaf; hierarchical = one Choice per level",
    )
    p.add_argument("--out", type=Path, default=None)
    return p


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if settings.typesafe_api_key is None:
        print("error: TYPESAFE_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2

    loaded = load_domain(args.domains, args.domain)
    if not loaded.has_taxonomy:
        print(f"error: domain {args.domain!r} has no taxonomy to route into", file=sys.stderr)
        return 2

    require_role(DatasetSource.BITEXT, DatasetRole.NLU_GROUND_TRUTH)
    corpus = next((args.raw / DatasetSource.BITEXT.value).glob("*.jsonl"), None)
    if corpus is None:
        print(f"error: no .jsonl under {args.raw / DatasetSource.BITEXT.value}", file=sys.stderr)
        return 2

    # The same sampler eval_router.py uses, so both runs grade the same rows.
    cases = holdout_cases(corpus, args.limit, args.seed)
    if not cases:
        print("error: no held-out rows found", file=sys.stderr)
        return 2

    criteria = criteria_from_taxonomy(loaded.taxonomy)
    redaction = build_pipeline(
        Path("configs") / "redaction_policy.yaml", pack=loaded.pack, mode=RedactionMode.BATCH
    )
    vault = PlaceholderVault()
    client = TypeSafeClient(
        api_key=settings.typesafe_api_key.get_secret_value(),
        base_url=settings.typesafe_base_url,
        model=settings.typesafe_model,
    )

    print(f"  domain       {args.domain}  ({len(criteria)} intents in the choice set)")
    print(f"  corpus       {corpus.name}, held-out split only")
    print(f"  mode         {args.mode}")
    print(f"  routing      {len(cases)} rows through typesafe:{settings.typesafe_model}\n")

    outcomes: list[tuple[RouterCase, RouterOutcome]] = []
    errors: dict[str, int] = {}
    tokens = 0
    calls = 0
    wall = time.perf_counter()
    try:
        for case in cases:
            content = redaction.redact(case.utterance, redaction.new_allocator(vault))
            try:
                choice = (
                    await client.choose(content, criteria)
                    if args.mode == "flat"
                    else await client.choose_hierarchical(content, loaded.taxonomy)
                )
                calls += choice.calls
                outcome = RouterOutcome(
                    predicted=choice.intent_id,
                    confidence=choice.confidence,
                    latency_ms=choice.latency_ms,
                )
                tokens += choice.input_tokens + choice.output_tokens
            except (LLMProviderError, ValueError, TimeoutError) as exc:
                outcome = outcome_for_exception(exc)
                errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
            outcomes.append((case, outcome))
            mark = (
                "."
                if outcome.predicted == case.expected_intent
                else ("-" if outcome.unavailable else "~" if outcome.answered else "x")
            )
            print(mark, end="", flush=True)
    finally:
        await client.aclose()
    elapsed = time.perf_counter() - wall
    print("\n")

    report = score_router(outcomes)
    print(f"  {report.summary()}")
    print(f"  wall {elapsed:.1f}s, {tokens} tokens, {calls} API calls")
    print(f"  cost  ${tokens / 1_000_000 * 0.042:.4f} at $0.042/M input tokens\n")
    print(
        "  baseline (openrouter_alt, 2026-09-19): 23/26 exact (88.5%), category 100.0%, p95 3496 ms"
    )
    if errors:
        print("\n  failures by cause")
        for name, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>3}x  {name}")
    if report.measured and report.confusions:
        print("\n  confusions (expected -> predicted)")
        for expected, predicted, n in report.confusions[:8]:
            print(f"    {n:>3}x  {expected}  ->  {predicted}")

    LOG.info(
        "spike.jev_router",
        correlation_id=f"spike-{args.domain}",
        measured=report.measured,
        unmeasured=report.unmeasured,
        exact_accuracy=round(report.exact_accuracy, 4),
        category_accuracy=round(report.category_accuracy, 4),
        p95_latency_ms=report.p95_latency_ms,
        model=settings.typesafe_model,
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "model": settings.typesafe_model,
                    "measured": report.measured,
                    "unmeasured": report.unmeasured,
                    "exact_accuracy": report.exact_accuracy,
                    "category_accuracy": report.category_accuracy,
                    "p95_latency_ms": report.p95_latency_ms,
                    "wall_seconds": round(elapsed, 1),
                    "tokens": tokens,
                    "api_calls": calls,
                    "mode": args.mode,
                    "failures_by_cause": errors,
                    "confusions": [list(c) for c in report.confusions],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\n  wrote {args.out}")
    return 0 if report.measured else 3


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
