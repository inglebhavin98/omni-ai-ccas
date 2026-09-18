"""Measure whether the chat router assigns the right intent (9.18).

    uv run --frozen python scripts/eval_router.py --domain retail --limit 40

Replays held-out corpus rows through the real router and compares the predicted intent to
the label the corpus publishes. Until this runs, "the chat channel works" means "it routes
without crashing".

Only rows the taxonomy was *not* derived from are eligible, and `score_router` refuses the
rest rather than trusting this script to have filtered them (ADR-0020).

Budget note: this costs one LLM call per row. The OpenRouter free tier allows 50 per day
per *account*, so `--limit` defaults low. A throttled row is reported as unmeasured rather
than wrong, so a partial run is still honest -- it just says less.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ccas.config.domain_loader import load_domain
from ccas.config.settings import Settings
from ccas.evals.router_accuracy import (
    RouterCase,
    RouterOutcome,
    naturalise,
    outcome_for_exception,
    score_router,
)
from ccas.graph.router import IntentRouter
from ccas.ingestion.datasets import DatasetRole, require_role
from ccas.llm.base import LLMProviderError
from ccas.llm.bindings import load_bindings
from ccas.llm.factory import build_provider
from ccas.mining.adopt import DERIVATION_SPLIT, split_of
from ccas.observability.logging import configure_logging, get_logger
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.redaction.placeholder import PlaceholderVault
from ccas.schemas.call_log import DatasetSource

LOG = get_logger("scripts.eval_router")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_router", description=__doc__)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--source", default="bitext", choices=[s.value for s in DatasetSource])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--domains", type=Path, default=Path("domains"))
    parser.add_argument("--limit", type=int, default=40, help="rows to route. One LLM call each")
    parser.add_argument("--variant", default=None, help="router variant; default is primary")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _holdout_cases(path: Path, limit: int, seed: int) -> list[RouterCase]:
    """Sample held-out rows, stratified by intent so rare ones are not lost to chance."""
    by_intent: dict[str, list[RouterCase]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row_id = f"{path.stem}:{index}"
            if split_of(row_id) == DERIVATION_SPLIT:
                continue
            row = json.loads(line)
            category, intent = row.get("category"), row.get("intent")
            text = (row.get("instruction") or "").strip()
            if not (category and intent and text):
                continue
            expected = f"{_slug(str(category))}.{_slug(str(intent))}"
            by_intent.setdefault(expected, []).append(
                RouterCase(row_id=row_id, utterance=naturalise(text), expected_intent=expected)
            )

    rng = np.random.default_rng(seed)
    picked: list[RouterCase] = []
    intents = sorted(by_intent)
    # Round-robin across intents: 40 rows drawn at random would miss a third of a
    # 27-intent taxonomy entirely, and a per-intent breakdown needs every intent present.
    for depth in itertools.count():
        if len(picked) >= limit:
            break
        added = False
        for intent in intents:
            rows = by_intent[intent]
            if depth < len(rows) and len(picked) < limit:
                picked.append(rows[int(rng.integers(0, len(rows)))])
                added = True
        if not added:
            break
    return picked[:limit]


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    loaded = load_domain(args.domains, args.domain)
    if not loaded.has_taxonomy:
        print(f"error: domain {args.domain!r} has no taxonomy to route into", file=sys.stderr)
        return 2

    source = DatasetSource(args.source)
    require_role(source, DatasetRole.NLU_GROUND_TRUTH)
    corpus = next((args.raw / source.value).glob("*.jsonl"), None)
    if corpus is None:
        print(f"error: no .jsonl under {args.raw / source.value}", file=sys.stderr)
        return 2

    cases = _holdout_cases(corpus, args.limit, args.seed)
    if not cases:
        print("error: no held-out rows found", file=sys.stderr)
        return 2

    bindings = load_bindings(Path("configs/models.yaml"))
    binding = (
        bindings.resolve("router", args.variant) if args.variant else bindings.resolve("router")
    )
    provider = build_provider(binding, settings)
    router = IntentRouter(provider, binding, loaded.taxonomy)
    # BATCH: this is offline, so there is no reason to use the weaker realtime pass.
    redaction = build_pipeline(
        Path("configs") / "redaction_policy.yaml", pack=loaded.pack, mode=RedactionMode.BATCH
    )
    vault = PlaceholderVault()

    print(f"  domain       {args.domain}  ({len(loaded.taxonomy.leaves())} intents)")
    print(f"  corpus       {corpus.name}, held-out split only")
    print("  utterances   template markup stripped ({{X}} -> x); no values invented")
    print(f"  routing      {len(cases)} rows through {binding.provider.value}:{binding.model}\n")

    outcomes = []
    errors: dict[str, int] = {}
    try:
        for n, case in enumerate(cases, 1):
            content = redaction.redact(case.utterance, redaction.new_allocator(vault))
            started = time.perf_counter_ns()
            try:
                prediction = await router.classify(content, history=())
                outcome = RouterOutcome(
                    predicted=prediction.intent_id or "<none>",
                    confidence=prediction.confidence,
                    latency_ms=(time.perf_counter_ns() - started) // 1_000_000,
                )
            except (LLMProviderError, ValueError, TimeoutError) as exc:
                outcome = outcome_for_exception(exc)
                # Which way it failed is the whole diagnosis, and it was missing from the
                # first live run's report -- 19 unserved rows were indistinguishable from
                # 19 unparseable ones without re-deriving it from wall clock.
                errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
            outcomes.append((case, outcome))
            # "-" is a row nobody served; it leaves the denominator, so it must not look
            # like a wrong answer on the progress line either.
            mark = (
                "."
                if outcome.predicted == case.expected_intent
                else ("-" if outcome.unavailable else "~" if outcome.answered else "x")
            )
            print(mark, end="", flush=True)
            if n % 40 == 0:
                print()
    finally:
        await provider.aclose()
    print("\n")

    report = score_router(outcomes)
    print(f"  {report.summary()}\n")
    if errors:
        print("  failures by cause")
        for name, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            # Not inferable from the name any more: whether a timeout left the denominator
            # depends on whether this model answered anything else (ADR-0021).
            served = (
                "never served"
                if "RateLimited" in name
                else "timed out; unmeasured only if this model answered elsewhere"
                if "Timeout" in name
                else "served, scored"
            )
            print(f"    {count:>3}x  {name}  ({served})")
        print()
    if report.measured:
        print("  weakest intents")
        for intent, score in report.worst_intents(5):
            print(f"    {score.accuracy:>6.0%}  {score.correct}/{score.total}  {intent}")
        if report.confusions:
            print("\n  most frequent confusions (expected -> predicted)")
            for expected, predicted, count in report.confusions[:5]:
                print(f"    {count:>3}x  {expected}  ->  {predicted}")

    LOG.info(
        "eval.router",
        correlation_id=f"eval-{args.domain}",
        measured=report.measured,
        unmeasured=report.unmeasured,
        exact_accuracy=round(report.exact_accuracy, 4),
        category_accuracy=round(report.category_accuracy, 4),
        p95_latency_ms=report.p95_latency_ms,
        model=binding.model,
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "domain": args.domain,
                    "model": binding.model,
                    "measured": report.measured,
                    "unmeasured": report.unmeasured,
                    "exact_accuracy": report.exact_accuracy,
                    "category_accuracy": report.category_accuracy,
                    "p95_latency_ms": report.p95_latency_ms,
                    "failures_by_cause": errors,
                    "per_intent": {i: [s.correct, s.total] for i, s in report.per_intent},
                    "confusions": [list(c) for c in report.confusions],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\n  wrote {args.out}")

    if report.unmeasured and not report.measured:
        print("\n  every row was rate limited -- this run measured nothing.", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
