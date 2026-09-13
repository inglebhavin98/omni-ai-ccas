"""Mine an L1/L2/L3 intent taxonomy from ingested call logs.

    uv run python scripts/mine_taxonomy.py --domain retail
    uv run python scripts/mine_taxonomy.py --domain retail --source aixblock --provider vllm
    uv run python scripts/mine_taxonomy.py --domain retail --dry-run

Reads redacted CallLog partitions from data/interim, clusters caller utterances, has an
LLM name the clusters, and writes domains/<domain>/taxonomy.json.

Refuses a corpus not admitted for mining (ADR-0004) -- notably Bitext, which is
supervision and would simply rediscover its own labels.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from ccas.config.domain_loader import DomainPackNotFoundError, load_pack
from ccas.config.settings import Settings
from ccas.ingestion.datasets import DatasetRole, DatasetRoleError, require_role
from ccas.ingestion.writer import read_jsonl
from ccas.llm.base import LLMProviderError
from ccas.llm.bindings import load_bindings
from ccas.llm.factory import build_provider
from ccas.mining.build_taxonomy import (
    DEFAULT_MIN_CHARS,
    TaxonomyBuilder,
    collect_utterances,
)
from ccas.mining.cluster import DEFAULT_MIN_SAMPLES, HdbscanClusterer
from ccas.mining.embedder import DEFAULT_MODEL, EmbeddingCache, build_embedder
from ccas.mining.labeler import IntentLabeler
from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import new_correlation_id
from ccas.schemas.call_log import CallLog, DatasetSource
from ccas.schemas.llm import ProviderName

LOG = get_logger("scripts.mine_taxonomy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mine_taxonomy", description=__doc__)
    parser.add_argument("--domain", default="retail")
    parser.add_argument(
        "--source",
        default=None,
        choices=[s.value for s in DatasetSource],
        help="restrict to one corpus; default is every mining-admitted partition",
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=None, help="default: the pack's taxonomy_ref")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/embeddings.npz"))
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=(
            "core-distance neighbourhood. The strongest lever on coverage: measured on "
            "6,000 utterances at min-cluster-size 25, inheriting min_cluster_size gave "
            "8.4%% where 5 gave 14.0%% and 3 gave 15.8%%"
        ),
    )
    parser.add_argument("--exemplars", type=int, default=8)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=(
            "shortest turn worth clustering. Raise it for a pause-segmented corpus, "
            "whose segments are fragments of a turn rather than turns (ADR-0013)"
        ),
    )
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument(
        "--provider",
        default=None,
        choices=[p.value for p in ProviderName],
        help="override the binding's provider",
    )
    parser.add_argument(
        "--allow-hashing-embedder",
        action="store_true",
        help="mine from lexical overlap instead of semantics. Throwaway taxonomies only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="cluster and report, but do not call an LLM or write a taxonomy",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_logs(interim: Path, source: str | None) -> list[CallLog]:
    logs: list[CallLog] = []
    for part in sorted(interim.rglob("*.jsonl.gz")):
        if source and f"source={source}" not in part.parts[-3:]:
            continue
        logs.extend(read_jsonl(part))
    return logs


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    correlation_id = new_correlation_id()

    try:
        pack = load_pack(settings.domains_dir, args.domain)
    except DomainPackNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    logs = _load_logs(args.interim, args.source)
    if not logs:
        print(
            f"error: no CallLog partitions under {args.interim}"
            + (f" for source={args.source}" if args.source else "")
            + "\n       run scripts/ingest.py first",
            file=sys.stderr,
        )
        return 2

    # Gate: every corpus present must be admitted for mining (ADR-0004).
    try:
        for corpus in sorted({log.source for log in logs}):
            require_role(corpus, DatasetRole.INTENT_MINING)
    except DatasetRoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    utterances = collect_utterances(logs, min_chars=args.min_chars)
    print(f"  corpora       {sorted({log.source.value for log in logs})}")
    print(f"  call logs     {len(logs)}")
    print(f"  utterances    {len(utterances)} caller turns long enough to mine")

    try:
        embedder = build_embedder(args.embedding_model, allow_fallback=args.allow_hashing_embedder)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"  embedder      {embedder.name}")

    clusterer = HdbscanClusterer(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples
    )
    if not clusterer.available:
        print("error: hdbscan missing; run `uv sync --extra mining`", file=sys.stderr)
        return 3

    if args.dry_run:
        cache = EmbeddingCache(args.cache, embedder.name)
        vectors = cache.encode(embedder, [u.text for u in utterances])
        cache.save()
        result = clusterer.fit_predict(vectors)
        print(f"\n  clusters      {result.n_clusters}")
        print(f"  coverage      {result.coverage:.1%}  (noise {result.noise_ratio:.1%})")
        print(f"  sizes         {sorted(result.sizes().values(), reverse=True)[:15]}")
        print("\n  dry run: no LLM called, no taxonomy written.")
        return 0

    registry = load_bindings(settings.models_config)
    provider_name = ProviderName(args.provider) if args.provider else None
    binding = registry.resolve("taxonomy_labeler", provider_name)
    try:
        provider = build_provider(binding, settings)
    except LLMProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"  labeler       {binding.provider.value}:{binding.model}")

    builder = TaxonomyBuilder(
        embedder=embedder,
        clusterer=clusterer,
        labeler=IntentLabeler(
            provider,
            binding,
            tool_names=list(pack.tools_by_name),
            tool_arguments={
                name: [str(a) for a in (spec.input_schema.get("required") or [])]
                for name, spec in pack.tools_by_name.items()
            },
        ),
        thresholds=pack.quadrant_thresholds,
        exemplars_per_cluster=args.exemplars,
        min_chars=args.min_chars,
        cache=EmbeddingCache(args.cache, embedder.name),
    )

    LOG.info(
        "mine.start",
        correlation_id=correlation_id,
        domain=pack.domain,
        call_logs=len(logs),
        utterances=len(utterances),
        embedder=embedder.name,
        labeler=f"{binding.provider.value}:{binding.model}",
    )
    started = time.perf_counter()
    try:
        taxonomy = await builder.build(logs, domain=pack.domain, version=args.version)
    finally:
        await provider.aclose()
    elapsed_s = round(time.perf_counter() - started, 2)

    out = args.out or (settings.domains_dir / pack.domain / pack.taxonomy_ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(taxonomy.model_dump_json(indent=2), encoding="utf-8")

    LOG.info(
        "mine.end",
        correlation_id=correlation_id,
        elapsed_s=elapsed_s,
        nodes=len(taxonomy.nodes),
        leaves=len(taxonomy.leaves()),
        coverage=taxonomy.coverage,
        noise_ratio=taxonomy.noise_ratio,
        output=str(out),
    )

    print(f"\n  nodes         {len(taxonomy.nodes)}  ({len(taxonomy.leaves())} leaves)")
    print(f"  coverage      {taxonomy.coverage:.1%}  (noise {taxonomy.noise_ratio:.1%})")
    print(f"  written       {out}")
    print(f"  elapsed       {elapsed_s}s\n")
    for node in sorted(taxonomy.nodes, key=lambda n: n.intent_id):
        indent = "  " * (node.level - 1)
        print(
            f"  {indent}{node.intent_id:<46} {node.volume.share_of_total:>6.1%}  "
            f"{node.automation.quadrant.value}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(Path("logs/execution.log"), level=args.log_level, console=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
