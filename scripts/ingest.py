"""Ingest a corpus into redacted, contract-valid CallLog partitions.

    uv run python scripts/ingest.py --source synthetic --domain retail --limit 50
    uv run python scripts/ingest.py --source bitext --domain retail
    uv run python scripts/ingest.py --source aixblock --domain retail --path data/raw/aixblock

Refuses a corpus that is not admitted for transcript ingestion (ADR-0004), and refuses
to run at all if the batch redaction engines are unavailable (CLAUDE.md Rule 2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ccas.config.domain_loader import DomainPackNotFoundError, load_pack
from ccas.config.settings import Settings
from ccas.ingestion.datasets import DatasetRole, DatasetRoleError, raw_dir_name, require_role
from ccas.ingestion.normalizer import Normalizer, NormalizeStats
from ccas.ingestion.registry import adapter_for
from ccas.ingestion.writer import create_writer, partition_dir
from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import new_correlation_id
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.schemas.call_log import DatasetSource

LOG = get_logger("scripts.ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=[s.value for s in DatasetSource],
        help="which corpus to read",
    )
    parser.add_argument("--domain", default="retail", help="domain pack whose patterns to layer")
    parser.add_argument("--path", type=Path, default=None, help="input file or directory")
    parser.add_argument("--out", type=Path, default=Path("data/interim"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "parquet"])
    parser.add_argument(
        "--max-quarantine-rate",
        type=float,
        default=0.05,
        help="fail the run if more than this fraction of records are dropped",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(Path("logs/execution.log"), level=args.log_level, console=True)
    settings = Settings()
    correlation_id = new_correlation_id()
    source = DatasetSource(args.source)

    # Gate 1: is this corpus even admitted for ingestion?
    try:
        require_role(source, DatasetRole.TRANSCRIPT_INGESTION)
    except DatasetRoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        pack = load_pack(settings.domains_dir, args.domain)
    except DomainPackNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Gate 2: batch redaction must be fully available. Never fail open.
    pipeline = build_pipeline(settings.redaction_policy, pack=pack, mode=RedactionMode.BATCH)
    if not pipeline.ready:
        missing = ", ".join(pipeline.unavailable_engines())
        detail = ""
        presidio = pipeline.engine_load_error("presidio_onnx")
        if presidio:
            detail = f"\n       cause: {presidio}"
        print(
            f"error: redaction engines unavailable ({missing}); refusing to ingest.{detail}\n"
            "       fix: uv sync --extra redaction   (this installs the pinned spaCy model)\n"
            "       or:  record a deliberate downgrade in configs/redaction_policy.yaml\n"
            "            by setting fail_closed: false on that engine -- which degrades\n"
            "            to regex-only detection and loses name redaction.",
            file=sys.stderr,
        )
        return 3

    adapter = adapter_for(source)
    path = args.path or (Path("data/raw") / raw_dir_name(source))
    if source is not DatasetSource.SYNTHETIC and not path.exists():
        print(f"error: no input at {path}\n       {adapter.expected_layout}", file=sys.stderr)
        return 2

    out_dir = partition_dir(args.out, source.value)
    stats = NormalizeStats()
    normalizer = Normalizer(pipeline)
    started = time.perf_counter()

    LOG.info(
        "ingest.start",
        correlation_id=correlation_id,
        source=source.value,
        domain=pack.domain,
        input=str(path),
        output=str(out_dir),
        limit=args.limit,
    )

    with create_writer(out_dir, args.format) as writer:
        for outcome in normalizer.normalize_many(adapter.read(path, args.limit), stats):
            if outcome.call_log is not None:
                writer.write(outcome.call_log)
            else:
                writer.quarantine(outcome)
                LOG.warning(
                    "ingest.quarantined",
                    correlation_id=correlation_id,
                    record_id=outcome.record_id,
                    reason=outcome.reason,
                )
        result = writer.close()

    elapsed_s = round(time.perf_counter() - started, 2)
    LOG.info(
        "ingest.end",
        correlation_id=correlation_id,
        elapsed_s=elapsed_s,
        **stats.as_dict(),  # counts and reason names only -- never content
    )

    print(f"\n  source        {source.value}")
    print(f"  domain        {pack.domain}")
    print(f"  written       {result.written} -> {result.path}")
    print(
        f"  quarantined   {result.quarantined}"
        + (f" -> {result.quarantine_path}" if result.quarantine_path else "")
    )
    print(f"  entities      {stats.entity_counts or 'none'}")
    print(f"  elapsed       {elapsed_s}s")

    if stats.seen == 0:
        print(f"\nerror: no records read.\n       {adapter.expected_layout}", file=sys.stderr)
        return 2
    if stats.quarantine_rate > args.max_quarantine_rate:
        print(
            f"\nerror: quarantine rate {stats.quarantine_rate:.1%} exceeds "
            f"{args.max_quarantine_rate:.1%}; reasons: {stats.reasons}",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
