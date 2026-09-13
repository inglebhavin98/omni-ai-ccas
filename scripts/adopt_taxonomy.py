"""Adopt a labelled corpus's published label set as a domain taxonomy.

    uv run python scripts/adopt_taxonomy.py --domain retail --source bitext

This is not mining and must not be confused with it (ADR-0020). Rule 9 forbids clustering
a labelled corpus; taking the labels it publishes is a different operation, and the
artefact says so in `provenance`. Half the corpus defines the taxonomy and the other half
stays held out, so an evaluator cannot grade a router on the rows that defined its labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccas.config.domain_loader import load_pack
from ccas.ingestion.datasets import DatasetRoleError, raw_dir_name
from ccas.mining.adopt import adopt_taxonomy, holdout_rows
from ccas.observability.logging import configure_logging, get_logger
from ccas.schemas.call_log import DatasetSource

LOG = get_logger("scripts.adopt_taxonomy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adopt_taxonomy", description=__doc__)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--source", default="bitext", choices=[s.value for s in DatasetSource])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--domains", type=Path, default=Path("domains"))
    parser.add_argument("--out", type=Path, default=None, help="default: the pack's taxonomy_ref")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "read only the first N rows. NOT a sample -- Bitext is grouped by category, "
            "so 4,000 rows yields 5 intents of 27. Use for a smoke test, never for a "
            "taxonomy you intend to ship"
        ),
    )
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)

    source = DatasetSource(args.source)
    pack = load_pack(args.domains, args.domain)
    corpus = next((args.raw / raw_dir_name(source)).glob("*.jsonl"), None)
    if corpus is None:
        print(f"error: no .jsonl under {args.raw / raw_dir_name(source)}", file=sys.stderr)
        return 2

    try:
        taxonomy = adopt_taxonomy(
            corpus,
            domain=args.domain,
            source=source,
            limit=args.limit,
            version=args.version,
            slot_pii=pack.slot_pii,
            slot_aliases=pack.slot_aliases,
            intent_tools=pack.intent_tools,
            pack_tools=pack.tools_by_name,
        )
    except DatasetRoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.out or (args.domains / args.domain / pack.taxonomy_ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json.loads(taxonomy.model_dump_json()), indent=2) + "\n")

    rows = sum(1 for _ in corpus.open(encoding="utf-8"))
    held = len(holdout_rows(f"{corpus.stem}:{i}" for i in range(rows)))
    LOG.info(
        "adopt.done",
        correlation_id=taxonomy.taxonomy_id,
        domain=args.domain,
        source=source.value,
        intents=len(taxonomy.leaves()),
        slots=sum(len(n.slots) for n in taxonomy.nodes),
        derived_rows=rows - held,
        holdout_rows=held,
    )
    print(f"\n  wrote        {out}")
    print(f"  provenance   adopted from {source.value}, split {taxonomy.derivation_split!r}")
    print(f"  hierarchy    {len(taxonomy.nodes)} nodes, {len(taxonomy.leaves())} intents")
    print(f"  slots        {sum(len(n.slots) for n in taxonomy.nodes)}")
    bound = [n for n in taxonomy.nodes if n.required_tools]
    print(f"  tool-bound   {len(bound)} of {len(taxonomy.leaves())} intents can act")
    print(f"  rows         {rows - held} derived / {held} held out for evaluation")
    if args.limit:
        print(
            f"\n  WARNING: --limit {args.limit} truncates rather than samples. This corpus "
            f"is grouped\n           by category, so the taxonomy holds only the intents "
            f"that appear early."
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
