"""Fetch the three corpora from the Hugging Face Hub into ``data/raw/``.

    uv run python scripts/fetch_datasets.py --all
    uv run python scripts/fetch_datasets.py --corpus aixblock --include-large
    uv run python scripts/fetch_datasets.py --corpus bitext --corpus natcs

Files are streamed with ``httpx`` -- the locked HTTP client -- rather than the
``datasets`` library, which is not in the locked stack and would need an ADR (Rule 5).

Nothing here redacts anything, and that is the point: ``data/raw/`` is the input to
Module 2, never to a model. Redaction happens once, downstream, in ``scripts/ingest.py``
-> ``Normalizer``, which is the only path that can produce a ``CallLog`` -- and a
``CallLog`` cannot be built from unredacted text (Rule 2, Rule 9).

Each corpus is written in the shape its existing adapter already reads, so hydrating
the platform needs no change to ``src/ccas/``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import httpx
from aixblock_transcripts import ARCHIVES, DEFAULT_GAP_MS, MIN_TURNS, read_archive

from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import new_correlation_id

LOG = get_logger("scripts.fetch_datasets")

_RESOLVE: Final = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
#: Rule 3: every external call carries an explicit timeout. Reads are generous because a
#: single AIxBlock archive is hundreds of megabytes over a cold CDN edge.
_TIMEOUT: Final = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)
_CHUNK: Final = 1 << 20

AIXBLOCK_REPO: Final = "AIxBlock/92k-real-world-call-center-scripts-english"
BITEXT_REPO: Final = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_CSV: Final = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
NATCS_REPO: Final = "splevine/dstc11-intent"
NATCS_CSV: Final = "dstc11_one_intent.csv"

CORPORA: Final = ("aixblock", "bitext", "natcs")


def _cached_bytes(dest: Path) -> int:
    """Size of an already-complete download, or 0. Blocking; called in a thread."""
    return dest.stat().st_size if dest.exists() else 0


async def _download(client: httpx.AsyncClient, repo: str, path: str, dest: Path) -> Path:
    """Stream one Hub file to disk, never leaving a partial file behind."""
    cached = await asyncio.to_thread(_cached_bytes, dest)
    if cached > 0:
        print(f"  cached      {dest.name[:52]:<52} {cached / 1e6:6.0f} MB")
        return dest
    await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = _RESOLVE.format(repo=repo, path=quote(path, safe=""))
    written = 0
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as handle:
            async for chunk in response.aiter_bytes(_CHUNK):
                handle.write(chunk)
                written += len(chunk)
                _progress(dest.name, written, total)
    tmp.replace(dest)
    print(f"\r  downloaded  {dest.name[:52]:<52} {written / 1e6:6.0f} MB")
    return dest


def _progress(name: str, written: int, total: int) -> None:
    suffix = f"{100 * written / total:5.1f}%" if total else f"{written / 1e6:6.0f} MB"
    sys.stdout.write(f"\r  fetching    {name[:52]:<52} {suffix}")
    sys.stdout.flush()


def _write_jsonl(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{k: (v or "") for k, v in row.items() if k} for row in csv.DictReader(handle)]


async def fetch_aixblock(client: httpx.AsyncClient, args: argparse.Namespace) -> int:
    """Themed ZIP archives -> one JSONL per vertical, shaped for ``AixBlockAdapter``."""
    chosen = [a for a in ARCHIVES if args.include_large or not a.large]
    if args.domain:
        chosen = [a for a in chosen if a.output in args.domain]
    if not chosen:
        print("  nothing selected for aixblock", file=sys.stderr)
        return 0

    per_output: dict[str, list[str]] = {}
    totals: dict[str, int] = {}
    for archive in chosen:
        path = await _download(
            client, AIXBLOCK_REPO, archive.filename, args.cache / archive.filename
        )
        lines, stats = await asyncio.to_thread(read_archive, path, args.gap_ms, args.min_turns)
        per_output.setdefault(archive.output, []).extend(lines)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"    {archive.filename[:44]:<44} {stats['kept']:>5} kept "
            f"{stats['too_short']:>5} short {stats['turns']:>7} turns"
        )
        LOG.info(
            "fetch.archive",
            correlation_id=args.correlation_id,
            corpus="aixblock",
            archive=archive.filename,
            output=archive.output,
            campaign=archive.campaign,
            **stats,
        )

    written = 0
    for output, lines in sorted(per_output.items()):
        target = args.out / "aixblock" / f"{output}.jsonl"
        _write_jsonl(target, lines)
        written += len(lines)
        print(f"  wrote       {len(lines):>6} dialogues -> {target}")
    print(
        f"  summary     {totals.get('members', 0)} transcripts read, "
        f"{totals.get('too_short', 0)} below {args.min_turns} turns, "
        f"{totals.get('unparsable', 0)} unparsable"
    )
    return written


async def fetch_bitext(client: httpx.AsyncClient, args: argparse.Namespace) -> int:
    """One labelled CSV, kept as CSV *and* mirrored to JSONL.

    ``BitextAdapter`` reads CSV, so the CSV is what actually feeds the eval suites; the
    JSONL mirror exists for tooling that would rather not parse a 19 MB CSV.
    """
    out_dir = args.out / "bitext"
    csv_path = await _download(client, BITEXT_REPO, BITEXT_CSV, out_dir / "customer_support.csv")
    rows = _read_csv(csv_path)
    _write_jsonl(
        out_dir / "customer_support.jsonl", [json.dumps(r, ensure_ascii=False) for r in rows]
    )
    intents = {r.get("intent", "") for r in rows}
    categories = {r.get("category", "") for r in rows}
    print(f"  wrote       {len(rows):>6} labelled pairs -> {out_dir / 'customer_support.jsonl'}")
    print(f"  labels      {len(intents)} intents across {len(categories)} categories")
    LOG.info(
        "fetch.corpus",
        correlation_id=args.correlation_id,
        corpus="bitext",
        records=len(rows),
        intents=len(intents),
        categories=len(categories),
    )
    return len(rows)


async def fetch_natcs(client: httpx.AsyncClient, args: argparse.Namespace) -> int:
    """DSTC11 intent turns -> per-dialogue JSONL, shaped for ``NatcsAdapter``.

    The published rows are caller turns only, with no agent side and no timings, so these
    are caller trajectories rather than full turn-taking. That is a real limit of this
    release and is recorded rather than papered over: nothing here invents an agent reply
    or a timestamp the corpus does not contain.
    """
    out_dir = args.out / "natcs"
    csv_path = await _download(client, NATCS_REPO, NATCS_CSV, out_dir / NATCS_CSV)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(csv_path):
        grouped.setdefault(row.get("dialogue_id", ""), []).append(row)

    lines = [
        json.dumps(_natcs_record(dialogue_id, sorted(rows, key=lambda r: r.get("turn_id", ""))))
        for dialogue_id, rows in sorted(grouped.items())
        if dialogue_id
    ]
    _write_jsonl(out_dir / "dialogues.jsonl", lines)

    multi = sum(1 for rows in grouped.values() if len(rows) >= args.min_turns)
    print(f"  wrote       {len(lines):>6} dialogues -> {out_dir / 'dialogues.jsonl'}")
    print(f"  multi-turn  {multi} with >= {args.min_turns} turns (caller side only)")
    LOG.info(
        "fetch.corpus",
        correlation_id=args.correlation_id,
        corpus="natcs",
        dialogues=len(lines),
        multi_turn=multi,
    )
    return len(lines)


def _natcs_record(dialogue_id: str, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    industry = rows[0].get("industry", "")
    # The insurance subset numbers its dialogues from 0 while the others are prefixed,
    # so a bare upstream id is not unique across the corpus. ``id`` is what the adapter
    # reads; ``dialogue_id`` stays exactly as published.
    return {
        "id": f"{industry}_{dialogue_id}" if dialogue_id.isdigit() else dialogue_id,
        "dialogue_id": dialogue_id,
        "industry": industry,
        "turns": [
            {
                "speaker": (row.get("speaker_role") or "customer").lower(),
                "text": row.get("utterance", ""),
                "turn_id": row.get("turn_id", ""),
                "intents": row.get("intents", ""),
                "dialogue_acts": row.get("dialogue_acts", ""),
            }
            for row in rows
            if (row.get("utterance") or "").strip()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fetch_datasets", description=__doc__)
    parser.add_argument("--corpus", action="append", choices=CORPORA, default=None)
    parser.add_argument("--all", action="store_true", help=f"fetch all of {CORPORA}")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/raw/.archives"), help="downloaded archives"
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="restrict AIxBlock to archives feeding this output file; repeatable. "
        "NOTE: retail/healthcare are historical misnomers for insurance calls (ADR-0013)",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="also pull the multi-hundred-MB archives (medicare_inbound is 826 MB)",
    )
    parser.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS)
    parser.add_argument("--min-turns", type=int, default=MIN_TURNS)
    parser.add_argument("--log-level", default="INFO")
    return parser


async def run(args: argparse.Namespace) -> int:
    wanted = CORPORA if (args.all or not args.corpus) else tuple(dict.fromkeys(args.corpus))
    fetchers = {"aixblock": fetch_aixblock, "bitext": fetch_bitext, "natcs": fetch_natcs}
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers={"user-agent": "ai-ccas"}) as client:
        for name in wanted:
            print(f"\n{name}")
            try:
                await fetchers[name](client, args)
            except httpx.HTTPError as exc:
                print(f"error: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 3
    print("\n  raw corpora are unredacted by design; run scripts/ingest.py next.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(Path("logs/execution.log"), level=args.log_level, console=False)
    args.correlation_id = new_correlation_id()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
