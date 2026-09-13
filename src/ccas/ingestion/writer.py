"""``CallLog`` sinks.

Output is partitioned by source and ingest date so a corpus can be re-ingested without
rewriting history, and so a role check (``require_role``) can be applied to a path.

Quarantined records get their own file holding ids and reasons only -- never text. The
quarantine file is the audit trail for what was dropped and why.
"""

from __future__ import annotations

import gzip
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from ccas.ingestion.normalizer import NormalizeOutcome
from ccas.schemas.call_log import CallLog

__all__ = [
    "CallLogWriter",
    "JsonlWriter",
    "WriteResult",
    "create_writer",
    "partition_dir",
    "read_jsonl",
]


@dataclass(slots=True)
class WriteResult:
    path: Path
    quarantine_path: Path | None
    written: int
    quarantined: int


def partition_dir(root: Path, source: str, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    return root / f"source={source}" / f"date={stamp}"


class CallLogWriter(ABC):
    """Append-only sink. Use as a context manager so partial runs still flush."""

    @abstractmethod
    def write(self, log: CallLog) -> None: ...

    @abstractmethod
    def quarantine(self, outcome: NormalizeOutcome) -> None: ...

    @abstractmethod
    def close(self) -> WriteResult: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class JsonlWriter(CallLogWriter):
    """Gzipped JSON lines. One object per ``CallLog``, schema-versioned by the model."""

    def __init__(self, directory: Path, basename: str = "part-000") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / f"{basename}.jsonl.gz"
        self._quarantine_path = directory / f"{basename}.quarantine.jsonl"
        # The writer owns these handles for its lifetime; close() releases them.
        self._handle: IO[str] = gzip.open(self._path, "wt", encoding="utf-8")  # noqa: SIM115
        self._quarantine_handle: IO[str] | None = None
        self._written = 0
        self._quarantined = 0

    def write(self, log: CallLog) -> None:
        self._handle.write(log.model_dump_json() + "\n")
        self._written += 1

    def quarantine(self, outcome: NormalizeOutcome) -> None:
        if self._quarantine_handle is None:
            # Opened lazily so a clean run leaves no empty quarantine file behind.
            self._quarantine_handle = self._quarantine_path.open("w", encoding="utf-8")
        # Ids and reasons only. The record that failed redaction is not written anywhere.
        line = json.dumps({"record_id": outcome.record_id, "reason": outcome.reason})
        self._quarantine_handle.write(line + "\n")
        self._quarantined += 1

    def close(self) -> WriteResult:
        self._handle.close()
        if self._quarantine_handle is not None:
            self._quarantine_handle.close()
        return WriteResult(
            path=self._path,
            quarantine_path=self._quarantine_path if self._quarantined else None,
            written=self._written,
            quarantined=self._quarantined,
        )


def create_writer(directory: Path, fmt: str = "jsonl", basename: str = "part-000") -> CallLogWriter:
    if fmt == "jsonl":
        return JsonlWriter(directory, basename)
    if fmt == "parquet":
        raise NotImplementedError(
            "the parquet sink needs pyarrow (`uv sync --extra mining`) and is not "
            "implemented yet; use --format jsonl. Tracked in docs/future-scoped-work.md"
        )
    raise ValueError(f"unknown output format {fmt!r}; expected 'jsonl' or 'parquet'")


def read_jsonl(path: Path) -> list[CallLog]:
    """Read a partition back. Round-tripping is what makes the sink trustworthy."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [CallLog.model_validate_json(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [CallLog.model_validate_json(line) for line in handle if line.strip()]
