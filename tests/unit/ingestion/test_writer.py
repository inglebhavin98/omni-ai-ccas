from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ccas.ingestion.normalizer import NormalizeOutcome
from ccas.ingestion.writer import (
    JsonlWriter,
    create_writer,
    partition_dir,
    read_jsonl,
)
from ccas.schemas.call_log import CallLog, DatasetSource, Utterance
from ccas.schemas.common import Channel, Speaker
from tests.factories import make_report, redacted


def call_log(call_id: str = "log-000000000001") -> CallLog:
    return CallLog(
        call_id=call_id,
        source=DatasetSource.SYNTHETIC,
        source_record_id=call_id,
        channel=Channel.VOICE,
        utterances=(Utterance(index=0, speaker=Speaker.CALLER, content=redacted("hi [PERSON_1]")),),
        redaction=make_report(),
    )


def test_partitioning_is_by_source_and_date() -> None:
    when = datetime(2026, 3, 11, tzinfo=UTC)
    assert partition_dir(Path("data/interim"), "aixblock", when) == Path(
        "data/interim/source=aixblock/date=2026-03-11"
    )


def test_written_logs_round_trip(tmp_path: Path) -> None:
    with JsonlWriter(tmp_path) as writer:
        writer.write(call_log("log-000000000001"))
        writer.write(call_log("log-000000000002"))
        result = writer.close()

    assert result.written == 2
    restored = read_jsonl(result.path)
    assert [log.call_id for log in restored] == ["log-000000000001", "log-000000000002"]


def test_output_is_gzipped(tmp_path: Path) -> None:
    with JsonlWriter(tmp_path) as writer:
        writer.write(call_log())
        result = writer.close()
    assert result.path.suffix == ".gz"
    with gzip.open(result.path, "rt", encoding="utf-8") as handle:
        assert handle.read().strip()


def test_a_clean_run_leaves_no_quarantine_file(tmp_path: Path) -> None:
    with JsonlWriter(tmp_path) as writer:
        writer.write(call_log())
        result = writer.close()
    assert result.quarantine_path is None
    assert not list(tmp_path.glob("*.quarantine.jsonl"))


def test_quarantine_records_ids_and_reasons_only(tmp_path: Path) -> None:
    """The record that failed redaction is not written anywhere."""
    with JsonlWriter(tmp_path) as writer:
        writer.quarantine(NormalizeOutcome(record_id="rec-9", reason="dirty:payment_card"))
        result = writer.close()

    assert result.quarantined == 1
    assert result.quarantine_path is not None
    entry = json.loads(result.quarantine_path.read_text().strip())
    assert entry == {"record_id": "rec-9", "reason": "dirty:payment_card"}


def test_the_writer_closes_on_exception(tmp_path: Path) -> None:
    """A partial run must still leave a readable partition."""
    writer = JsonlWriter(tmp_path)
    try:
        with writer:
            writer.write(call_log())
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert read_jsonl(tmp_path / "part-000.jsonl.gz")


def test_the_directory_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    with JsonlWriter(nested) as writer:
        writer.write(call_log())
    assert nested.is_dir()


def test_parquet_is_refused_with_an_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="pyarrow"):
        create_writer(tmp_path, "parquet")


def test_an_unknown_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown output format"):
        create_writer(tmp_path, "avro")
