from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ccas.observability.logging import (
    ForbiddenLogFieldError,
    configure_logging,
    get_logger,
)
from tests.factories import redacted


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    path = tmp_path / "execution.log"
    configure_logging(path, console=False, force=True)
    return path


def test_events_are_json_lines(log_file: Path) -> None:
    get_logger("test").info("stage.done", stage="route", elapsed_ms=42)
    (event,) = read_events(log_file)
    assert event["event"] == "stage.done"
    assert event["stage"] == "route"
    assert event["elapsed_ms"] == 42


def test_every_event_carries_a_level_and_utc_timestamp(log_file: Path) -> None:
    get_logger("test").warning("slow.turn", stage="llm")
    (event,) = read_events(log_file)
    assert event["level"] == "warning"
    assert str(event["timestamp"]).endswith("Z")


@pytest.mark.parametrize(
    "field", ["utterance", "transcript", "raw_text", "digits", "ani", "slot_value"]
)
def test_raw_content_fields_are_refused(log_file: Path, field: str) -> None:
    """CLAUDE.md Rule 2: logs are the easiest place to leak by accident."""
    with pytest.raises(ForbiddenLogFieldError, match="raw-content field"):
        get_logger("test").info("turn.received", **{field: "my SSN is 123-45-6789"})


def test_the_refusal_names_the_offending_field(log_file: Path) -> None:
    with pytest.raises(ForbiddenLogFieldError, match="transcript"):
        get_logger("test").info("turn.received", transcript="...")


def test_redacted_text_is_unwrapped_when_clean(log_file: Path) -> None:
    get_logger("test").info("summary.built", summary=redacted("Caller asked about [REF_1]."))
    (event,) = read_events(log_file)
    assert event["summary"] == "Caller asked about [REF_1]."


def test_redacted_text_that_failed_is_never_written(log_file: Path) -> None:
    from ccas.schemas.pii import RedactionStatus

    get_logger("test").info(
        "summary.built", summary=redacted("SSN 123-45-6789", RedactionStatus.UNVERIFIED)
    )
    (event,) = read_events(log_file)
    assert event["summary"] == "<redaction-failed>"
    assert "123-45-6789" not in log_file.read_text(encoding="utf-8")


def test_safe_metadata_passes_through(log_file: Path) -> None:
    """Entity types and counts are the right level of detail to log."""
    get_logger("test").info(
        "redaction.done", entity_counts={"person": 2, "phone": 1}, elapsed_us=840
    )
    (event,) = read_events(log_file)
    assert event["entity_counts"] == {"person": 2, "phone": 1}


def test_paths_are_serialised(log_file: Path) -> None:
    get_logger("test").info("pack.loaded", path=Path("domains/retail/pack.yaml"))
    (event,) = read_events(log_file)
    assert event["path"] == "domains/retail/pack.yaml"


def test_the_log_directory_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "logs" / "execution.log"
    configure_logging(nested, console=False, force=True)
    get_logger("test").info("boot")
    assert nested.is_file()


def test_only_ccas_loggers_reach_the_json_file(log_file: Path) -> None:
    """spaCy and Presidio write plain-text warnings; one in this file would corrupt
    every consumer that parses it as JSON lines."""
    logging.getLogger("presidio-analyzer").warning("Fetching all recognizers for en")
    logging.getLogger("spacy").warning("Model not installed. Downloading...")
    get_logger("test").info("ours")

    lines = [line for line in log_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "ours"


def test_logger_names_are_namespaced(log_file: Path) -> None:
    get_logger("cli.demo").info("from.cli")
    get_logger("ccas.already").info("already.prefixed")
    assert len(read_events(log_file)) == 2
