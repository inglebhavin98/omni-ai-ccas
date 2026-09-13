from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccas.config.domain_loader import load_domain
from ccas.observability.logging import configure_logging
from ccas.observability.tracing import current_trace_context
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.session import LatencyLedger
from cli.demo import main
from cli.stages import StageContext, StageStatus, build_pipeline, run_pipeline

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path) -> None:
    configure_logging(tmp_path / "execution.log", console=False, force=True)


def context(domain: str = "retail") -> StageContext:
    return StageContext(
        text="where is my delivery",
        domain=load_domain(REPO / "domains", domain),
        trace=current_trace_context(),
        ledger=LatencyLedger(),
        config_dir=REPO / "configs",
        source=DatasetSource.LIVE_CAPTURE,
    )


def test_pipeline_covers_every_module() -> None:
    names = [s.module for s in build_pipeline()]
    assert "redaction.pipeline" in names
    assert "ingestion.normalizer" in names
    assert "orchestration.router" in names
    assert "graph.assembly" in names
    assert "voice.session" in names


def test_pending_stages_declare_the_phase_that_delivers_them() -> None:
    """A demo that hid its gaps would be worse than no demo."""
    for result in run_pipeline(context()):
        if result.status is StageStatus.PENDING:
            assert result.phase is not None, f"{result.name} is pending with no phase"


def test_live_stages_produce_real_output() -> None:
    live = [r for r in run_pipeline(context()) if r.ok]
    assert live, "no stage is live; the demo would be vacuous"
    for result in live:
        assert result.lines


def test_the_pack_stage_reflects_the_loaded_pack() -> None:
    results = {r.name: r for r in run_pipeline(context("retail"))}
    body = "\n".join(results["Domain pack"].lines)
    assert "retail" in body
    assert "pci_dss" in body


def test_the_demo_works_against_both_packs() -> None:
    """Rule 1 again: swapping the pack must not need a code change."""
    retail = "\n".join(line for r in run_pipeline(context("retail")) for line in r.lines)
    healthcare = "\n".join(line for r in run_pipeline(context("healthcare")) for line in r.lines)
    assert "pci_dss" in retail
    assert "hipaa" in healthcare


def test_pipeline_command_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet-log", "pipeline", "where is my delivery"]) == 0
    out = capsys.readouterr().out
    assert "stages live" in out


def test_gate_command_shows_every_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet-log", "gate"]) == 0
    out = capsys.readouterr().out
    assert out.count("refused:") == 3
    assert "4111" not in out, "the gate demo must not echo the sample card number"


def test_datasets_command_shows_a_real_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet-log", "datasets"]) == 0
    out = capsys.readouterr().out
    assert "not a valid corpus for" in out
    assert "permitted" in out


def test_budget_command_renders_the_ceiling(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet-log", "budget"]) == 0
    out = capsys.readouterr().out
    assert "800 ms ceiling" in out
    # The configured default provider, whichever it is -- not a hardcoded model id.
    assert "openrouter" in out


def test_pack_command_renders_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--quiet-log", "pack", "retail"]) == 0
    assert "get_order_status" in capsys.readouterr().out


def test_a_run_leaves_a_structured_audit_trail(tmp_path: Path) -> None:
    log_path = tmp_path / "execution.log"
    configure_logging(log_path, console=False, force=True)
    main(["--quiet-log", "pipeline", "where is my delivery"])
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    names = {e["event"] for e in events}
    assert {"demo.pipeline.start", "demo.stage", "demo.pipeline.end"} <= names
    assert all("correlation_id" in e for e in events if e["event"].startswith("demo."))


def test_the_audit_trail_never_contains_the_input(tmp_path: Path) -> None:
    """Rule 2: the demo logs metadata about the transcript, never the transcript."""
    log_path = tmp_path / "execution.log"
    configure_logging(log_path, console=False, force=True)
    main(["--quiet-log", "pipeline", "my card is 4111111111111111"])
    assert "4111111111111111" not in log_path.read_text(encoding="utf-8")
