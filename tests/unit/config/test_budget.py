from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ccas.config.budget import LatencyBudget, StageBudgets, load_budget

REPO = Path(__file__).resolve().parents[3]
BUDGET_FILE = REPO / "configs" / "latency_budget.yaml"


def stages(**kw: int) -> StageBudgets:
    base = {
        "vad_ms": 100,
        "stt_ms": 180,
        "redact_ms": 3,
        "router_ms": 90,
        "tool_ms": 150,
        "llm_ttft_ms": 180,
        "tts_ttfb_ms": 120,
    }
    base.update(kw)
    return StageBudgets(**base)


def test_shipped_budget_file_parses() -> None:
    budget = load_budget(BUDGET_FILE)
    assert budget.budget_ms == 800
    assert budget.percentile == 95


def test_shipped_budget_matches_the_documented_ceiling() -> None:
    """CLAUDE.md Rule 3 states 800ms. If someone edits the file, this catches it."""
    raw = yaml.safe_load(BUDGET_FILE.read_text(encoding="utf-8"))
    assert raw["budget_ms"] == 800, "the 800ms ceiling changed without an ADR"


def test_stage_totals_are_deliberately_tight() -> None:
    budget = load_budget(BUDGET_FILE)
    assert budget.slack_ms < 0, "budget has slack; regressions would hide in it"
    assert budget.stages.total_ms == 823


def test_a_wildly_oversubscribed_budget_is_rejected() -> None:
    with pytest.raises(ValidationError, match="that is not a tight budget"):
        LatencyBudget(budget_ms=800, stages=stages(llm_ttft_ms=5000))


def test_exceeds_compares_against_the_named_stage() -> None:
    budget = load_budget(BUDGET_FILE)
    assert budget.exceeds("stt_ms", 250)
    assert not budget.exceeds("stt_ms", 150)


def test_exceeds_rejects_an_unknown_stage() -> None:
    with pytest.raises(KeyError, match="unknown latency stage"):
        load_budget(BUDGET_FILE).exceeds("vibes_ms", 1)


def test_new_ledger_inherits_the_ceiling() -> None:
    ledger = load_budget(BUDGET_FILE).new_ledger()
    assert ledger.budget_ms == 800
    assert not ledger.breached


def test_every_ledger_stage_has_a_budget() -> None:
    """A stage we time but never budget is a blind spot."""
    ledger_stages = set(load_budget(BUDGET_FILE).new_ledger().stages())
    assert ledger_stages == set(load_budget(BUDGET_FILE).stages.as_dict())
