"""The latency gate, seeded in Phase 1.

Phases 2-5 add measured assertions (redaction p99, per-stage timings, end-to-end RTT).
Until then this suite guards the contract itself, so `make latency` is never vacuously
green and a budget edit cannot slip through unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ccas.config.budget import load_budget

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.latency


def test_the_budget_file_and_the_rulebook_agree() -> None:
    """CLAUDE.md Rule 3 states the ceiling in prose; configs state it in numbers."""
    budget = load_budget(REPO / "configs" / "latency_budget.yaml")
    rulebook = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    stated = re.search(r"audio round trip:\s*\*\*<=\s*(\d+)\s*ms", rulebook)
    assert stated is not None, "CLAUDE.md no longer states an audio round-trip ceiling"
    assert int(stated.group(1)) == budget.budget_ms


def test_every_stage_on_the_call_path_is_budgeted() -> None:
    budget = load_budget(REPO / "configs" / "latency_budget.yaml")
    assert set(budget.stages.as_dict()) == {
        "vad_ms",
        "stt_ms",
        "redact_ms",
        "router_ms",
        "tool_ms",
        "llm_ttft_ms",
        "tts_ttfb_ms",
    }


def test_realtime_redaction_stays_inside_its_slice() -> None:
    """Rule 2 runs on the call path, so Rule 3 has to pay for it."""
    budget = load_budget(REPO / "configs" / "latency_budget.yaml")
    assert budget.stages.redact_ms <= 5, "redaction has crept out of its microsecond slice"


def test_call_path_bindings_declare_a_latency_budget() -> None:
    from ccas.llm.bindings import load_bindings

    registry = load_bindings(REPO / "configs" / "models.yaml")
    for node in ("router", "task_agent"):
        for binding in registry.pair(node):
            assert binding.latency_budget_ms is not None, (
                f"call-path node {node!r} has an unbudgeted {binding.provider.value} binding"
            )
