"""The eval gates must leave a record (docs/future-scoped-work.md 6.7).

Testing the harness rather than the code, which is unusual and is the point: the Rule 6
gate passed for the first time and recorded nothing, so the harness was the defect. A
green gate that cannot say what it measured is only slightly better than no gate.

None of this costs an LLM call -- it asserts the plumbing, not the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from ccas.observability.logging import get_logger

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / "logs" / "execution.log"


def _events_after(offset: int) -> list[dict[str, object]]:
    with LOG_PATH.open(encoding="utf-8") as handle:
        handle.seek(offset)
        return [json.loads(line) for line in handle if line.strip()]


def test_an_eval_event_reaches_the_shared_log() -> None:
    """The exact failure 6.7 describes: the event was emitted and went nowhere."""
    assert LOG_PATH.exists(), "the package fixture should have created the log"
    offset = LOG_PATH.stat().st_size

    get_logger("tests.evals").info(
        "eval.logging_probe",
        correlation_id="probe-6-7",
        measured=1,
    )

    events = _events_after(offset)
    probe = [e for e in events if e.get("event") == "eval.logging_probe"]
    assert probe, "an eval gate's structured log must land in logs/execution.log"
    assert probe[-1]["correlation_id"] == "probe-6-7"
    assert probe[-1]["measured"] == 1


def test_the_record_carries_a_correlation_id() -> None:
    """Rule 11: every event carries one, so a gate run can be tied to everything else
    that happened in the same session."""
    offset = LOG_PATH.stat().st_size
    get_logger("tests.evals").info("eval.logging_probe", correlation_id="probe-corr")
    assert all("correlation_id" in event for event in _events_after(offset))
