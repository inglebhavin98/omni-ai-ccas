"""Per-stage latency gates (CLAUDE.md Rule 3).

Redaction is the first stage with a measurable implementation, and the only one that
sits on the call path *because of another rule* -- Rule 2 puts it there, so Rule 3 has
to pay for it. This file is what stops that cost drifting.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from ccas.config.budget import load_budget
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.redaction.policy import load_policy
from ccas.redaction.regex_engine import RegexEngine

REPO = Path(__file__).resolve().parents[2]
BUDGET = load_budget(REPO / "configs" / "latency_budget.yaml")

#: Both markers, as a list. A second `pytestmark = ...` silently overwrites the first,
#: which is how these ran in the default suite for a while after ADR-0018 froze voice.
#: They assert the *voice* budget's slices, so they are frozen with it.
pytestmark = [pytest.mark.voice, pytest.mark.latency]

#: Representative of what an ASR emits mid-call: short, mixed, mostly clean.
UTTERANCES = [
    "hi, this is Dana Whitfield and I need some help",
    "my card is 4111 1111 1111 1111",
    "the reference is ORD-884210, sent to dana@example.com",
    "I just want to know when it arrives, nothing else",
    "yes that's right, 415-555-0142, and my PIN is 4821",
    "no, the other one, the one from last week",
    "GB82 WEST 1234 5698 7654 32 is the account",
    "okay thanks, that's all I needed",
]


@pytest.fixture(scope="module")
def realtime() -> RedactionPipeline:
    return RedactionPipeline(
        load_policy(REPO / "configs" / "redaction_policy.yaml"),
        RegexEngine.from_path(),
        presidio=None,
        mode=RedactionMode.REALTIME,
    )


def _percentiles(pipeline: RedactionPipeline, rounds: int = 40) -> tuple[float, float]:
    for utterance in UTTERANCES:  # warm the regex cache
        pipeline.redact(utterance)
    samples: list[float] = []
    for _ in range(rounds):
        for utterance in UTTERANCES:
            started = time.perf_counter_ns()
            pipeline.redact(utterance)
            samples.append((time.perf_counter_ns() - started) / 1000)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.99)]


def test_realtime_redaction_fits_its_slice(realtime: RedactionPipeline) -> None:
    budget_us = BUDGET.stages.redact_ms * 1000
    median_us, p99_us = _percentiles(realtime)
    assert p99_us < budget_us, (
        f"realtime redaction p99 {p99_us:.0f}us exceeds its "
        f"{budget_us}us slice (median {median_us:.0f}us)"
    )


def test_realtime_redaction_keeps_real_headroom(realtime: RedactionPipeline) -> None:
    """A stage that only just fits on a fast laptop will not fit under load."""
    budget_us = BUDGET.stages.redact_ms * 1000
    _, p99_us = _percentiles(realtime)
    assert p99_us < budget_us / 5, (
        f"realtime redaction p99 {p99_us:.0f}us has less than 5x headroom against "
        f"{budget_us}us -- investigate before adding another pattern"
    )


def test_the_call_path_never_runs_full_ner(realtime: RedactionPipeline) -> None:
    """The measurement that forced the two-mode split: a spaCy pass is ~7ms p99,
    2.5x the slice, with the *small* model. See docs/adr/0007-two-mode-redaction.md."""
    assert realtime.mode is RedactionMode.REALTIME
    assert realtime.redact("hello").report.engines_run == ("regex",)


def test_the_reported_elapsed_time_is_populated(realtime: RedactionPipeline) -> None:
    """`elapsed_us` is what production alerts on; an unset value would hide a regression."""
    report = realtime.redact("my card is 4111 1111 1111 1111").report
    assert report.elapsed_us > 0


def test_clean_utterances_are_the_cheap_case(realtime: RedactionPipeline) -> None:
    """Most turns carry no PII at all, so the no-match path must not be the slow one."""
    for _ in range(20):
        realtime.redact("no, the other one, the one from last week")
    clean = realtime.redact("no, the other one, the one from last week").report.elapsed_us
    loaded = realtime.redact(
        "card 4111 1111 1111 1111 ssn 123-45-6789 dana@example.com"
    ).report.elapsed_us
    assert clean <= loaded * 3
