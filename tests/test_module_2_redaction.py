"""Module gate: M2 — zero-leakage redaction.

Asserts the module is usable as a whole and that its guarantees hold end to end.
Granular tests live in ``tests/unit/redaction/``; the fuzz lives in
``tests/security/test_zero_leakage.py``; the budget gate in ``tests/latency/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.schemas.call_log import CallLog, DatasetSource, Utterance
from ccas.schemas.common import Channel, Speaker
from ccas.schemas.pii import RedactionStatus

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "configs" / "redaction_policy.yaml"

TRANSCRIPT = [
    "hi, this is Dana Whitfield",
    "my card is 4111 1111 1111 1111 and my ssn is 123-45-6789",
    "you can email me at dana@example.com or call 415-555-0142",
    "the reference is ORD-884210",
    "thanks, that's everything",
]

LEAKABLE = [
    "Dana Whitfield",
    "4111 1111 1111 1111",
    "123-45-6789",
    "dana@example.com",
    "415-555-0142",
    "ORD-884210",
]


@pytest.fixture(scope="module")
def pipeline():
    pack = load_pack(REPO / "domains", "retail")
    return build_pipeline(POLICY, pack=pack, presidio=None, mode=RedactionMode.REALTIME)


def test_the_pipeline_assembles_from_shipped_config(pipeline) -> None:
    assert pipeline.ready
    assert pipeline.policy.version


def test_a_whole_transcript_redacts_clean(pipeline) -> None:
    results = pipeline.redact_many(TRANSCRIPT)
    assert all(r.report.status is RedactionStatus.CLEAN for r in results)
    joined = " ".join(r.text for r in results)
    for secret in LEAKABLE:
        assert secret not in joined, f"{secret!r} survived redaction"


def test_redacted_output_can_build_a_call_log(pipeline) -> None:
    """The handshake between M2 and M1: output of one must satisfy the other's gate."""
    results = pipeline.redact_many(TRANSCRIPT)
    log = CallLog(
        call_id=CallLog.make_call_id(DatasetSource.SYNTHETIC, "gate-1"),
        source=DatasetSource.SYNTHETIC,
        source_record_id="gate-1",
        channel=Channel.VOICE,
        domain_hint="retail",
        utterances=tuple(
            Utterance(index=i, speaker=Speaker.CALLER, content=r) for i, r in enumerate(results)
        ),
        redaction=results[0].report,
    )
    assert log.turn_count == len(TRANSCRIPT)
    serialized = log.model_dump_json()
    for secret in LEAKABLE:
        assert secret not in serialized


def test_pack_patterns_are_layered_in(pipeline) -> None:
    result = pipeline.redact("the reference is ORD-884210")
    assert "ORD-884210" not in result.text
    assert "account_ref" in result.report.entity_counts


def test_both_modes_are_constructible() -> None:
    batch = build_pipeline(POLICY, mode=RedactionMode.BATCH)
    assert batch.mode is RedactionMode.BATCH
    assert batch.for_mode(RedactionMode.REALTIME).mode is RedactionMode.REALTIME


def test_the_module_never_returns_text_it_has_not_cleared(pipeline) -> None:
    """Whatever the status, a payload that is not egress-permitted carries no text."""
    for text in TRANSCRIPT:
        result = pipeline.redact(text)
        if not result.report.egress_permitted:
            assert result.text == ""


def test_reports_are_safe_to_log(pipeline) -> None:
    from ccas.observability.logging import get_logger

    report = pipeline.redact(TRANSCRIPT[1]).report
    # entity_counts is the shape Rule 2 permits: types and counts, never content.
    get_logger("gate").info(
        "redaction.done",
        entity_counts=report.entity_counts,
        elapsed_us=report.elapsed_us,
        status=report.status.value,
    )
    assert "4111" not in str(report.entity_counts)
