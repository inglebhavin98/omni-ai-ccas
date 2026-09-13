"""Every contract must survive JSON serialization.

`CallLog` lands in parquet, `HandoffContext` crosses the wire to an agent desktop, and
`IntentTaxonomy` is written back into a domain pack -- so a model that cannot round-trip
is a latent data-loss bug rather than a style problem.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from ccas.schemas import (
    CallLog,
    Channel,
    DatasetSource,
    DomainPack,
    EvalReport,
    HandoffContext,
    HandoffReason,
    IntentTaxonomy,
    QueueSpec,
    SessionState,
    Speaker,
    TraceContext,
    Urgency,
    Utterance,
)
from tests.factories import make_report, redacted
from tests.unit.schemas.test_taxonomy import node, taxonomy


def a_call_log() -> CallLog:
    return CallLog(
        call_id=CallLog.make_call_id(DatasetSource.AIXBLOCK, "r-9"),
        source=DatasetSource.AIXBLOCK,
        source_record_id="r-9",
        channel=Channel.VOICE,
        domain_hint="retail",
        utterances=(
            Utterance(index=0, speaker=Speaker.CALLER, content=redacted("hello")),
            Utterance(index=1, speaker=Speaker.BOT, content=redacted("hi")),
        ),
        redaction=make_report(),
    )


def a_handoff(trace: TraceContext) -> HandoffContext:
    return HandoffContext(
        handoff_id="h-1",
        session_id="s-1",
        trace=trace,
        domain="retail",
        reason=HandoffReason.NEGATIVE_SENTIMENT,
        urgency=Urgency.HIGH,
        target_queue="tier-2",
        intent_confidence=0.91,
        intent_path=("billing", "billing.dispute"),
        summary=redacted("Caller is unhappy about [ACCOUNT_REF_1]."),
    )


def a_pack() -> DomainPack:
    return DomainPack(
        domain="retail",
        version="1.0.0",
        display_name="Retail",
        taxonomy_ref="taxonomy.json",
        queues=(QueueSpec(name="tier-1", display_name="Tier 1"),),
        greeting="How can I help?",
    )


def a_taxonomy() -> IntentTaxonomy:
    return taxonomy(node("billing", 1), node("billing.dispute", 2, parent="billing"))


@pytest.mark.parametrize(
    ("model_cls", "factory"),
    [
        (CallLog, a_call_log),
        (DomainPack, a_pack),
        (IntentTaxonomy, a_taxonomy),
    ],
)
def test_models_round_trip_through_json(
    model_cls: type[BaseModel], factory: Callable[[], BaseModel]
) -> None:
    original = factory()
    restored = model_cls.model_validate_json(original.model_dump_json())
    assert restored == original


def test_handoff_round_trips(trace: TraceContext) -> None:
    original = a_handoff(trace)
    assert HandoffContext.model_validate_json(original.model_dump_json()) == original


def test_session_state_round_trips(session: SessionState) -> None:
    restored = SessionState.model_validate_json(session.model_dump_json())
    assert restored.session_id == session.session_id
    assert restored.latency.total_rtt_ms == session.latency.total_rtt_ms


def test_eval_report_round_trips() -> None:
    original = EvalReport(report_id="r", domain="retail", suite="regression")
    assert EvalReport.model_validate_json(original.model_dump_json()) == original


def test_computed_fields_are_serialized() -> None:
    """`quadrant` and `total_rtt_ms` must reach consumers that never import our code."""
    payload = a_taxonomy().model_dump()
    assert "quadrant" in payload["nodes"][0]["automation"]
    assert (
        "total_rtt_ms"
        in SessionState.model_json_schema(mode="serialization")["$defs"]["LatencyLedger"][
            "properties"
        ]
    )
