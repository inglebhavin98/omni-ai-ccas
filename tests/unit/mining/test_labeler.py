from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.mining.labeler import (
    LABEL_SCHEMA,
    SYSTEM_PROMPT,
    ClusterBrief,
    ClusterLabel,
    IntentLabeler,
)
from ccas.schemas.llm import ModelBinding, ProviderName
from ccas.schemas.pii import RedactionStatus
from tests.factories import redacted
from tests.mining_stub import StubLabelerProvider

BINDING = ModelBinding(
    node="taxonomy_labeler", provider=ProviderName.VLLM, model="stub", stream=False
)


def brief(*texts: str, cluster_id: int = 0, size: int = 40) -> ClusterBrief:
    return ClusterBrief(
        cluster_id=cluster_id,
        size=size,
        share=0.2,
        exemplars=tuple(redacted(t) for t in texts),
    )


def labeler(**kw: object) -> IntentLabeler:
    return IntentLabeler(StubLabelerProvider(), BINDING, **kw)  # type: ignore[arg-type]


def test_a_brief_refuses_unredacted_exemplars() -> None:
    with pytest.raises(ValidationError, match="not egress-permitted"):
        ClusterBrief(
            cluster_id=0,
            size=1,
            share=0.1,
            exemplars=(redacted("x", RedactionStatus.UNVERIFIED),),
        )


def test_the_prompt_carries_only_redacted_content() -> None:
    request = labeler().build_request(brief("this is [PERSON_1] calling"), [])
    assert request.system is not None
    assert request.system.egress_permitted
    assert all(m.content.egress_permitted for m in request.messages)
    assert "[PERSON_1]" in request.messages[0].content.text


def test_the_prompt_lists_only_declared_tools() -> None:
    request = labeler(tool_names=["get_order_status"]).build_request(brief("hello"), [])
    body = request.messages[0].content.text
    assert "get_order_status" in body
    assert "invented" not in body


def test_the_prompt_carries_no_vertical_vocabulary() -> None:
    """Rule 1: the model must derive vocabulary from utterances, not from us."""
    lowered = SYSTEM_PROMPT.lower()
    for term in ("claim", "policy number", "patient", "prescription", "sku", "refund"):
        assert term not in lowered


def test_known_l1_slugs_are_offered_for_reuse() -> None:
    request = labeler().build_request(brief("hello"), ["payments", "status"])
    assert "payments, status" in request.messages[0].content.text


def test_the_response_schema_is_strict() -> None:
    assert LABEL_SCHEMA["additionalProperties"] is False
    assert "is_intent" in LABEL_SCHEMA["required"]


async def test_labelling_returns_a_validated_label() -> None:
    label = await labeler(tool_names=["get_order_status"]).label(
        brief("there is a charge I do not recognise")
    )
    assert label.l1_slug == "payments"
    assert label.l2_id == "payments.dispute"
    assert label.l3_id == "payments.dispute.resolve"


async def test_invented_tools_are_dropped() -> None:
    """The registry is the authority; a hallucinated name would fail much later."""
    label = await labeler(tool_names=["get_order_status"]).label(brief("where is it"))
    assert label.required_tools == ("get_order_status",)


async def test_every_tool_is_dropped_when_none_are_declared() -> None:
    label = await labeler(tool_names=[]).label(brief("where is it"))
    assert label.required_tools == ()


async def test_labels_are_assigned_largest_cluster_first() -> None:
    """Big clusters establish the L1 vocabulary that smaller ones then reuse."""
    stub = StubLabelerProvider()
    lab = IntentLabeler(stub, BINDING)
    labels = await lab.label_all(
        (
            brief("I need to change the details on my account", cluster_id=1, size=10),
            brief("there is a charge I do not recognise", cluster_id=2, size=90),
        )
    )
    assert [label.cluster_id for label in labels] == [2, 1]
    assert stub.calls == 2


def test_a_multi_level_slug_is_rejected() -> None:
    """ "billing.dispute" as one level would silently produce a depth-4 id."""
    with pytest.raises(ValidationError, match="single segment without dots"):
        ClusterLabel(
            cluster_id=0,
            l1_slug="billing",
            l1_label="B",
            l1_description="d",
            l2_slug="billing.dispute",
            l2_label="D",
            l2_description="d",
            l3_slug="status",
            l3_label="S",
            l3_description="d",
            complexity=0.2,
            feasibility=0.9,
            rationale="r",
        )


def test_ids_compose_into_a_dotted_path() -> None:
    label = ClusterLabel(
        cluster_id=0,
        l1_slug="billing",
        l1_label="B",
        l1_description="d",
        l2_slug="dispute",
        l2_label="D",
        l2_description="d",
        l3_slug="status",
        l3_label="S",
        l3_description="d",
        complexity=0.2,
        feasibility=0.9,
        rationale="r",
    )
    assert (label.l1_id, label.l2_id, label.l3_id) == (
        "billing",
        "billing.dispute",
        "billing.dispute.status",
    )


def test_the_model_name_is_recorded_for_provenance() -> None:
    assert labeler().model_name == "vllm:stub"
