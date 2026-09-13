from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    AutomationScore,
    IntentNode,
    IntentTaxonomy,
    Quadrant,
    QuadrantThresholds,
    SlotSpec,
    SlotType,
    VolumeStats,
)


def score(volume_share: float = 0.2, complexity: float = 0.2) -> AutomationScore:
    return AutomationScore(
        feasibility=0.9,
        complexity=complexity,
        confidence=0.9,
        rationale="synthetic",
        volume_share=volume_share,
    )


def node(intent_id: str, level: int, parent: str | None = None, **kw: object) -> IntentNode:
    base: dict[str, object] = {
        "intent_id": intent_id,
        "level": level,
        "parent_id": parent,
        "label": intent_id,
        "description": "d",
        "volume": VolumeStats(utterance_count=10, call_count=5, share_of_total=0.2),
        "automation": score(),
    }
    base.update(kw)
    return IntentNode(**base)  # type: ignore[arg-type]


def taxonomy(*nodes: IntentNode, **kw: object) -> IntentTaxonomy:
    base: dict[str, object] = {
        "taxonomy_id": "t1",
        "domain": "retail",
        "version": "1.0.0",
        "nodes": nodes,
        "embedding_model": "bge-large-en-v1.5",
        "clusterer": "hdbscan",
        "labeler_model": "test",
        "coverage": 0.85,
        "noise_ratio": 0.15,
    }
    base.update(kw)
    return IntentTaxonomy(**base)  # type: ignore[arg-type]


def test_intent_id_depth_must_match_level() -> None:
    with pytest.raises(ValidationError, match="has depth"):
        node("billing.refund", 1)


def test_l1_must_not_have_a_parent() -> None:
    with pytest.raises(ValidationError, match="must not have a parent"):
        node("billing", 1, parent="root")


def test_child_must_be_nested_under_its_parent() -> None:
    with pytest.raises(ValidationError, match="not nested under"):
        node("billing.refund", 2, parent="shipping")


def test_deeper_levels_require_a_parent() -> None:
    with pytest.raises(ValidationError, match="requires a parent"):
        IntentNode(
            intent_id="billing.refund",
            level=2,
            label="x",
            description="d",
            volume=VolumeStats(utterance_count=1, call_count=1, share_of_total=0.1),
            automation=score(),
        )


def test_taxonomy_rejects_a_missing_parent() -> None:
    with pytest.raises(ValidationError, match="missing parent"):
        taxonomy(node("billing.refund", 2, parent="billing"))


def test_taxonomy_rejects_duplicate_intent_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate intent_id"):
        taxonomy(node("billing", 1), node("billing", 1))


def test_coverage_and_noise_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"must equal 1\.0"):
        taxonomy(node("billing", 1), coverage=0.5, noise_ratio=0.2)


def test_path_walks_l1_to_l3() -> None:
    tax = taxonomy(
        node("billing", 1),
        node("billing.refund", 2, parent="billing"),
        node("billing.refund.status", 3, parent="billing.refund"),
    )
    assert [n.intent_id for n in tax.path("billing.refund.status")] == [
        "billing",
        "billing.refund",
        "billing.refund.status",
    ]
    assert [n.intent_id for n in tax.leaves()] == ["billing.refund.status"]
    assert [n.intent_id for n in tax.children("billing")] == ["billing.refund"]


def test_resolve_raises_a_readable_error_for_unknown_intents() -> None:
    with pytest.raises(KeyError, match="unknown intent"):
        taxonomy(node("billing", 1)).resolve("nope")


@pytest.mark.parametrize(
    ("volume_share", "complexity", "expected"),
    [
        (0.30, 0.10, Quadrant.IMMEDIATE_MIGRATION),
        (0.30, 0.90, Quadrant.PHASED_AGENTIC),
        (0.01, 0.10, Quadrant.SELF_SERVICE_SCRIPTED),
        (0.01, 0.90, Quadrant.DIRECT_AGENT_ROUTE),
    ],
)
def test_quadrant_is_derived_from_volume_and_complexity(
    volume_share: float, complexity: float, expected: Quadrant
) -> None:
    assert score(volume_share, complexity).quadrant is expected


def test_quadrant_thresholds_are_pack_configurable() -> None:
    strict = AutomationScore(
        feasibility=0.9,
        complexity=0.4,
        confidence=0.9,
        rationale="r",
        volume_share=0.10,
        thresholds=QuadrantThresholds(volume_share=0.20, complexity=0.5),
    )
    assert strict.quadrant is Quadrant.SELF_SERVICE_SCRIPTED


def test_enum_slots_require_values() -> None:
    with pytest.raises(ValidationError, match="requires non-empty enum_values"):
        SlotSpec(name="reason", slot_type=SlotType.ENUM, elicitation_prompt="Why?")


def test_enum_values_are_rejected_for_non_enum_slots() -> None:
    with pytest.raises(ValidationError, match="only valid for ENUM"):
        SlotSpec(
            name="ref",
            slot_type=SlotType.IDENTIFIER,
            elicitation_prompt="Ref?",
            enum_values=("a",),
        )


def test_duplicate_slot_names_are_rejected() -> None:
    slot = SlotSpec(name="ref", slot_type=SlotType.IDENTIFIER, elicitation_prompt="Ref?")
    with pytest.raises(ValidationError, match="duplicate slot names"):
        node("billing", 1, slots=(slot, slot))


def test_required_slots_filters_optional_ones() -> None:
    required = SlotSpec(name="a", slot_type=SlotType.STRING, elicitation_prompt="A?")
    optional = SlotSpec(
        name="b", slot_type=SlotType.STRING, elicitation_prompt="B?", required=False
    )
    assert node("billing", 1, slots=(required, optional)).required_slots == (required,)
