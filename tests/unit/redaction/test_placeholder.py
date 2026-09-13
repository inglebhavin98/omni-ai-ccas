from __future__ import annotations

import pytest

from ccas.redaction.placeholder import PlaceholderAllocator
from ccas.schemas.pii import PiiEntityType


def test_the_same_entity_keeps_one_token() -> None:
    """Coreference: "is that the same address?" is unanswerable if it does not."""
    alloc = PlaceholderAllocator()
    first = alloc.allocate(PiiEntityType.PERSON, "Dana Whitfield")
    second = alloc.allocate(PiiEntityType.PERSON, "Dana Whitfield")
    assert first == second == "[PERSON_1]"
    assert alloc.issued == 1


def test_matching_is_case_and_whitespace_insensitive() -> None:
    alloc = PlaceholderAllocator()
    assert alloc.allocate(PiiEntityType.PERSON, "Dana Whitfield") == alloc.allocate(
        PiiEntityType.PERSON, "  dana whitfield "
    )


def test_distinct_entities_get_distinct_tokens() -> None:
    alloc = PlaceholderAllocator()
    assert alloc.allocate(PiiEntityType.PERSON, "Dana") == "[PERSON_1]"
    assert alloc.allocate(PiiEntityType.PERSON, "Priya") == "[PERSON_2]"
    assert alloc.issued == 2


def test_counters_are_per_label() -> None:
    alloc = PlaceholderAllocator()
    assert alloc.allocate(PiiEntityType.PERSON, "Dana") == "[PERSON_1]"
    assert alloc.allocate(PiiEntityType.PHONE, "+15550100") == "[PHONE_1]"
    assert alloc.counts() == {"PERSON": 1, "PHONE": 1}


def test_custom_labels_get_their_own_namespace() -> None:
    alloc = PlaceholderAllocator()
    token = alloc.allocate(PiiEntityType.CUSTOM, "X-99", custom_label="vessel_ref")
    assert token == "[VESSEL_REF_1]"


def test_the_same_surface_under_two_labels_does_not_collide() -> None:
    alloc = PlaceholderAllocator()
    assert alloc.allocate(PiiEntityType.PERSON, "Paris") != alloc.allocate(
        PiiEntityType.LOCATION, "Paris"
    )


def test_format_is_configurable() -> None:
    alloc = PlaceholderAllocator("<<{entity}:{n}>>")
    assert alloc.allocate(PiiEntityType.EMAIL, "a@b.com") == "<<EMAIL:1>>"


@pytest.mark.parametrize("bad", ["[{entity}]", "[{n}]", "static"])
def test_a_format_missing_a_placeholder_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="must contain"):
        PlaceholderAllocator(bad)


def test_counts_expose_no_surface_forms() -> None:
    """The allocator's summary must be safe to log."""
    alloc = PlaceholderAllocator()
    alloc.allocate(PiiEntityType.PERSON, "Dana Whitfield")
    assert "Dana" not in str(alloc.counts())
