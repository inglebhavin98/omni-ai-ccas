"""Presidio engine tests, driven by a fake analyzer. No model load -- Rule 7."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ccas.redaction.presidio_onnx import ALLOW_LIST, ENTITY_MAP, PresidioEngine
from ccas.schemas.pii import PiiEntityType


@dataclass
class FakeResult:
    start: int
    end: int
    entity_type: str
    score: float


class FakeAnalyzer:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str] | None]] = []

    def analyze(
        self, text: str, language: str, entities: list[str] | None = None
    ) -> list[FakeResult]:
        self.calls.append((text, entities))
        return self.results


def engine(results: list[FakeResult], **kw: object) -> PresidioEngine:
    return PresidioEngine(analyzer=FakeAnalyzer(results), **kw)  # type: ignore[arg-type]


def test_labels_are_mapped_into_our_taxonomy() -> None:
    text = "Dana lives in Berlin"
    found = engine([FakeResult(0, 4, "PERSON", 0.9), FakeResult(14, 20, "LOCATION", 0.9)]).detect(
        text
    )
    assert {s.entity_type for s in found} == {PiiEntityType.PERSON, PiiEntityType.LOCATION}


def test_unmapped_labels_are_ignored_not_guessed() -> None:
    """A mislabelled entity still gets redacted, but the report would then be wrong."""
    assert engine([FakeResult(0, 4, "NRP", 0.99)]).detect("Kurd") == ()


def test_low_confidence_results_are_dropped() -> None:
    results = [FakeResult(0, 4, "PERSON", 0.4)]
    assert engine(results, min_score=0.6).detect("Dana") == ()
    assert len(engine(results, min_score=0.3).detect("Dana")) == 1


@pytest.mark.parametrize("word", ["SSN", "CVV", "IBAN", "Email", "PIN"])
def test_protocol_vocabulary_is_never_treated_as_an_entity(word: str) -> None:
    """Observed with en_core_web_sm: SSN and CVV as ORGANIZATION, Email as PERSON."""
    found = engine([FakeResult(0, len(word), "ORGANIZATION", 0.99)]).detect(word)
    assert found == ()


def test_a_real_name_is_not_allow_listed() -> None:
    assert len(engine([FakeResult(0, 4, "PERSON", 0.9)]).detect("Dana")) == 1


def test_allow_listing_needs_every_token_to_be_protocol_vocabulary() -> None:
    text = "Card Dana"
    assert len(engine([FakeResult(0, len(text), "PERSON", 0.9)]).detect(text)) == 1


def test_only_the_requested_labels_are_asked_for() -> None:
    analyzer = FakeAnalyzer([])
    PresidioEngine(entities=(PiiEntityType.PERSON,), analyzer=analyzer).detect("x")
    _, requested = analyzer.calls[0]
    assert requested == ["PERSON"]


def test_an_unbuildable_engine_reports_unavailable() -> None:
    """Never degrades to 'found nothing' -- the pipeline must see that it is missing."""
    broken = PresidioEngine(spacy_model="no_such_model_xyz")
    assert not broken.available
    assert broken.load_error is not None
    assert broken.detect("my name is Dana") == ()


def test_an_injected_analyzer_is_available() -> None:
    assert engine([]).available


def test_the_entity_map_only_contains_generic_entities() -> None:
    """Rule 1: the mapping must not encode a vertical's notion of an identifier."""
    assert set(ENTITY_MAP.values()) <= set(PiiEntityType)


def test_the_allow_list_is_lowercase() -> None:
    assert all(word == word.casefold() for word in ALLOW_LIST)
