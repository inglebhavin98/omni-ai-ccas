from __future__ import annotations

from ccas.redaction.engine import DetectedSpan, merge_spans
from ccas.schemas.pii import PiiEntityType


def span(start: int, end: int, score: float = 0.9, **kw: object) -> DetectedSpan:
    base: dict[str, object] = {
        "start": start,
        "end": end,
        "entity_type": PiiEntityType.PHONE,
        "score": score,
        "engine": "regex",
    }
    base.update(kw)
    return DetectedSpan(**base)  # type: ignore[arg-type]


def test_non_overlapping_spans_all_survive() -> None:
    kept = merge_spans((span(0, 5), span(10, 15), span(20, 25)))
    assert [(s.start, s.end) for s in kept] == [(0, 5), (10, 15), (20, 25)]


def test_the_higher_score_wins_an_overlap() -> None:
    kept = merge_spans((span(0, 10, 0.6), span(2, 8, 0.99)))
    assert len(kept) == 1
    assert kept[0].score == 0.99


def test_at_equal_score_the_longer_match_wins() -> None:
    """A full card beats the digit run inside it: longer usually means more specific."""
    kept = merge_spans((span(4, 8, 0.9), span(0, 16, 0.9)))
    assert len(kept) == 1
    assert (kept[0].start, kept[0].end) == (0, 16)


def test_results_are_returned_in_document_order() -> None:
    kept = merge_spans((span(20, 25), span(0, 5, 0.99), span(10, 15, 0.7)))
    assert [s.start for s in kept] == [0, 10, 20]


def test_adjacent_spans_do_not_count_as_overlapping() -> None:
    assert len(merge_spans((span(0, 5), span(5, 10)))) == 2


def test_a_span_carries_no_matched_text() -> None:
    """Offsets travel; the original does not."""
    fields = set(DetectedSpan.model_fields)
    assert "text" not in fields
    assert "matched" not in fields
