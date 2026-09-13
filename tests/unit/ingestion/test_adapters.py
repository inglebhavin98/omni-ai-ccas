from __future__ import annotations

from pathlib import Path

import pytest

from ccas.ingestion.adapters.aixblock import AixBlockAdapter
from ccas.ingestion.adapters.bitext import BitextAdapter
from ccas.ingestion.adapters.generic_csv import ColumnMapping, GenericCsvAdapter
from ccas.ingestion.adapters.natcs import NatcsAdapter
from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.datasets import DatasetRole
from ccas.ingestion.registry import adapter_for
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Speaker

CORPORA = Path(__file__).resolve().parents[2] / "fixtures" / "corpora"


# --------------------------------------------------------------------- bitext
def test_bitext_reads_labelled_single_turns() -> None:
    records = list(BitextAdapter().read(CORPORA / "bitext" / "sample.csv"))
    assert len(records) == 4, "the blank row must be skipped, not emitted"
    first = records[0]
    assert first.turns[0].speaker is Speaker.CALLER
    assert first.turns[1].speaker is Speaker.HUMAN_AGENT
    assert first.labels is not None
    assert first.labels.intent == "check_status"
    assert first.labels.category == "status"


def test_bitext_labels_are_slugified() -> None:
    """Bitext ships SCREAMING_SNAKE; our intent ids are dotted lowercase slugs."""
    records = list(BitextAdapter().read(CORPORA / "bitext" / "sample.csv"))
    for record in records:
        assert record.labels is not None
        assert record.labels.category is not None
        assert record.labels.category.islower()


def test_bitext_is_not_admitted_for_mining() -> None:
    assert not BitextAdapter().supports(DatasetRole.INTENT_MINING)
    assert BitextAdapter().supports(DatasetRole.NLU_GROUND_TRUTH)


# ------------------------------------------------------------------- aixblock
def test_aixblock_reads_structured_turns() -> None:
    records = {r.record_id: r for r in AixBlockAdapter().read(CORPORA / "aixblock")}
    assert records["ax-001"].turns[1].speaker is Speaker.CALLER
    assert "Dana Whitfield" in records["ax-001"].turns[1].text


def test_aixblock_splits_a_script_into_turns() -> None:
    records = {r.record_id: r for r in AixBlockAdapter().read(CORPORA / "aixblock")}
    speakers = [t.speaker for t in records["ax-002"].turns]
    assert speakers == [Speaker.HUMAN_AGENT, Speaker.CALLER, Speaker.CALLER, Speaker.HUMAN_AGENT]


def test_aixblock_handles_an_alternate_text_key() -> None:
    records = {r.record_id: r for r in AixBlockAdapter().read(CORPORA / "aixblock")}
    assert len(records["ax-003"].turns) == 2


def test_aixblock_skips_records_with_no_turns() -> None:
    records = {r.record_id: r for r in AixBlockAdapter().read(CORPORA / "aixblock")}
    assert "ax-004" not in records


def test_aixblock_is_the_mining_corpus() -> None:
    assert AixBlockAdapter().supports(DatasetRole.INTENT_MINING)


# ---------------------------------------------------------------------- natcs
def test_natcs_reads_multi_turn_dialogues_with_timings() -> None:
    records = {r.record_id: r for r in NatcsAdapter().read(CORPORA / "natcs")}
    first = records["nat-001"]
    assert len(first.turns) == 3
    assert first.turns[0].speaker is Speaker.CALLER
    assert first.turns[0].start_ms == 0
    assert first.turns[0].end_ms == 2400, "seconds must be normalised to ms"


def test_natcs_accepts_millisecond_timings_unchanged() -> None:
    records = {r.record_id: r for r in NatcsAdapter().read(CORPORA / "natcs")}
    assert records["nat-002"].turns[0].end_ms == 2300


def test_natcs_records_a_duration() -> None:
    records = {r.record_id: r for r in NatcsAdapter().read(CORPORA / "natcs")}
    assert records["nat-001"].duration_ms == 5900


def test_natcs_is_supervision_not_a_mining_corpus() -> None:
    """Its roles moved in ADR-0015; the adapter reads the same registry, so this follows."""
    adapter = NatcsAdapter()
    assert adapter.supports(DatasetRole.NLU_GROUND_TRUTH)
    assert not adapter.supports(DatasetRole.INTENT_MINING)
    assert not adapter.supports(DatasetRole.DIALOGUE_CONTEXT_BENCHMARK)


# ---------------------------------------------------------------- generic csv
def test_generic_csv_groups_rows_into_conversations() -> None:
    records = list(GenericCsvAdapter().read(CORPORA / "generic" / "sample.csv"))
    assert len(records) == 2
    assert len(records[0].turns) == 3
    assert records[0].turns[0].start_ms == 0


def test_generic_csv_without_grouping_yields_one_record_per_row() -> None:
    mapping = ColumnMapping(conversation_id=None)
    records = list(GenericCsvAdapter(mapping).read(CORPORA / "generic" / "sample.csv"))
    assert len(records) == 5


# ------------------------------------------------------------------ synthetic
def test_synthetic_is_deterministic() -> None:
    first = list(SyntheticAdapter(count=5, seed=99).read())
    second = list(SyntheticAdapter(count=5, seed=99).read())
    assert [r.texts for r in first] == [r.texts for r in second]


def test_a_different_seed_gives_a_different_corpus() -> None:
    a = list(SyntheticAdapter(count=5, seed=1).read())
    b = list(SyntheticAdapter(count=5, seed=2).read())
    assert [r.texts for r in a] != [r.texts for r in b]


def test_synthetic_respects_the_limit() -> None:
    assert len(list(SyntheticAdapter(count=50, seed=1).read(limit=7))) == 7


def test_limit_is_authoritative_for_a_generated_corpus() -> None:
    """Asking for 40 and silently getting the constructor default is a bad surprise."""
    assert len(list(SyntheticAdapter(count=20, seed=1).read(limit=40))) == 40


def test_synthetic_carries_pii_shapes_to_exercise_the_redactor() -> None:
    records = list(SyntheticAdapter(count=30, seed=5, pii_ratio=1.0).read())
    joined = " ".join(text for r in records for text in r.texts)
    assert "@example.com" in joined or "4" in joined
    assert any(r.dtmf for r in records)


def test_synthetic_carries_no_vertical_vocabulary() -> None:
    """Rule 1: a synthetic corpus encoding one vertical would hide core leakage."""
    joined = " ".join(
        text for r in SyntheticAdapter(count=40, seed=3).read() for text in r.texts
    ).lower()
    for term in ("claim", "policy number", "prescription", "deductible", "sku"):
        assert term not in joined


# ------------------------------------------------------------------- registry
@pytest.mark.parametrize(
    "source",
    [
        DatasetSource.AIXBLOCK,
        DatasetSource.BITEXT,
        DatasetSource.NATCS,
        DatasetSource.GENERIC_CSV,
        DatasetSource.SYNTHETIC,
    ],
)
def test_every_supported_source_has_an_adapter(source: DatasetSource) -> None:
    adapter = adapter_for(source)
    assert adapter.source is source
    assert adapter.expected_layout


def test_an_unsupported_source_raises_with_the_available_list() -> None:
    with pytest.raises(KeyError, match="available"):
        adapter_for(DatasetSource.LIVE_CAPTURE)


# --------------------------------------------------------------- raw records
def test_raw_records_never_render_their_text() -> None:
    """A traceback or a debugger watch must not become a leak."""
    record = next(iter(SyntheticAdapter(count=1, seed=1).read()))
    rendered = f"{record!r} {record.turns[0]!r} {record.turns[0]!s}"
    for text in record.texts:
        assert text not in rendered


def test_metadata_cannot_smuggle_free_text() -> None:
    from pydantic import ValidationError

    from ccas.ingestion.base import RawRecord, RawTurn

    with pytest.raises(ValidationError, match="free text must go in a turn"):
        RawRecord(
            source=DatasetSource.SYNTHETIC,
            record_id="x",
            turns=(RawTurn(speaker=Speaker.CALLER, text="hi"),),
            metadata={"notes": "x" * 200},
        )
