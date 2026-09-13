"""Module gate: M1 — ingestion and normalisation.

Granular tests live in ``tests/unit/ingestion/``. This asserts the module works as a
whole, including the two handshakes that matter: with M2 (nothing unredacted survives)
and with the dataset role registry (corpora are not interchangeable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.datasets import DatasetRole, DatasetRoleError, require_role
from ccas.ingestion.normalizer import Normalizer, NormalizeStats
from ccas.ingestion.registry import adapter_for
from ccas.ingestion.writer import JsonlWriter, read_jsonl
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.redaction.policy import load_pattern_set, load_policy
from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH, RegexEngine
from ccas.schemas.call_log import DatasetSource

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "configs" / "redaction_policy.yaml"


@pytest.fixture(scope="module")
def normalizer() -> Normalizer:
    """Regex-only but pack-aware -- the same pattern set scripts/ingest.py assembles.

    NER is disabled so the gate stays fast and model-independent. The pack patterns are
    not optional: they are where corpus-specific references are caught, and leaving them
    out here would let the gate pass on a configuration production never uses.
    """
    policy = load_policy(POLICY)
    relaxed = policy.model_copy(
        update={
            "engines": tuple(
                e.model_copy(update={"enabled": False}) if e.name == "presidio_onnx" else e
                for e in policy.engines
            )
        }
    )
    pack = load_pack(REPO / "domains", "retail")
    patterns = load_pattern_set(GENERIC_PATTERNS_PATH).merged_with(pack.redaction_patterns)
    return Normalizer(
        RedactionPipeline(relaxed, RegexEngine(patterns), presidio=None, mode=RedactionMode.BATCH)
    )


def test_every_corpus_adapter_declares_its_layout() -> None:
    """A bad path should produce guidance, not 'no records found'."""
    for source in (DatasetSource.AIXBLOCK, DatasetSource.BITEXT, DatasetSource.NATCS):
        assert len(adapter_for(source).expected_layout) > 20


def test_ingestion_is_gated_on_corpus_role() -> None:
    require_role(DatasetSource.AIXBLOCK, DatasetRole.TRANSCRIPT_INGESTION)
    require_role(DatasetSource.NATCS, DatasetRole.TRANSCRIPT_INGESTION)
    with pytest.raises(DatasetRoleError):
        require_role(DatasetSource.BITEXT, DatasetRole.TRANSCRIPT_INGESTION)


def test_a_corpus_flows_end_to_end_to_a_partition(normalizer: Normalizer, tmp_path: Path) -> None:
    stats = NormalizeStats()
    with JsonlWriter(tmp_path) as writer:
        for outcome in normalizer.normalize_many(
            SyntheticAdapter(count=25, seed=42, pii_ratio=1.0).read(), stats
        ):
            if outcome.call_log is not None:
                writer.write(outcome.call_log)
            else:
                writer.quarantine(outcome)
        result = writer.close()

    assert stats.seen == 25
    assert result.written == stats.emitted
    assert stats.quarantine_rate == 0.0

    restored = read_jsonl(result.path)
    assert len(restored) == result.written
    assert all(log.redaction.egress_permitted for log in restored)


def test_nothing_pii_shaped_reaches_storage(normalizer: Normalizer, tmp_path: Path) -> None:
    """The whole point of M1+M2: what lands on disk is already clean."""
    import re

    with JsonlWriter(tmp_path) as writer:
        for outcome in normalizer.normalize_many(
            SyntheticAdapter(count=40, seed=7, pii_ratio=1.0).read()
        ):
            if outcome.call_log is not None:
                writer.write(outcome.call_log)
        result = writer.close()

    import gzip

    with gzip.open(result.path, "rt", encoding="utf-8") as handle:
        raw = handle.read()
    for name, pattern in (
        ("card", r"\b4\d{3} \d{4} \d{4} \d{4}\b"),
        ("email", r"[a-z]+@example\.com"),
        ("phone", r"415-555-\d{4}"),
        ("reference", r"ORD-\d{6}"),
    ):
        assert not re.search(pattern, raw), f"{name} survived into storage"
    assert re.search(r"\[[A-Z_]+_\d+\]", raw), "nothing was redacted -- check the fixtures"


def test_dtmf_is_masked_before_storage(normalizer: Normalizer) -> None:
    outcomes = list(
        normalizer.normalize_many(SyntheticAdapter(count=25, seed=3, pii_ratio=1.0).read())
    )
    events = [
        event
        for outcome in outcomes
        if outcome.call_log is not None
        for event in outcome.call_log.dtmf_events
    ]
    assert events, "the fixture should produce DTMF"
    assert all(set(e.digits) == {"*"} for e in events if e.redacted)


def test_a_quarantined_record_leaves_an_auditable_trace(
    normalizer: Normalizer, tmp_path: Path
) -> None:
    from ccas.ingestion.normalizer import NormalizeOutcome

    with JsonlWriter(tmp_path) as writer:
        writer.quarantine(NormalizeOutcome(record_id="rec-1", reason="unverified:engine"))
        result = writer.close()
    assert result.quarantine_path is not None
    entry = json.loads(result.quarantine_path.read_text().strip())
    assert set(entry) == {"record_id", "reason"}


def test_re_ingesting_the_same_corpus_is_idempotent(normalizer: Normalizer) -> None:
    """Deterministic call ids mean a re-run overwrites rather than duplicates."""
    first = [
        o.call_log.call_id
        for o in normalizer.normalize_many(SyntheticAdapter(count=10, seed=5).read())
        if o.call_log
    ]
    second = [
        o.call_log.call_id
        for o in normalizer.normalize_many(SyntheticAdapter(count=10, seed=5).read())
        if o.call_log
    ]
    assert first == second
    assert len(set(first)) == len(first)
