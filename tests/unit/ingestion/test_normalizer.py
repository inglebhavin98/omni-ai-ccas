from __future__ import annotations

from pathlib import Path

import pytest

from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.base import RawDtmf, RawRecord, RawTurn
from ccas.ingestion.normalizer import (
    DTMF_MASK_THRESHOLD,
    Normalizer,
    NormalizeStats,
)
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline, build_pipeline
from ccas.redaction.policy import load_policy
from ccas.redaction.presidio_onnx import PresidioEngine
from ccas.redaction.regex_engine import RegexEngine
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Channel, Speaker

REPO = Path(__file__).resolve().parents[3]
POLICY = REPO / "configs" / "redaction_policy.yaml"


def batch_pipeline_without_ner() -> RedactionPipeline:
    """Regex-only, but declared BATCH.

    Keeps the suite fast and model-independent while still exercising the normalizer's
    BATCH-mode requirement. A no-fail-closed policy stands in for a present model.
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
    return RedactionPipeline(
        relaxed, RegexEngine.from_path(), presidio=None, mode=RedactionMode.BATCH
    )


@pytest.fixture(scope="module")
def normalizer() -> Normalizer:
    return Normalizer(batch_pipeline_without_ner())


def record(*texts: str, dtmf: tuple[RawDtmf, ...] = ()) -> RawRecord:
    return RawRecord(
        source=DatasetSource.SYNTHETIC,
        record_id="rec-1",
        channel=Channel.VOICE,
        turns=tuple(
            RawTurn(speaker=Speaker.CALLER if i % 2 == 0 else Speaker.BOT, text=t)
            for i, t in enumerate(texts)
        ),
        dtmf=dtmf,
    )


def test_ingestion_refuses_realtime_mode() -> None:
    """Stored records are exactly what must not contain a name (ADR-0007)."""
    realtime = build_pipeline(POLICY, presidio=None, mode=RedactionMode.REALTIME)
    with pytest.raises(ValueError, match=r"must use RedactionMode\.BATCH"):
        Normalizer(realtime)


def test_a_record_becomes_a_clean_call_log(normalizer: Normalizer) -> None:
    outcome = normalizer.normalize(record("my card is 4111 1111 1111 1111", "thanks"))
    assert not outcome.quarantined
    log = outcome.call_log
    assert log is not None
    assert log.redaction.egress_permitted
    assert "4111" not in log.model_dump_json()


def test_utterance_indices_and_speakers_are_preserved(normalizer: Normalizer) -> None:
    log = normalizer.normalize(record("one", "two", "three")).call_log
    assert log is not None
    assert [u.index for u in log.utterances] == [0, 1, 2]
    assert [u.speaker for u in log.utterances] == [Speaker.CALLER, Speaker.BOT, Speaker.CALLER]


def test_one_allocator_per_record_gives_coreference(normalizer: Normalizer) -> None:
    log = normalizer.normalize(
        record("email me at dana@example.com", "yes, dana@example.com is right")
    ).call_log
    assert log is not None
    assert all("[EMAIL_1]" in u.content.text for u in log.utterances)


def test_tokens_do_not_carry_across_records(normalizer: Normalizer) -> None:
    """A placeholder must never mean two different people."""
    first = normalizer.normalize(record("email me at a@example.com")).call_log
    second = normalizer.normalize(record("email me at b@example.com")).call_log
    assert first is not None
    assert second is not None
    assert "[EMAIL_1]" in first.utterances[0].content.text
    assert "[EMAIL_1]" in second.utterances[0].content.text


def test_long_dtmf_runs_are_masked(normalizer: Normalizer) -> None:
    log = normalizer.normalize(record("ok", dtmf=(RawDtmf(digits="4821", at_ms=10),))).call_log
    assert log is not None
    event = log.dtmf_events[0]
    assert event.redacted
    assert event.digits == "****"
    assert len(event.digits) == DTMF_MASK_THRESHOLD


def test_short_dtmf_runs_are_kept(normalizer: Normalizer) -> None:
    log = normalizer.normalize(record("ok", dtmf=(RawDtmf(digits="1", at_ms=10),))).call_log
    assert log is not None
    assert log.dtmf_events[0].digits == "1"
    assert not log.dtmf_events[0].redacted


def test_a_record_that_fails_redaction_is_quarantined_not_weakened() -> None:
    """The worst possible artifact would be a partly redacted record that looks normal."""
    policy = load_policy(POLICY)
    broken = RedactionPipeline(
        policy,
        RegexEngine.from_path(),
        presidio=PresidioEngine(spacy_model="no_such_model_xyz"),
        mode=RedactionMode.BATCH,
    )
    outcome = Normalizer(broken).normalize(record("my card is 4111 1111 1111 1111"))
    assert outcome.quarantined
    assert outcome.call_log is None
    assert outcome.reason is not None
    assert outcome.reason.startswith("unverified:")


def test_the_quarantine_reason_carries_no_content() -> None:
    policy = load_policy(POLICY)
    broken = RedactionPipeline(
        policy,
        RegexEngine.from_path(),
        presidio=PresidioEngine(spacy_model="no_such_model_xyz"),
        mode=RedactionMode.BATCH,
    )
    outcome = Normalizer(broken).normalize(record("my card is 4111 1111 1111 1111"))
    assert outcome.reason is not None
    assert "4111" not in outcome.reason


def test_stats_roll_up_counts_and_reasons(normalizer: Normalizer) -> None:
    stats = NormalizeStats()
    list(normalizer.normalize_many(SyntheticAdapter(count=8, seed=11).read(), stats))
    assert stats.seen == 8
    assert stats.emitted + stats.quarantined == 8
    assert stats.as_dict()["seen"] == 8


def test_stats_are_safe_to_log(normalizer: Normalizer) -> None:
    stats = NormalizeStats()
    list(normalizer.normalize_many(SyntheticAdapter(count=8, seed=11).read(), stats))
    rendered = str(stats.as_dict())
    assert "@example.com" not in rendered
    assert "4111" not in rendered


def test_gold_labels_survive_normalisation() -> None:
    normalizer = Normalizer(batch_pipeline_without_ner())
    outcome = normalizer.normalize(next(iter(SyntheticAdapter(count=1, seed=4).read())))
    assert outcome.call_log is not None
    assert outcome.call_log.labels is not None
    assert outcome.call_log.labels.intent


def test_call_ids_are_stable_across_runs(normalizer: Normalizer) -> None:
    first = normalizer.normalize(record("hello")).call_log
    second = normalizer.normalize(record("hello")).call_log
    assert first is not None
    assert second is not None
    assert first.call_id == second.call_id
