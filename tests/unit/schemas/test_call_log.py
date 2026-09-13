from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    CallLog,
    Channel,
    DatasetSource,
    DtmfEvent,
    GoldLabels,
    RedactionStatus,
    Speaker,
    Utterance,
)
from tests.factories import make_report, redacted


def utterance(index: int, speaker: Speaker = Speaker.CALLER, **kw: object) -> Utterance:
    return Utterance(index=index, speaker=speaker, content=redacted(f"turn {index}"), **kw)  # type: ignore[arg-type]


def call_log(**kw: object) -> CallLog:
    base: dict[str, object] = {
        "call_id": CallLog.make_call_id(DatasetSource.BITEXT, "rec-1"),
        "source": DatasetSource.BITEXT,
        "source_record_id": "rec-1",
        "channel": Channel.CHAT,
        "utterances": (utterance(0), utterance(1, Speaker.BOT)),
        "redaction": make_report(),
    }
    base.update(kw)
    return CallLog(**base)  # type: ignore[arg-type]


def test_call_id_is_deterministic_per_source_record() -> None:
    first = CallLog.make_call_id(DatasetSource.BITEXT, "rec-1")
    assert first == CallLog.make_call_id(DatasetSource.BITEXT, "rec-1")
    assert first != CallLog.make_call_id(DatasetSource.NATCS, "rec-1")
    assert len(first) == 32


@pytest.mark.parametrize(
    "status", [RedactionStatus.DIRTY, RedactionStatus.UNVERIFIED, RedactionStatus.BYPASSED]
)
def test_call_log_refuses_a_non_clean_report(status: RedactionStatus) -> None:
    with pytest.raises(ValidationError, match="refusing construction"):
        call_log(redaction=make_report(status))


def test_call_log_refuses_an_unredacted_utterance() -> None:
    leaky = Utterance(
        index=0,
        speaker=Speaker.CALLER,
        content=redacted("my id is 123", RedactionStatus.UNVERIFIED),
    )
    with pytest.raises(ValidationError, match="carries redaction status"):
        call_log(utterances=(leaky,))


def test_utterance_indices_must_be_contiguous_from_zero() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        call_log(utterances=(utterance(0), utterance(2)))


def test_utterance_rejects_inverted_timing() -> None:
    with pytest.raises(ValidationError, match="precedes start_ms"):
        utterance(0, start_ms=500, end_ms=100)


def test_utterance_duration_is_derived() -> None:
    assert utterance(0, start_ms=100, end_ms=450).duration_ms == 350
    assert utterance(0).duration_ms is None


def test_long_dtmf_runs_must_be_redacted() -> None:
    with pytest.raises(ValidationError, match="must be redacted"):
        DtmfEvent(digits="4111111111111111", at_ms=10, redacted=False)
    assert DtmfEvent(digits="1", at_ms=10, redacted=False).digits == "1"


def test_caller_utterances_filters_by_speaker() -> None:
    log = call_log()
    assert len(log.utterances) == 2
    assert len(log.caller_utterances) == 1
    assert log.turn_count == 2


def test_domain_hint_is_a_free_slug_not_an_enum() -> None:
    # A brand-new vertical must not require a code change (CLAUDE.md Rule 1).
    assert call_log(domain_hint="municipal-utilities").domain_hint == "municipal-utilities"


def test_gold_labels_are_optional_and_slug_shaped() -> None:
    log = call_log(labels=GoldLabels(intent="track-order", category="shipping"))
    assert log.labels is not None
    assert log.labels.intent == "track-order"
