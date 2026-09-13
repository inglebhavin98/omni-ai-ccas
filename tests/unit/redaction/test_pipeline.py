from __future__ import annotations

from pathlib import Path

import pytest

from ccas.redaction.pipeline import RedactionMode, RedactionPipeline, build_pipeline
from ccas.redaction.policy import load_policy
from ccas.redaction.presidio_onnx import PresidioEngine
from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH, RegexEngine
from ccas.schemas.pii import PiiEntityType, RedactionStatus, sha256_hex

REPO = Path(__file__).resolve().parents[3]
POLICY = REPO / "configs" / "redaction_policy.yaml"


@pytest.fixture(scope="module")
def realtime() -> RedactionPipeline:
    """Regex only -- what actually runs on the call path."""
    return RedactionPipeline(
        load_policy(POLICY),
        RegexEngine.from_path(GENERIC_PATTERNS_PATH),
        presidio=None,
        mode=RedactionMode.REALTIME,
    )


def test_structured_identifiers_are_removed(realtime: RedactionPipeline) -> None:
    result = realtime.redact("card 4111 1111 1111 1111 and ssn 123-45-6789")
    assert result.report.status is RedactionStatus.CLEAN
    assert "4111" not in result.text
    assert "123-45-6789" not in result.text
    assert result.report.entity_counts == {"payment_card": 1, "national_id": 1}


def test_clean_text_passes_through_unchanged(realtime: RedactionPipeline) -> None:
    text = "I just want to know when it arrives."
    result = realtime.redact(text)
    assert result.text == text
    assert result.report.egress_permitted
    assert result.report.spans == ()


def test_the_original_is_never_stored(realtime: RedactionPipeline) -> None:
    text = "ssn 123-45-6789"
    result = realtime.redact(text)
    assert result.source_sha256 == sha256_hex(text)
    assert "123-45-6789" not in result.model_dump_json()


def test_recorded_offsets_refer_to_the_original(realtime: RedactionPipeline) -> None:
    text = "my card is 4111 1111 1111 1111 thanks"
    span = realtime.redact(text).report.spans[0]
    assert text[span.start : span.end] == "4111 1111 1111 1111"


def test_one_entity_keeps_one_token_across_a_conversation(
    realtime: RedactionPipeline,
) -> None:
    turns = [
        "you can reach me at dana@example.com",
        "yes, dana@example.com, that's right",
    ]
    results = realtime.redact_many(turns)
    assert "[EMAIL_1]" in results[0].text
    assert "[EMAIL_1]" in results[1].text


def test_realtime_catches_a_self_identified_name_but_not_a_bare_one(
    realtime: RedactionPipeline,
) -> None:
    """The documented limit of regex-only mode.

    Self-identification is where a name actually first appears, and that is caught.
    A later bare mention is not -- which is exactly why nothing is stored or handed to
    a human without a BATCH pass. See docs/adr/0007-two-mode-redaction.md.
    """
    results = realtime.redact_many(["this is Dana Whitfield", "yes, Dana Whitfield, that's right"])
    assert "[PERSON_1]" in results[0].text
    assert "Dana Whitfield" in results[1].text


def test_separate_calls_do_not_share_an_allocator(realtime: RedactionPipeline) -> None:
    """A token must not collide across callers."""
    first = realtime.redact("this is Dana Whitfield")
    second = realtime.redact("this is Priya Raman")
    assert "[PERSON_1]" in first.text
    assert "[PERSON_1]" in second.text


def test_overlapping_detections_are_resolved(realtime: RedactionPipeline) -> None:
    """The 12-digit rule and the card rule both match; the card must win."""
    result = realtime.redact("card 4111 1111 1111 1111")
    assert [s.entity_type for s in result.report.spans] == [PiiEntityType.PAYMENT_CARD]


def test_realtime_mode_does_not_block_on_a_missing_model() -> None:
    """Presidio never runs on the call path, so its absence cannot fail a turn."""
    pipeline = RedactionPipeline(
        load_policy(POLICY),
        RegexEngine.from_path(),
        presidio=PresidioEngine(spacy_model="no_such_model_xyz"),
        mode=RedactionMode.REALTIME,
    )
    assert pipeline.ready
    assert pipeline.redact("hello").report.egress_permitted


def test_batch_mode_fails_closed_on_a_missing_model() -> None:
    """Rule 2: never fail open. A missing engine blocks egress, it does not skip."""
    pipeline = RedactionPipeline(
        load_policy(POLICY),
        RegexEngine.from_path(),
        presidio=PresidioEngine(spacy_model="no_such_model_xyz"),
        mode=RedactionMode.BATCH,
    )
    assert not pipeline.ready
    assert pipeline.unavailable_engines() == ("presidio_onnx",)

    result = pipeline.redact("this is Dana Whitfield, ssn 123-45-6789")
    assert result.report.status is RedactionStatus.UNVERIFIED
    assert not result.report.egress_permitted
    assert result.text == "", "a blocked payload must not carry the text"


def test_a_dirty_result_withholds_the_text() -> None:
    """If the leak check fails, the payload must not carry the thing that leaked."""
    from ccas.redaction.policy import PatternSet
    from ccas.schemas.domain import PatternSpec

    # A pattern that detects nothing, but whose leak rule matches everything.
    useless = PatternSpec(
        name="never_matches", pattern=r"\bZZZQQQ\b", entity_type=PiiEntityType.ACCOUNT_REF
    )
    leaky = PatternSet(patterns=(useless,), leak_detection=())
    engine = RegexEngine(leaky)
    pipeline = RedactionPipeline(
        load_policy(POLICY), engine, presidio=None, mode=RedactionMode.REALTIME
    )
    # A 20-digit run trips the detector's belt-and-braces long-digit-run rule.
    result = pipeline.redact("reference 12345678901234567890")
    assert result.report.status is RedactionStatus.DIRTY
    assert result.report.residual_patterns == ("long_digit_run",)
    assert result.text == ""


def test_for_mode_reuses_the_same_engines(realtime: RedactionPipeline) -> None:
    batch = realtime.for_mode(RedactionMode.BATCH)
    assert batch.mode is RedactionMode.BATCH
    assert realtime.mode is RedactionMode.REALTIME


def test_the_report_records_which_engines_ran(realtime: RedactionPipeline) -> None:
    assert realtime.redact("hello").report.engines_run == ("regex",)


def test_build_pipeline_layers_pack_patterns() -> None:
    from ccas.config.domain_loader import load_pack

    pack = load_pack(REPO / "domains", "retail")
    pipeline = build_pipeline(POLICY, pack=pack, presidio=None, mode=RedactionMode.REALTIME)
    result = pipeline.redact("my reference is ORD-884210")
    assert "ORD-884210" not in result.text
    assert result.report.egress_permitted
