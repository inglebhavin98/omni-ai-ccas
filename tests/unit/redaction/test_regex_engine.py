from __future__ import annotations

import pytest

from ccas.redaction.policy import PatternSet
from ccas.redaction.regex_engine import RegexEngine
from ccas.schemas.domain import PatternSpec
from ccas.schemas.pii import PiiEntityType


@pytest.fixture(scope="module")
def engine() -> RegexEngine:
    return RegexEngine.from_path()


def entities(engine: RegexEngine, text: str) -> set[PiiEntityType]:
    return {s.entity_type for s in engine.detect(text)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("card 4111 1111 1111 1111", PiiEntityType.PAYMENT_CARD),
        ("ssn 123-45-6789", PiiEntityType.NATIONAL_ID),
        ("GB82 WEST 1234 5698 7654 32", PiiEntityType.IBAN),
        ("reach me at dana@example.com", PiiEntityType.EMAIL),
        ("call +1 415-555-0142", PiiEntityType.PHONE),
        ("call 415-555-0142", PiiEntityType.PHONE),
        ("see https://example.com/x", PiiEntityType.URL),
        ("host 192.168.1.10", PiiEntityType.IP),
        ("my PIN is 4821", PiiEntityType.CREDENTIAL),
        ("the CVV is 909", PiiEntityType.CREDENTIAL),
        ("date of birth 1984-03-11", PiiEntityType.DATE_OF_BIRTH),
        ("postcode SW1A 1AA", PiiEntityType.ADDRESS),
        ("this is Dana Whitfield", PiiEntityType.PERSON),
    ],
)
def test_each_entity_class_is_detected(
    engine: RegexEngine, text: str, expected: PiiEntityType
) -> None:
    assert expected in entities(engine, text)


def test_a_luhn_invalid_digit_run_is_not_a_card(engine: RegexEngine) -> None:
    """Reference numbers are 16 digits too. The checksum is what separates them."""
    assert PiiEntityType.PAYMENT_CARD not in entities(engine, "reference 1234567812345678")


def test_a_bad_iban_checksum_is_rejected(engine: RegexEngine) -> None:
    assert PiiEntityType.IBAN not in entities(engine, "GB82 WEST 1234 5698 7654 33")


def test_clean_text_yields_nothing(engine: RegexEngine) -> None:
    assert engine.detect("I just want to know when it arrives.") == ()


def test_the_name_rule_redacts_the_name_not_the_cue(engine: RegexEngine) -> None:
    text = "this is Dana Whitfield"
    span = next(s for s in engine.detect(text) if s.entity_type is PiiEntityType.PERSON)
    assert text[span.start : span.end] == "Dana Whitfield"


@pytest.mark.parametrize(
    "text",
    [
        "This is the second time I have called.",
        "It is Monday and I am frustrated.",
        "I'm calling about the thing.",
    ],
)
def test_the_name_rule_does_not_fire_on_ordinary_speech(engine: RegexEngine, text: str) -> None:
    """The cue is case-insensitive; the name is not. Over-redaction ruins transcripts."""
    assert PiiEntityType.PERSON not in entities(engine, text)


def test_leak_detection_reports_names_never_matches(engine: RegexEngine) -> None:
    leaks = engine.detect_leaks("my card is 4111 1111 1111 1111")
    assert "payment_card" in leaks
    assert all("4111" not in name for name in leaks)


def test_leak_detection_is_quiet_on_redacted_text(engine: RegexEngine) -> None:
    assert engine.detect_leaks("my card is [PAYMENT_CARD_1]") == ()


def test_a_pattern_asking_for_a_missing_group_is_rejected() -> None:
    spec = PatternSpec(
        name="broken", pattern=r"\d+", entity_type=PiiEntityType.ACCOUNT_REF, capture_group=1
    )
    with pytest.raises(ValueError, match="but defines 0"):
        RegexEngine(PatternSet(patterns=(spec,)))


def test_an_uncompilable_pattern_is_rejected() -> None:
    spec = PatternSpec(name="broken", pattern="(unclosed", entity_type=PiiEntityType.ORG)
    with pytest.raises(ValueError, match="does not compile"):
        RegexEngine(PatternSet(patterns=(spec,)))


def test_pack_patterns_layer_on_top_of_the_global_set() -> None:
    from ccas.redaction.policy import load_pattern_set
    from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH

    extra = PatternSpec(
        name="vessel_ref",
        pattern=r"\bVSL-\d{6}\b",
        entity_type=PiiEntityType.CUSTOM,
        custom_label="vessel_ref",
    )
    merged = load_pattern_set(GENERIC_PATTERNS_PATH).merged_with((extra,))
    found = RegexEngine(merged).detect("booking VSL-884210 please")
    assert found[0].custom_label == "vessel_ref"


def test_a_pack_may_not_shadow_a_global_pattern() -> None:
    from ccas.redaction.policy import load_pattern_set
    from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH

    shadow = PatternSpec(name="email", pattern=r"x", entity_type=PiiEntityType.EMAIL)
    with pytest.raises(ValueError, match="may not shadow"):
        load_pattern_set(GENERIC_PATTERNS_PATH).merged_with((shadow,))
