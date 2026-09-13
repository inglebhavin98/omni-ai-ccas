from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas.pii import (
    PiiEntityType,
    RedactedText,
    RedactionReport,
    RedactionSpan,
    RedactionStatus,
    sha256_hex,
)


def span(**kw: object) -> RedactionSpan:
    base: dict[str, object] = {
        "start": 0,
        "end": 5,
        "entity_type": PiiEntityType.PERSON,
        "score": 0.99,
        "engine": "regex",
        "replacement": "[PERSON_1]",
    }
    base.update(kw)
    return RedactionSpan(**base)  # type: ignore[arg-type]


def test_span_end_must_exceed_start() -> None:
    with pytest.raises(ValidationError, match="must exceed start"):
        span(start=5, end=5)


def test_custom_entity_requires_a_label() -> None:
    with pytest.raises(ValidationError, match="custom_label is required"):
        span(entity_type=PiiEntityType.CUSTOM)


def test_custom_label_is_rejected_for_standard_entities() -> None:
    with pytest.raises(ValidationError, match="only valid when entity_type is CUSTOM"):
        span(custom_label="loyalty_ref")


def test_spans_detect_overlap() -> None:
    assert span(start=0, end=10).overlaps(span(start=5, end=15))
    assert not span(start=0, end=5).overlaps(span(start=5, end=10))


def test_clean_requires_a_passed_leak_check() -> None:
    with pytest.raises(ValidationError, match="leak_check_passed"):
        RedactionReport(
            status=RedactionStatus.CLEAN,
            engines_run=("regex",),
            policy_version="p1",
            elapsed_us=1,
            leak_check_passed=False,
        )


def test_clean_forbids_residual_patterns() -> None:
    with pytest.raises(ValidationError, match="residual_patterns"):
        RedactionReport(
            status=RedactionStatus.CLEAN,
            engines_run=("regex",),
            policy_version="p1",
            elapsed_us=1,
            leak_check_passed=True,
            residual_patterns=("national_id",),
        )


def test_clean_requires_an_engine_to_have_run() -> None:
    with pytest.raises(ValidationError, match="at least one engine"):
        RedactionReport(
            status=RedactionStatus.CLEAN,
            engines_run=(),
            policy_version="p1",
            elapsed_us=1,
            leak_check_passed=True,
        )


def test_non_clean_cannot_claim_a_passed_leak_check() -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        RedactionReport(
            status=RedactionStatus.DIRTY,
            engines_run=("regex",),
            policy_version="p1",
            elapsed_us=1,
            leak_check_passed=True,
        )


@pytest.mark.parametrize(
    "status", [RedactionStatus.DIRTY, RedactionStatus.UNVERIFIED, RedactionStatus.BYPASSED]
)
def test_only_clean_permits_egress(status: RedactionStatus) -> None:
    report = RedactionReport(
        status=status, engines_run=("regex",), policy_version="p1", elapsed_us=1
    )
    assert not report.egress_permitted


def test_unverified_factory_never_permits_egress() -> None:
    report = RedactionReport.unverified(policy_version="p1", reason="onnx_model_missing")
    assert not report.egress_permitted
    assert report.status is RedactionStatus.UNVERIFIED


def test_entity_counts_group_by_label() -> None:
    report = RedactionReport.clean(
        spans=(
            span(start=0, end=3),
            span(start=4, end=7, replacement="[PERSON_2]"),
            span(start=8, end=11, entity_type=PiiEntityType.PHONE, replacement="[PHONE_1]"),
        ),
        engines_run=("regex",),
        policy_version="p1",
        elapsed_us=5,
    )
    assert report.entity_counts == {"person": 2, "phone": 1}


def test_require_egress_raises_on_unredacted_text() -> None:
    text = "hello"
    dirty = RedactedText(
        text=text,
        report=RedactionReport(
            status=RedactionStatus.UNVERIFIED,
            engines_run=(),
            policy_version="p1",
            elapsed_us=0,
        ),
        source_sha256=sha256_hex(text),
    )
    with pytest.raises(PermissionError, match="egress blocked"):
        dirty.require_egress()


def test_require_egress_returns_text_when_clean() -> None:
    text = "hello [PERSON_1]"
    clean = RedactedText(
        text=text,
        report=RedactionReport.clean(engines_run=("regex",), policy_version="p1", elapsed_us=1),
        source_sha256=sha256_hex("hello Dana"),
    )
    assert clean.require_egress() == text


def test_source_hash_must_be_sha256_shaped() -> None:
    with pytest.raises(ValidationError):
        RedactedText(
            text="x",
            report=RedactionReport.clean(engines_run=("regex",), policy_version="p1", elapsed_us=1),
            source_sha256="deadbeef",
        )
