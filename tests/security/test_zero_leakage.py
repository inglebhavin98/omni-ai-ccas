"""Property-based zero-leakage fuzz (CLAUDE.md Rule 2).

Example-based tests prove the patterns we thought of work. This proves the pipeline does
not emit a generated identifier, across the whole shape of each identifier class -- which
is the only form of the claim worth making.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.redaction.policy import load_policy
from ccas.redaction.regex_engine import RegexEngine
from ccas.schemas.pii import RedactionStatus

REPO = Path(__file__).resolve().parents[2]
POLICY = REPO / "configs" / "redaction_policy.yaml"

pytestmark = pytest.mark.security

# Regex-only: the call path, and the mode a fuzz run can drive thousands of times.
PIPELINE = RedactionPipeline(
    load_policy(POLICY), RegexEngine.from_path(), presidio=None, mode=RedactionMode.REALTIME
)

FILLER = st.sampled_from(
    [
        "hi there",
        "I wanted to check on something",
        "can you help me with this",
        "sorry, one moment",
        "thanks for waiting",
        "it's been a few days now",
        "",
    ]
)


def _luhn_complete(prefix_digits: list[int]) -> str:
    """Append the check digit that makes a digit run Luhn-valid."""
    total = 0
    for index, digit in enumerate(reversed(prefix_digits)):
        value = digit
        if index % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return "".join(map(str, prefix_digits)) + str((10 - total % 10) % 10)


@st.composite
def payment_cards(draw: st.DrawFn) -> str:
    length = draw(st.sampled_from([12, 13, 15, 18]))
    body = draw(st.lists(st.integers(0, 9), min_size=length, max_size=length))
    digits = _luhn_complete(body)
    separator = draw(st.sampled_from(["", " ", "-"]))
    if not separator:
        return digits
    return separator.join(digits[i : i + 4] for i in range(0, len(digits), 4))


@st.composite
def national_ids(draw: st.DrawFn) -> str:
    area = draw(st.integers(1, 899).filter(lambda n: n != 666))
    group = draw(st.integers(1, 99))
    serial = draw(st.integers(1, 9999))
    separator = draw(st.sampled_from(["-", " "]))
    return f"{area:03d}{separator}{group:02d}{separator}{serial:04d}"


@st.composite
def emails(draw: st.DrawFn) -> str:
    local = draw(st.from_regex(r"\A[a-z][a-z0-9._%+-]{0,18}[a-z0-9]\Z"))
    domain = draw(st.from_regex(r"\A[a-z][a-z0-9-]{0,12}\Z"))
    tld = draw(st.sampled_from(["com", "org", "net", "co.uk", "io"]))
    return f"{local}@{domain}.{tld}"


@st.composite
def phone_numbers(draw: st.DrawFn) -> str:
    area = draw(st.integers(200, 999))
    exchange = draw(st.integers(200, 999))
    line = draw(st.integers(0, 9999))
    separator = draw(st.sampled_from(["-", ".", " "]))
    return f"{area}{separator}{exchange}{separator}{line:04d}"


SECRETS = st.one_of(payment_cards(), national_ids(), emails(), phone_numbers())


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(secret=SECRETS, before=FILLER, after=FILLER)
def test_a_generated_identifier_never_survives_redaction(
    secret: str, before: str, after: str
) -> None:
    result = PIPELINE.redact(f"{before} {secret} {after}".strip())
    assert result.report.status is RedactionStatus.CLEAN
    assert secret not in result.text


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(first=SECRETS, second=SECRETS)
def test_multiple_identifiers_in_one_utterance_all_go(first: str, second: str) -> None:
    result = PIPELINE.redact(f"it's {first} and also {second}")
    assert result.report.status is RedactionStatus.CLEAN
    assert first not in result.text
    assert second not in result.text


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(secret=SECRETS)
def test_a_repeated_identifier_gets_one_stable_token(secret: str) -> None:
    result = PIPELINE.redact(f"it is {secret}, I said {secret}")
    assert secret not in result.text
    tokens = {span.replacement for span in result.report.spans}
    assert len(tokens) == 1, "the same value must not produce two placeholders"


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(secret=SECRETS)
def test_a_report_never_carries_what_it_redacted(secret: str) -> None:
    """Reports are logged; their subjects are not."""
    report = PIPELINE.redact(f"my details are {secret}").report
    assert secret not in report.model_dump_json()


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.text(max_size=200))
def test_arbitrary_text_never_crashes_and_never_fails_open(text: str) -> None:
    """Whatever comes out of an ASR, the result is either CLEAN or blocked -- never a
    half-redacted payload presented as safe."""
    result = PIPELINE.redact(text)
    if result.report.egress_permitted:
        assert result.report.status is RedactionStatus.CLEAN
    else:
        assert result.text == ""


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.text(max_size=200))
def test_redaction_is_idempotent(text: str) -> None:
    """Redacting already-redacted text must not mangle the placeholders."""
    once = PIPELINE.redact(text)
    if not once.report.egress_permitted:
        return
    twice = PIPELINE.redact(once.text)
    assert twice.report.egress_permitted
    assert twice.text == once.text
