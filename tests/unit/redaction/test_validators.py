from __future__ import annotations

import pytest

from ccas.redaction.validators import aba_routing, iban_mod97, luhn, resolve_validator


@pytest.mark.parametrize(
    "value",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "5500005555555559",
        "378282246310005",
    ],
)
def test_luhn_accepts_valid_cards(value: str) -> None:
    assert luhn(value)


@pytest.mark.parametrize(
    "value",
    [
        "4111111111111112",
        "1234567890123456",
        "0000000000000001",
        "411111111111",
        "41111111111111111111",
    ],
)
def test_luhn_rejects_invalid_or_wrong_length(value: str) -> None:
    assert not luhn(value)


def test_luhn_rejects_short_runs() -> None:
    """A 6-digit reference must not be mistaken for a card."""
    assert not luhn("123456")


@pytest.mark.parametrize("value", ["111000025", "021000021", "011401533"])
def test_aba_routing_accepts_valid_numbers(value: str) -> None:
    assert aba_routing(value)


@pytest.mark.parametrize("value", ["111000026", "123456789", "12345678"])
def test_aba_routing_rejects_invalid(value: str) -> None:
    assert not aba_routing(value)


@pytest.mark.parametrize(
    "value", ["GB82WEST12345698765432", "GB82 WEST 1234 5698 7654 32", "DE89370400440532013000"]
)
def test_iban_accepts_valid(value: str) -> None:
    assert iban_mod97(value)


@pytest.mark.parametrize("value", ["GB82WEST12345698765433", "XX00", "1234567890123456", ""])
def test_iban_rejects_invalid(value: str) -> None:
    assert not iban_mod97(value)


def test_resolve_returns_a_no_op_for_unvalidated_patterns() -> None:
    assert resolve_validator(None)("anything")


def test_resolve_rejects_an_unregistered_name() -> None:
    with pytest.raises(KeyError, match="unknown validator"):
        resolve_validator("verhoeff")
