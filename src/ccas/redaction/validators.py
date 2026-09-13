"""Post-match validators.

A regex that matches every 16-digit run also matches order numbers, tracking codes and
timestamps. Checksums turn a broad pattern into a precise one, which matters in both
directions: false negatives are a breach, and false positives make transcripts unusable
for the humans who read them.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["VALIDATORS", "aba_routing", "iban_mod97", "luhn", "resolve_validator"]


def _digits(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def luhn(value: str) -> bool:
    """Mod-10 check used by payment cards."""
    digits = _digits(value)
    if not 12 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def aba_routing(value: str) -> bool:
    """Nine-digit routing transit number checksum."""
    digits = _digits(value)
    if len(digits) != 9:
        return False
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    total = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
    return total % 10 == 0


def iban_mod97(value: str) -> bool:
    """ISO 13616 mod-97 check."""
    compact = "".join(value.split()).upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        numeric = "".join(str(int(c, 36)) if c.isalpha() else c for c in rearranged)
    except ValueError:
        return False
    if not numeric.isdigit():
        return False
    return int(numeric) % 97 == 1


def always(_value: str) -> bool:
    """Explicit no-op, for a pattern precise enough not to need a check."""
    return True


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "luhn": luhn,
    "aba_routing": aba_routing,
    "iban_mod97": iban_mod97,
    "always": always,
}


def resolve_validator(name: str | None) -> Callable[[str], bool]:
    if name is None:
        return always
    try:
        return VALIDATORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown validator {name!r}; registered: {sorted(VALIDATORS)}") from exc
