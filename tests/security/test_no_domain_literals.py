"""CLAUDE.md Rule 1: the core must carry no vertical's vocabulary.

This is the gate that keeps "domain-agnostic" honest. If it fails, the fix is a field
on ``DomainPack`` -- never a branch in ``src/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: The operator console renders pack data, so it must stay generic too.
SCANNED = (REPO / "src" / "ccas", REPO / "src" / "cli")

#: Terms that only make sense inside one vertical. Deliberately excludes engineering
#: words that merely sound domain-ish -- "policy" (RetryPolicy, redaction policy),
#: "account" (ACCOUNT_REF), "iban"/"card" (generic PII entity types) -- because banning
#: those would produce noise rather than signal.
BANNED = [
    # healthcare / payer
    "patient",
    "diagnosis",
    "deductible",
    "copay",
    "coinsurance",
    "formulary",
    "prior_auth",
    "member_id",
    "ehr",
    "icd10",
    "cpt_code",
    "prescription",
    "clinical",
    "eligibility",
    "provider_network",
    # retail / e-commerce
    "sku",
    "refund",
    "cart",
    "checkout",
    "shipment",
    "tracking_number",
    "order_id",
    "order_number",
    "loyalty",
    # banking / insurance
    "overdraft",
    "mortgage",
    "premium",
    "policyholder",
    "claim",
    "chargeback",
    "routing_number",
    # telecom / utilities
    "roaming",
    "data_plan",
    "meter_reading",
    "tariff",
]

ALLOWED_PATHS: set[str] = set()
"""Escape hatch for a file that legitimately needs a term. Empty by design."""


def _sources() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for root in SCANNED:
        found.extend((p, p.relative_to(REPO).as_posix()) for p in sorted(root.rglob("*.py")))
    return found


def test_the_scan_actually_sees_source_files() -> None:
    assert len(_sources()) > 10, "domain-literal scan found no sources -- gate is inert"


@pytest.mark.security
@pytest.mark.parametrize("term", BANNED)
def test_core_contains_no_domain_vocabulary(term: str) -> None:
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    offenders: list[str] = []
    for path, rel in _sources():
        if rel in ALLOWED_PATHS:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"domain term {term!r} leaked into the core (CLAUDE.md Rule 1).\n"
        "Add a DomainPack field instead of a branch.\n" + "\n".join(offenders)
    )
