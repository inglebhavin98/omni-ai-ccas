"""Synthetic object factories shared across the suite.

No real customer data lives here or anywhere under ``tests/`` (CLAUDE.md Rule 2).
"""

from __future__ import annotations

from ccas.schemas import RedactedText, RedactionReport, RedactionStatus, sha256_hex

__all__ = ["POLICY_VERSION", "make_report", "redacted", "sha256_hex"]

POLICY_VERSION = "test-policy-1"


def make_report(status: RedactionStatus = RedactionStatus.CLEAN) -> RedactionReport:
    if status is RedactionStatus.CLEAN:
        return RedactionReport.clean(
            engines_run=("regex",), policy_version=POLICY_VERSION, elapsed_us=120
        )
    return RedactionReport(
        status=status,
        engines_run=("regex",),
        policy_version=POLICY_VERSION,
        elapsed_us=120,
        leak_check_passed=False,
    )


def redacted(text: str, status: RedactionStatus = RedactionStatus.CLEAN) -> RedactedText:
    return RedactedText(text=text, report=make_report(status), source_sha256=sha256_hex(text))
