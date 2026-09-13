"""Composing prompts without punching a hole in Rule 2.

``LLMRequest`` only accepts ``RedactedText``, which is correct -- but a prompt is part
repo-authored instruction and part caller-derived content, and those have different
provenance. This module keeps the distinction explicit instead of letting call sites
invent their own bypass.

- ``authored()`` wraps a string written in this repository. It carries no caller data,
  so it is clean by construction. **Never** pass a runtime value to it.
- ``compose()`` interpolates already-redacted parts into an authored template and
  aggregates their reports. If any part is not egress-permitted, neither is the result,
  so a leak upstream cannot be laundered by string formatting.
"""

from __future__ import annotations

from collections.abc import Mapping

from ccas.schemas.pii import (
    RedactedText,
    RedactionReport,
    RedactionStatus,
    sha256_hex,
)

__all__ = ["AUTHORED_POLICY", "authored", "compose"]

AUTHORED_POLICY = "authored-1.0"


def authored(text: str) -> RedactedText:
    """Wrap repository-authored prompt text.

    Only for string literals and templates that live in the codebase. Passing a value
    that originated from a caller, a dataset, a tool result, or a config file a customer
    edits would defeat the whole gate.
    """
    return RedactedText(
        text=text,
        report=RedactionReport.clean(
            engines_run=("manual",), policy_version=AUTHORED_POLICY, elapsed_us=0
        ),
        source_sha256=sha256_hex(text),
    )


def compose(template: str, parts: Mapping[str, RedactedText]) -> RedactedText:
    """Interpolate redacted parts into an authored template.

    The result inherits the *weakest* status among its parts: laundering unredacted
    content through an f-string is exactly the failure this prevents.
    """
    blocked = [name for name, part in parts.items() if not part.egress_permitted]
    rendered = template.format(**{name: part.text for name, part in parts.items()})

    if blocked:
        return RedactedText(
            text="",
            report=RedactionReport(
                status=RedactionStatus.DIRTY,
                engines_run=("manual",),
                policy_version=AUTHORED_POLICY,
                elapsed_us=0,
                leak_check_passed=False,
                residual_patterns=tuple(sorted(f"unredacted_part:{name}" for name in blocked)),
            ),
            source_sha256=sha256_hex(template),
        )

    spans = tuple(span for part in parts.values() for span in part.report.spans)
    engines: tuple[str, ...] = ("manual",)
    for part in parts.values():
        for engine in part.report.engines_run:
            if engine not in engines:
                engines = (*engines, engine)

    return RedactedText(
        text=rendered,
        report=RedactionReport.clean(
            spans=spans,
            engines_run=engines,  # type: ignore[arg-type]
            policy_version=AUTHORED_POLICY,
            elapsed_us=sum(part.report.elapsed_us for part in parts.values()),
        ),
        source_sha256=sha256_hex(rendered),
    )
