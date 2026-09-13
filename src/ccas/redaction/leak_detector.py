"""Post-substitution verification.

Detection can be wrong; substitution can be wrong; a pack pattern can overlap a global
one in a way that leaves a fragment behind. The only way to know the output is clean is
to look at the output.

A hit here forces ``DIRTY`` and blocks egress. That is the correct outcome even though
it fails a call -- a blocked turn is recoverable, a leak is not.
"""

from __future__ import annotations

import re

from ccas.redaction.regex_engine import RegexEngine

__all__ = ["LeakDetector", "LeakReport"]

#: Matches a well-formed placeholder, e.g. ``[PERSON_1]``.
_PLACEHOLDER = re.compile(r"\[[A-Z0-9_]+_\d+\]")

#: A digit run this long is not a reference number anyone reads aloud, whatever the
#: entity patterns think. Belt-and-braces against a pattern that silently stops matching.
_LONG_DIGIT_RUN = re.compile(r"(?<!\[)\b(?:\d[ -]?){15,}\d\b")


class LeakReport:
    __slots__ = ("placeholders", "residual_patterns")

    def __init__(self, residual_patterns: tuple[str, ...], placeholders: int) -> None:
        self.residual_patterns = residual_patterns
        self.placeholders = placeholders

    @property
    def clean(self) -> bool:
        return not self.residual_patterns

    def __repr__(self) -> str:
        return (
            f"LeakReport(clean={self.clean}, "
            f"residual={list(self.residual_patterns)}, placeholders={self.placeholders})"
        )


class LeakDetector:
    """Re-scans substituted text with a high-precision subset of the pattern rules."""

    __slots__ = ("_enabled", "_engine")

    def __init__(self, engine: RegexEngine, enabled: bool = True) -> None:
        self._engine = engine
        self._enabled = enabled

    def check(self, redacted_text: str) -> LeakReport:
        placeholders = len(_PLACEHOLDER.findall(redacted_text))
        if not self._enabled:
            return LeakReport((), placeholders)

        residual = list(self._engine.detect_leaks(redacted_text))
        if _LONG_DIGIT_RUN.search(redacted_text):
            residual.append("long_digit_run")
        return LeakReport(tuple(sorted(set(residual))), placeholders)
