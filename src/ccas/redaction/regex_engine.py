"""Regex detection -- the hot path.

Regex runs first and carries the realtime budget (3 ms of 800). Presidio is consulted
only for the entity classes regex genuinely cannot settle, because a model call on every
utterance would not fit.
"""

from __future__ import annotations

import re
from pathlib import Path

from ccas.redaction.engine import DetectedSpan, RedactionEngine
from ccas.redaction.policy import PatternSet, load_pattern_set
from ccas.redaction.validators import resolve_validator
from ccas.schemas.domain import PatternSpec
from ccas.schemas.pii import PiiEntityType

__all__ = ["GENERIC_PATTERNS_PATH", "RegexEngine"]

GENERIC_PATTERNS_PATH = Path(__file__).parent / "patterns" / "generic.yaml"


class _CompiledPattern:
    __slots__ = ("regex", "spec", "validate")

    def __init__(self, spec: PatternSpec) -> None:
        self.spec = spec
        try:
            self.regex = re.compile(spec.pattern)
        except re.error as exc:
            raise ValueError(f"pattern {spec.name!r} does not compile: {exc}") from exc
        if spec.capture_group > self.regex.groups:
            raise ValueError(
                f"pattern {spec.name!r} asks for group {spec.capture_group} "
                f"but defines {self.regex.groups}"
            )
        self.validate = resolve_validator(spec.validator_name)


class RegexEngine(RedactionEngine):
    name = "regex"

    def __init__(self, pattern_set: PatternSet, min_score: float = 0.0) -> None:
        self._set = pattern_set
        self._min_score = min_score
        self._compiled = tuple(
            _CompiledPattern(spec) for spec in pattern_set.patterns if spec.score >= min_score
        )
        self._leak = tuple(c for c in self._compiled if c.spec.name in pattern_set.leak_detection)

    @classmethod
    def from_path(cls, path: Path = GENERIC_PATTERNS_PATH, min_score: float = 0.0) -> RegexEngine:
        return cls(load_pattern_set(path), min_score)

    @property
    def available(self) -> bool:
        # Pure regex: if it imported, it works.
        return bool(self._compiled)

    @property
    def handled_entities(self) -> frozenset[PiiEntityType]:
        return frozenset(c.spec.entity_type for c in self._compiled)

    @property
    def pattern_names(self) -> tuple[str, ...]:
        return tuple(c.spec.name for c in self._compiled)

    def detect(self, text: str) -> tuple[DetectedSpan, ...]:
        return tuple(self._scan(text, self._compiled))

    def detect_leaks(self, text: str) -> tuple[str, ...]:
        """Pattern *names* that still match after substitution. Never the matched text."""
        return tuple(sorted({span.pattern_name or "" for span in self._scan(text, self._leak)}))

    @staticmethod
    def _scan(text: str, compiled: tuple[_CompiledPattern, ...]) -> list[DetectedSpan]:
        found: list[DetectedSpan] = []
        for pattern in compiled:
            group = pattern.spec.capture_group
            for match in pattern.regex.finditer(text):
                matched = match.group(group)
                if matched is None or not pattern.validate(matched):
                    continue
                found.append(
                    DetectedSpan(
                        start=match.start(group),
                        end=match.end(group),
                        entity_type=pattern.spec.entity_type,
                        score=pattern.spec.score,
                        engine="regex",
                        custom_label=pattern.spec.custom_label,
                        pattern_name=pattern.spec.name,
                    )
                )
        return found
