"""Redaction policy loading.

The policy is data so that an operator can tighten it without a deploy, and so that
``tests/`` asserts against the same file production reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, Slug
from ccas.schemas.domain import PatternSpec
from ccas.schemas.pii import PiiEntityType

__all__ = [
    "EnginePolicy",
    "PatternSet",
    "RedactionPolicy",
    "load_pattern_set",
    "load_policy",
]


class EnginePolicy(Frozen):
    name: str
    enabled: bool = True
    fail_closed: bool = True
    """An engine that is enabled but unavailable makes the pipeline UNVERIFIED."""

    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: tuple[PiiEntityType, ...] = ()
    """Restrict an engine to the entity classes it is actually good at. Empty = all."""

    spacy_model: str = "en_core_web_lg"
    """Named explicitly so the running model is a recorded decision. Detection quality
    differs materially between ``sm`` and ``lg``; a silent fallback would make two
    deployments disagree about what counts as a name."""


class LeakDetectionPolicy(Frozen):
    enabled: bool = True
    patterns: tuple[Slug, ...] = ()
    """Pattern names re-run after substitution. High precision only."""


class RedactionPolicy(Frozen):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    engines: tuple[EnginePolicy, ...] = Field(min_length=1)
    placeholder_format: str = "[{entity}_{n}]"
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    realtime_budget_us: int = Field(default=3000, ge=100)
    leak_detection: LeakDetectionPolicy = LeakDetectionPolicy()

    @model_validator(mode="after")
    def _check_placeholder_format(self) -> Self:
        if "{entity}" not in self.placeholder_format or "{n}" not in self.placeholder_format:
            raise ValueError("placeholder_format must contain {entity} and {n}")
        return self

    def engine(self, name: str) -> EnginePolicy | None:
        return next((e for e in self.engines if e.name == name), None)

    def enabled_engines(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.engines if e.enabled)


class PatternSet(Frozen):
    """Compiled-at-load collection of regex rules, plus the leak-detection subset."""

    patterns: tuple[PatternSpec, ...] = Field(min_length=1)
    leak_detection: tuple[Slug, ...] = ()

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        names = [p.name for p in self.patterns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate pattern name in set")
        missing = set(self.leak_detection) - set(names)
        if missing:
            raise ValueError(f"leak_detection references unknown patterns: {sorted(missing)}")
        return self

    def merged_with(self, extra: tuple[PatternSpec, ...]) -> PatternSet:
        """Layer a pack's patterns on top. Pack rules add; they never remove."""
        known = {p.name for p in self.patterns}
        clashing = known & {p.name for p in extra}
        if clashing:
            raise ValueError(f"pack patterns may not shadow global ones: {sorted(clashing)}")
        return PatternSet(patterns=self.patterns + extra, leak_detection=self.leak_detection)

    @property
    def by_name(self) -> dict[str, PatternSpec]:
        return {p.name: p for p in self.patterns}


def load_policy(path: Path) -> RedactionPolicy:
    return RedactionPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_pattern_set(path: Path) -> PatternSet:
    return PatternSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
