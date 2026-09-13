"""``DomainPack`` -- the manifest that makes the core generic.

Everything a vertical needs (taxonomy, tools, queues, prompts, thresholds, redaction
patterns, compliance posture) lives here as data. If a feature needs a domain ``if``,
it needs a pack field instead (CLAUDE.md Rule 1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, RiskTier, SchemaVersion, Slug, VerificationLevel
from ccas.schemas.pii import PiiEntityType
from ccas.schemas.taxonomy import QuadrantThresholds
from ccas.schemas.tools import ToolSpec

__all__ = [
    "ComplianceProfile",
    "ComplianceRegime",
    "ConfidenceThresholds",
    "DomainPack",
    "PatternSpec",
    "QueueSpec",
]


class ComplianceRegime(StrEnum):
    NONE = "none"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    GLBA = "glba"


class ComplianceProfile(Frozen):
    regimes: tuple[ComplianceRegime, ...] = (ComplianceRegime.NONE,)
    transcript_retention_days: int = Field(default=90, ge=1, le=3650)
    audio_retention_days: int = Field(default=30, ge=1, le=3650)
    recording_consent_required: bool = False
    consent_prompt: str | None = None

    @model_validator(mode="after")
    def _check_consent_prompt(self) -> Self:
        if self.recording_consent_required and not self.consent_prompt:
            raise ValueError("recording_consent_required needs a consent_prompt")
        return self


class PatternSpec(Frozen):
    """A pack-supplied regex redaction rule, layered over the generic pattern set."""

    name: Slug
    pattern: str = Field(min_length=1)
    entity_type: PiiEntityType
    custom_label: Slug | None = None
    score: float = Field(default=0.95, ge=0.0, le=1.0)
    validator_name: Slug | None = None
    """Optional post-match check registered in ``ccas.redaction``, e.g. ``luhn``."""

    capture_group: int = Field(default=0, ge=0, le=9)
    """Which group to redact. Lets a rule use surrounding words as context without
    redacting them -- "my name is Dana" should become "my name is [PERSON_1]", not
    "[PERSON_1]", or the transcript stops making sense to the agent who inherits it."""


class QueueSpec(Frozen):
    """A human agent destination. Maps onto a CTI queue in the target CCaaS."""

    name: Slug
    display_name: str = Field(min_length=1)
    skills: tuple[Slug, ...] = ()
    min_verification: VerificationLevel = VerificationLevel.NONE
    max_risk_tier: RiskTier = RiskTier.REGULATED
    cti_queue_id: str | None = None


class ConfidenceThresholds(Frozen):
    route: float = Field(default=0.82, ge=0.0, le=1.0)
    """Below this the router clarifies instead of dispatching."""

    clarify_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    """Below this, clarifying is pointless -- escalate directly."""

    slot_accept: float = Field(default=0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if self.clarify_floor > self.route:
            raise ValueError("clarify_floor must not exceed route threshold")
        return self


class DomainPack(Frozen):
    schema_version: SchemaVersion = "1.0"
    domain: Slug
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    display_name: str = Field(min_length=1)
    locales: tuple[str, ...] = ("en-US",)

    taxonomy_ref: str = Field(min_length=1)
    """Path relative to the pack directory, e.g. ``taxonomy.json``."""

    tools: tuple[ToolSpec, ...] = ()
    queues: tuple[QueueSpec, ...] = Field(min_length=1)
    redaction_patterns: tuple[PatternSpec, ...] = ()

    slot_pii: dict[Slug, PiiEntityType] = Field(default_factory=dict)
    """Slot names whose captured value is an identifier, and what kind.

    A slot with an entity is redacted *wholesale* on capture rather than scanned, which is
    what catches a bare reference number matching no pattern (Rule 2). The mapping is here
    rather than in the core because which slot names hold identifiers is vertical
    knowledge -- the core may not know what this vertical calls things (Rule 1)."""
    prompts: dict[str, str] = Field(default_factory=dict)
    greeting: str = Field(min_length=1)

    confidence: ConfidenceThresholds = ConfidenceThresholds()
    quadrant_thresholds: QuadrantThresholds = QuadrantThresholds()
    max_turns: int = Field(default=25, ge=1, le=200)
    compliance: ComplianceProfile = ComplianceProfile()

    @model_validator(mode="after")
    def _check_tools_belong_to_pack(self) -> Self:
        for tool in self.tools:
            if tool.domain != self.domain:
                raise ValueError(
                    f"tool {tool.name!r} declares domain {tool.domain!r}, "
                    f"but the pack is {self.domain!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        for label, names in (
            ("tool", [t.name for t in self.tools]),
            ("queue", [q.name for q in self.queues]),
            ("pattern", [p.name for p in self.redaction_patterns]),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} name in pack {self.domain!r}")
        return self

    @property
    def tools_by_name(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    @property
    def queues_by_name(self) -> dict[str, QueueSpec]:
        return {q.name: q for q in self.queues}

    @property
    def default_queue(self) -> QueueSpec:
        return self.queues[0]
