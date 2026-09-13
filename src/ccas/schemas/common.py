"""Primitives shared by every contract in this package.

This module has no intra-project imports and must keep it that way: every other
package depends on ``ccas.schemas``, so a dependency here would create a cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

__all__ = [
    "SCHEMA_VERSION",
    "Channel",
    "DropsComputedFields",
    "Frozen",
    "JsonValue",
    "RiskTier",
    "SchemaVersion",
    "Slug",
    "Speaker",
    "TraceContext",
    "Urgency",
    "VerificationLevel",
    "utcnow",
]

SCHEMA_VERSION = "1.0"

#: Every persisted contract carries one of these. The alias is shared, so a reader can
#: accept a record written by an older build; each model pins its own default, and a model
#: that has never changed shape still says "1.0". Widening this list is the *only* way a
#: contract's shape may change (CLAUDE.md Rule 8) -- the migration goes in
#: `docs/tech-spec.md` alongside the shape it describes.
SchemaVersion: TypeAlias = Literal["1.0", "1.1"]

#: Lowercase dotted/underscored identifier. Intent ids, slot names, tool names,
#: queue names and domain names are all slugs -- never enums -- so that a domain
#: pack can introduce new values without a code change (CLAUDE.md Rule 1).
Slug: TypeAlias = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
        min_length=1,
        max_length=128,
    ),
]

_HEX32: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
_HEX16: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]


def utcnow() -> datetime:
    """Timezone-aware now. Naive datetimes are banned (ruff DTZ)."""
    return datetime.now(UTC)


class DropsComputedFields(BaseModel):
    """Makes ``extra="forbid"`` compatible with ``@computed_field``.

    Computed fields are serialized out but rejected on the way back in, which breaks
    round-tripping through parquet, a CTI payload or a taxonomy file. Stripping only
    *our own* computed names keeps the strictness that catches an adapter typo.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_computed(cls, data: object) -> object:
        if isinstance(data, dict):
            computed = set(cls.model_computed_fields)
            if computed & data.keys():
                return {k: v for k, v in data.items() if k not in computed}
        return data


class Frozen(DropsComputedFields):
    """Immutable, closed base model.

    ``extra="forbid"`` matters more than it looks: an adapter that invents a field
    should fail loudly rather than silently smuggle unvalidated data (often raw PII)
    through the pipeline.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
    )


class Channel(StrEnum):
    VOICE = "voice"
    CHAT = "chat"
    SMS = "sms"
    EMAIL = "email"
    WEB = "web"


class Speaker(StrEnum):
    CALLER = "caller"
    BOT = "bot"
    HUMAN_AGENT = "human_agent"
    SYSTEM = "system"
    IVR = "ivr"


class RiskTier(StrEnum):
    """Generic risk ladder. What lands in each tier is pack-defined, not code-defined."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    REGULATED = "regulated"


class VerificationLevel(StrEnum):
    NONE = "none"
    SOFT = "soft"
    STRONG = "strong"
    STEP_UP = "step_up"

    @property
    def rank(self) -> int:
        return _VERIFICATION_RANK[self]

    def satisfies(self, required: VerificationLevel) -> bool:
        return self.rank >= required.rank


_VERIFICATION_RANK: dict[VerificationLevel, int] = {
    VerificationLevel.NONE: 0,
    VerificationLevel.SOFT: 1,
    VerificationLevel.STRONG: 2,
    VerificationLevel.STEP_UP: 3,
}


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TraceContext(Frozen):
    """W3C-shaped trace identity carried on every payload that crosses a boundary."""

    trace_id: _HEX32
    span_id: _HEX16
    correlation_id: str = Field(min_length=1, max_length=128)
    tenant_id: Slug = "default"
