"""``HandoffContext`` -- the CTI payload pushed to an agent desktop on escalation.

This is the platform's highest-risk egress surface: it leaves the process, lands in a
third-party desktop, and is retained. Every text-bearing field must therefore carry an
egress-permitted ``RedactionReport``, enforced below.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from ccas.schemas.common import (
    Frozen,
    SchemaVersion,
    Slug,
    TraceContext,
    Urgency,
    VerificationLevel,
    utcnow,
)
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import SentimentSnapshot, SlotValue, Turn
from ccas.schemas.tools import ToolRecord

__all__ = ["HandoffContext", "NextBestAction", "VerifiedIdentity"]


class VerifiedIdentity(Frozen):
    caller_ref: str = Field(min_length=8, max_length=128)
    """Hashed. A human agent resolves it against the CRM; it is not an identifier."""

    level: VerificationLevel
    method: Slug
    attributes: dict[Slug, str] = Field(default_factory=dict)
    verified_at: datetime

    @model_validator(mode="after")
    def _check_level(self) -> Self:
        if self.level is VerificationLevel.NONE:
            raise ValueError("VerifiedIdentity requires a level above NONE")
        return self


class NextBestAction(Frozen):
    action: str = Field(min_length=1, max_length=256)
    rationale: str = Field(min_length=1, max_length=512)
    confidence: float = Field(ge=0.0, le=1.0)


class HandoffContext(Frozen):
    schema_version: SchemaVersion = "1.0"
    handoff_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    trace: TraceContext
    domain: Slug

    reason: HandoffReason
    urgency: Urgency
    target_queue: Slug
    required_skills: tuple[Slug, ...] = ()

    identity: VerifiedIdentity | None = None
    intent_path: tuple[Slug, ...] = ()
    """L1 -> L2 -> L3, so the desktop can show the caller's journey, not just a leaf."""

    intent_confidence: float = Field(ge=0.0, le=1.0)
    collected_slots: dict[str, SlotValue] = Field(default_factory=dict)

    summary: RedactedText
    next_best_actions: tuple[NextBestAction, ...] = ()
    sentiment_trail: tuple[SentimentSnapshot, ...] = ()
    transcript: tuple[Turn, ...] = ()
    tool_trace: tuple[ToolRecord, ...] = ()

    cti_attributes: dict[str, str] = Field(default_factory=dict)
    """UUI / attached data for Genesys or Cisco. Values must already be redacted."""

    audio_recording_ref: str | None = Field(default=None, max_length=512)
    """Object-store URI. Audio never travels inline (Rule 2)."""

    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _enforce_clean(self) -> Self:
        if not self.summary.egress_permitted:
            raise ValueError(
                f"handoff {self.handoff_id}: summary redaction status is "
                f"{self.summary.report.status.value}"
            )
        for turn in self.transcript:
            if not turn.content.egress_permitted:
                raise ValueError(
                    f"handoff {self.handoff_id}: transcript turn {turn.index} is unredacted"
                )
        for name, slot in self.collected_slots.items():
            if not slot.raw.egress_permitted:
                raise ValueError(f"handoff {self.handoff_id}: slot {name!r} is unredacted")
        for record in self.tool_trace:
            if not record.result.safe_for_model:
                raise ValueError(
                    f"handoff {self.handoff_id}: tool result "
                    f"{record.payload.tool_name!r} is unredacted"
                )
        return self

    @model_validator(mode="after")
    def _check_intent_path(self) -> Self:
        for depth, intent_id in enumerate(self.intent_path, start=1):
            if intent_id.count(".") + 1 != depth:
                raise ValueError(
                    f"intent_path must descend L1->L2->L3, got {list(self.intent_path)}"
                )
        return self

    @property
    def leaf_intent(self) -> Slug | None:
        return self.intent_path[-1] if self.intent_path else None
