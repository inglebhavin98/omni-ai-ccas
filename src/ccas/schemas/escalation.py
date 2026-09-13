"""Escalation vocabulary.

Split out from ``handoff`` so that ``session`` can record a decision without importing
the full CTI payload (which itself imports session types).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ccas.schemas.common import Frozen, Slug, Urgency

__all__ = ["EscalationDecision", "HandoffReason"]


class HandoffReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    CALLER_REQUEST = "caller_request"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    RISK_TIER = "risk_tier"
    TOOL_FAILURE = "tool_failure"
    MAX_TURNS = "max_turns"
    VERIFICATION_FAILED = "verification_failed"
    POLICY = "policy"
    UNSUPPORTED_INTENT = "unsupported_intent"
    SYSTEM_ERROR = "system_error"


class EscalationDecision(Frozen):
    reason: HandoffReason
    triggered_by: Slug
    """Name of the policy that fired, e.g. ``confidence`` or ``sentiment``."""

    at_turn: int = Field(ge=0)
    urgency: Urgency = Urgency.NORMAL
    detail: str | None = Field(default=None, max_length=512)
