"""Request and response shapes for the workbench API.

Separate from ``ccas.schemas`` on purpose: these are a transport concern and may change
with the UI, while the contract layer is versioned and frozen.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ccas.schemas.common import JsonValue, Slug, VerificationLevel

__all__ = [
    "CreateSessionRequest",
    "InvokeToolRequest",
    "RedactionPreview",
    "SessionSnapshot",
    "SpeakRequest",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(Strict):
    domain: Slug | None = None
    verification: VerificationLevel = VerificationLevel.NONE


class SpeakRequest(Strict):
    text: str = Field(min_length=1, max_length=4000)


class InvokeToolRequest(Strict):
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    verification: VerificationLevel = VerificationLevel.STRONG
    intent_id: Slug | None = None


class RedactionPreview(Strict):
    """What the model would see, and what was removed to get there."""

    input_length: int
    redacted: str
    status: str
    egress_permitted: bool
    elapsed_us: int
    entity_counts: dict[str, int]
    residual_patterns: tuple[str, ...]
    mode: str


class SessionSnapshot(Strict):
    """Everything the workbench shows for one turn. Redacted throughout."""

    session_id: str
    domain: str
    turn_index: int
    terminal: bool
    transcript: list[dict[str, Any]]
    current_intent: dict[str, Any] | None
    active_agent: str | None
    slots: dict[str, Any]
    pending_slot: str | None
    counters: dict[str, int]
    policy_verdicts: list[dict[str, Any]]
    tool_records: list[dict[str, Any]]
    escalation: dict[str, Any] | None
    outcome: dict[str, Any] | None
    latency: dict[str, Any]
