"""``SessionState`` -- the LangGraph state object for Modules 4 and 5.

This is the only mutable model in the contract layer. Append-only collections carry
an ``add`` reducer so concurrent nodes merge instead of clobbering one another.
"""

from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ccas.schemas.common import (
    Channel,
    DropsComputedFields,
    Frozen,
    JsonValue,
    Slug,
    Speaker,
    TraceContext,
    VerificationLevel,
    utcnow,
)
from ccas.schemas.escalation import EscalationDecision
from ccas.schemas.pii import RedactedText
from ccas.schemas.tools import ToolRecord

__all__ = [
    "CallerContext",
    "IntentPrediction",
    "LatencyLedger",
    "SentimentSnapshot",
    "SessionOutcome",
    "SessionState",
    "SlotValue",
    "Turn",
]

IntentSource = Literal[
    "llm_router",
    "classifier",
    "dtmf",
    "gold",
    "fallback",
    # The router never answered -- provider down, quota spent, timeout. Distinct from
    # "fallback" because nobody failed to understand; nobody was asked.
    "unavailable",
]
CaptureMethod = Literal["speech", "dtmf", "crm_prefill", "inferred"]
OutcomeStatus = Literal["contained", "escalated", "abandoned", "error"]


class IntentPrediction(Frozen):
    intent_id: Slug | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: IntentSource
    alternatives: tuple[tuple[str, float], ...] = ()
    latency_ms: int = Field(ge=0)
    at: datetime = Field(default_factory=utcnow)

    @property
    def resolved(self) -> bool:
        return self.intent_id is not None


class SlotValue(Frozen):
    name: Slug
    raw: RedactedText
    parsed: JsonValue = None
    valid: bool = False
    attempts: int = Field(default=1, ge=1, le=5)
    captured_via: CaptureMethod


class SentimentSnapshot(Frozen):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    frustration_index: float = Field(ge=0.0, le=1.0)
    turn_index: int = Field(ge=0)
    at: datetime = Field(default_factory=utcnow)


class Turn(Frozen):
    index: int = Field(ge=0)
    speaker: Speaker
    content: RedactedText
    intent: IntentPrediction | None = None
    tool_call_ids: tuple[str, ...] = ()
    barged_in: bool = False
    rtt_ms: int | None = Field(default=None, ge=0)
    at: datetime = Field(default_factory=utcnow)


class CallerContext(Frozen):
    caller_ref: str = Field(min_length=8, max_length=128)
    """Hashed ANI or user id. The raw identifier never enters the state (Rule 2)."""

    verification: VerificationLevel = VerificationLevel.NONE
    verified_attributes: dict[Slug, str] = Field(default_factory=dict)
    locale: str = Field(default="en-US", max_length=32)
    prior_interaction_count: int = Field(default=0, ge=0)
    omnichannel_context: dict[str, JsonValue] = Field(default_factory=dict)
    """Active web/app session handed across channels."""


class LatencyLedger(DropsComputedFields):
    """Per-turn stage timings. Mutable -- Module 5 writes into it as the turn runs.

    The budget lives here rather than in a config lookup so that any code holding a
    session can answer "did we breach?" without I/O (CLAUDE.md Rule 3).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    budget_ms: int = Field(default=800, ge=1)
    vad_ms: int = Field(default=0, ge=0)
    stt_ms: int = Field(default=0, ge=0)
    redact_us: int = Field(default=0, ge=0)
    router_ms: int = Field(default=0, ge=0)
    tool_ms: int = Field(default=0, ge=0)
    llm_ttft_ms: int = Field(default=0, ge=0)
    tts_ttfb_ms: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_rtt_ms(self) -> int:
        return (
            self.vad_ms
            + self.stt_ms
            + round(self.redact_us / 1000)
            + self.router_ms
            + self.tool_ms
            + self.llm_ttft_ms
            + self.tts_ttfb_ms
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def breached(self) -> bool:
        return self.total_rtt_ms > self.budget_ms

    @property
    def headroom_ms(self) -> int:
        return self.budget_ms - self.total_rtt_ms

    def stages(self) -> dict[str, int]:
        return {
            "vad_ms": self.vad_ms,
            "stt_ms": self.stt_ms,
            "redact_ms": round(self.redact_us / 1000),
            "router_ms": self.router_ms,
            "tool_ms": self.tool_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
        }


class SessionOutcome(Frozen):
    status: OutcomeStatus
    resolved_intent: Slug | None = None
    turn_count: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    disposition_code: Slug | None = None
    notes: str | None = Field(default=None, max_length=1024)


class SessionState(BaseModel):
    """LangGraph ``StateGraph`` state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    session_id: str = Field(min_length=1, max_length=128)
    trace: TraceContext
    domain: Slug
    channel: Channel = Channel.VOICE
    caller: CallerContext

    # Append-only. The reducer is what makes parallel node writes safe.
    turns: Annotated[list[Turn], add] = Field(default_factory=list)
    intent_history: Annotated[list[IntentPrediction], add] = Field(default_factory=list)
    sentiment_trail: Annotated[list[SentimentSnapshot], add] = Field(default_factory=list)
    tool_records: Annotated[list[ToolRecord], add] = Field(default_factory=list)

    current_intent: IntentPrediction | None = None
    active_agent: Slug | None = None
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    pending_slot: Slug | None = None

    turn_index: int = Field(default=0, ge=0)
    clarification_count: int = Field(default=0, ge=0)
    no_input_count: int = Field(default=0, ge=0)
    no_match_count: int = Field(default=0, ge=0)
    barge_in_count: int = Field(default=0, ge=0)
    tool_failure_count: int = Field(default=0, ge=0)

    escalation: EscalationDecision | None = None
    outcome: SessionOutcome | None = None
    terminal: bool = False
    latency: LatencyLedger = Field(default_factory=LatencyLedger)
    started_at: datetime = Field(default_factory=utcnow)

    @property
    def sentiment(self) -> SentimentSnapshot | None:
        return self.sentiment_trail[-1] if self.sentiment_trail else None

    @property
    def escalated(self) -> bool:
        return self.escalation is not None

    def missing_required_slots(self, required: tuple[Slug, ...]) -> tuple[str, ...]:
        """Required slot names not yet captured *and validated*.

        The taxonomy is injected rather than imported so this model stays free of any
        dependency on a loaded pack.
        """
        return tuple(
            name for name in required if name not in self.slots or not self.slots[name].valid
        )

    def failed_tool_calls(self) -> tuple[ToolRecord, ...]:
        return tuple(r for r in self.tool_records if not r.result.ok)
