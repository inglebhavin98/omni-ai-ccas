"""Session checkpointing.

A voice call is many graph invocations sharing one state. The checkpointer is what makes
"resume the session" a lookup rather than a reconstruction -- and what lets a dropped
call be picked up on a callback without asking the caller to start again.

In-memory is the development default. A durable backend is a deployment concern; the
interface LangGraph exposes is the same either way.
"""

from __future__ import annotations

from typing import Any

from ccas.schemas.common import (
    Channel,
    RiskTier,
    Speaker,
    TraceContext,
    Urgency,
    VerificationLevel,
)
from ccas.schemas.escalation import EscalationDecision, HandoffReason
from ccas.schemas.pii import (
    PiiEntityType,
    RedactedText,
    RedactionReport,
    RedactionSpan,
    RedactionStatus,
)
from ccas.schemas.session import (
    CallerContext,
    IntentPrediction,
    LatencyLedger,
    SentimentSnapshot,
    SessionOutcome,
    SlotValue,
    Turn,
)
from ccas.schemas.tools import ToolPayload, ToolRecord, ToolResult, ToolStatus

__all__ = ["CHECKPOINT_TYPES", "memory_checkpointer", "thread_config"]

#: Everything ``SessionState`` can contain. LangGraph refuses to deserialize a type it
#: was not told about -- currently a warning, shortly an error -- so the list is
#: declared rather than discovered. A new field on a state model belongs here too.
CHECKPOINT_TYPES: tuple[type, ...] = (
    TraceContext,
    Channel,
    Speaker,
    RiskTier,
    Urgency,
    VerificationLevel,
    HandoffReason,
    EscalationDecision,
    PiiEntityType,
    RedactionStatus,
    RedactionSpan,
    RedactionReport,
    RedactedText,
    CallerContext,
    Turn,
    IntentPrediction,
    SlotValue,
    SentimentSnapshot,
    SessionOutcome,
    LatencyLedger,
    ToolStatus,
    ToolPayload,
    ToolResult,
    ToolRecord,
)


def _serializer() -> Any:
    """A serializer that accepts exactly our state types and nothing else.

    The default allows everything and warns on each unregistered type, which LangGraph
    intends to turn into an error. Declaring the list makes the checkpoint boundary
    explicit -- and means an unexpected type in the state is caught here rather than
    silently round-tripped.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def memory_checkpointer() -> Any:
    """Process-lifetime checkpointer. Lost on restart -- fine for dev, not for calls."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=_serializer())


def thread_config(session_id: str, trace: TraceContext | None = None) -> dict[str, Any]:
    """LangGraph addresses a conversation by thread id; ours is the session id."""
    configurable: dict[str, Any] = {"thread_id": session_id}
    if trace is not None:
        configurable["correlation_id"] = trace.correlation_id
    return {"configurable": configurable}
