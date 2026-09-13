"""Shared node machinery.

Nodes return partial state updates, which LangGraph merges through the reducers declared
on ``SessionState``. Append-only fields are returned as one-element lists; scalar fields
are returned as values.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ccas.graph.context import GraphContext
from ccas.schemas.common import Speaker
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import SessionState, Turn

__all__ = ["NodeFn", "caller_history", "latest_caller_text", "speak"]

NodeFn = Callable[[SessionState], Awaitable[dict[str, Any]]]


def speak(ctx: GraphContext, state: SessionState, text: str) -> dict[str, Any]:
    """Emit a bot turn.

    Bot text is redacted too. It is composed from already-clean inputs, so this should
    never find anything -- but it is the last gate before TTS, it costs ~100us, and a
    model echoing a value it was told not to read aloud is exactly the failure that
    would otherwise reach a caller's ear (CLAUDE.md Rule 2).
    """
    content = ctx.redaction.redact(text, ctx.redaction.new_allocator(ctx.vault))
    turn = Turn(index=state.turn_index, speaker=Speaker.BOT, content=content)
    return {"turns": [turn], "turn_index": state.turn_index + 1}


def latest_caller_text(state: SessionState) -> RedactedText | None:
    for turn in reversed(state.turns):
        if turn.speaker is Speaker.CALLER:
            return turn.content
    return None


def caller_history(state: SessionState, limit: int = 6) -> tuple[RedactedText, ...]:
    """Recent caller turns, oldest first. Excludes the one being classified."""
    caller_turns = [t.content for t in state.turns if t.speaker is Speaker.CALLER]
    return tuple(caller_turns[:-1][-limit:])
