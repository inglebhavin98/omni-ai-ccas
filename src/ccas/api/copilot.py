"""Live copilot stream for the agent desktop (tech-spec 3.2a).

``WS /v1/ws/copilot/{session_id}`` pushes what a human agent would watch while the bot
works: the redacted transcript as it grows, the current intent, and — when the session
ends in escalation — the same ``HandoffContext`` the CTI path delivers. Every payload is
a re-read of objects the graph already produced, so there is nothing here a model could
not also see.

This is a poll of in-process session state, not an event bus, and the interval is honest
about that: it is a development surface bound to loopback, not a production push fabric.
An unknown session is refused at accept time with a code the client can act on; a session
that ends while nobody is watching simply has nothing more to say.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect

from ccas.api.handoffs import handoff_id_for
from ccas.api.sessions import SessionManager, SessionNotFoundError
from ccas.api.views import snapshot_of
from ccas.observability.logging import get_logger

__all__ = ["POLL_INTERVAL_S", "run_copilot_stream"]

LOG = get_logger("api.copilot")

POLL_INTERVAL_S = 0.5

#: Close codes this surface uses beyond the defaults. 4000 keeps "wrong room" distinct
#: from the protocol-level 1008, which a browser turns into a generic failure.
SESSION_NOT_FOUND = 4000


async def run_copilot_stream(sessions: SessionManager, session_id: str, ws: WebSocket) -> None:
    """Push ``SessionSnapshot`` JSON until the session turns terminal or the client leaves."""
    await ws.accept()
    try:
        handle = sessions.get(session_id)
    except SessionNotFoundError:
        await ws.close(code=SESSION_NOT_FOUND, reason=f"no session {session_id!r}")
        return

    last_index = -1
    try:
        while True:
            state = handle.snapshot()
            if state.turn_index != last_index:
                last_index = state.turn_index
                snapshot = snapshot_of(handle)
                await ws.send_text(snapshot.model_dump_json())
                LOG.info(
                    "copilot.snapshot",
                    correlation_id=session_id,
                    turn_index=state.turn_index,
                    terminal=state.terminal,
                )
            if state.terminal:
                if state.escalated:
                    handoff = sessions.handoffs.get(handoff_id_for(session_id))
                    if handoff is not None:
                        await ws.send_text(json.dumps({"handoff": handoff.model_dump(mode="json")}))
                        LOG.info("copilot.handoff", correlation_id=session_id)
                await ws.close(code=1000, reason="session terminal")
                return
            await asyncio.sleep(POLL_INTERVAL_S)
    except WebSocketDisconnect:
        return
