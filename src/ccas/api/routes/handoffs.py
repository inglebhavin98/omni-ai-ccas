"""Agent-desktop routes (tech-spec 3.2a) and the copilot websocket.

Kept out of ``main.py`` so the workbench surface and the desktop surface are readable
separately. Everything served here is a ``schemas/`` model or a view over one -- the
handoff contract itself enforces egress cleanliness at construction, so these routes
cannot leak by re-reading what the session manager stored.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket

from ccas.api.copilot import run_copilot_stream
from ccas.api.handoffs import handoff_id_for
from ccas.api.sessions import SessionManager
from ccas.api.views import handoff_of

__all__ = ["build_desktop_router", "mount_copilot_ws"]


def build_desktop_router(sessions: SessionManager) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["agent-desktop"])

    @router.get("/handoffs/{handoff_id}")
    async def get_handoff(handoff_id: str) -> dict[str, object]:
        """Fetch a CTI payload for the agent desktop.

        404 covers both "never existed" and "rotated out of the bounded store" -- a
        workbench keeps the most recent handoffs of this process only. Distinct from a
        202-style "not yet": a session that has not escalated has no handoff, and
        pretending otherwise would make the desktop poll forever.
        """
        handoff = sessions.handoffs.get(handoff_id)
        if handoff is None:
            raise HTTPException(
                status_code=404, detail=f"no handoff {handoff_id!r} in this process"
            )
        return handoff_of(handoff)

    @router.get("/sessions/{session_id}/handoff")
    async def get_session_handoff(session_id: str) -> dict[str, object]:
        """The handoff for a session, by the deterministic id the escalate node assigns."""
        handoff = sessions.handoffs.get(handoff_id_for(session_id))
        if handoff is None:
            raise HTTPException(
                status_code=404,
                detail=f"session {session_id!r} has no handoff (it did not escalate, "
                "or it was rotated out)",
            )
        return handoff_of(handoff)

    return router


def mount_copilot_ws(app: FastAPI, sessions: SessionManager) -> None:
    """Register ``WS /v1/ws/copilot/{session_id}``.

    A function rather than a router member because the socket is one route with its own
    accept/close protocol, and keeping that apart from the HTTP router reads better than
    a router that serves one socket.
    """

    @app.websocket("/v1/ws/copilot/{session_id}")
    async def copilot_stream(ws: WebSocket, session_id: str) -> None:
        # run_copilot_stream refuses an unknown session itself: accept, then close 4000,
        # so a client that connected to a typo learns it in band.
        await run_copilot_stream(sessions, session_id, ws)
