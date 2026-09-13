"""Greeting: the first thing a caller hears.

Consent comes first where the pack's compliance profile requires it -- a recorded call
that never told the caller it was recording is a defect that no later step can repair.
"""

from __future__ import annotations

from typing import Any

from ccas.graph.context import GraphContext
from ccas.nodes.base import NodeFn, speak
from ccas.schemas.session import SessionState

__all__ = ["NAME", "make_node"]

NAME = "greet"


def make_node(ctx: GraphContext) -> NodeFn:
    async def greet(state: SessionState) -> dict[str, Any]:
        pack = ctx.pack
        parts: list[str] = []
        if pack.compliance.recording_consent_required and pack.compliance.consent_prompt:
            parts.append(pack.compliance.consent_prompt.strip())
        parts.append(pack.greeting.strip())
        return speak(ctx, state, " ".join(parts))

    return greet
