"""Tool backend interface.

A backend is transport and nothing else. It does not validate arguments, enforce
authorisation, apply timeouts, retry, or redact -- all of that lives in the executor, so
that adding a backend cannot weaken a guarantee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ccas.schemas.common import JsonValue
from ccas.schemas.tools import ToolPayload, ToolSpec, ToolStatus

__all__ = ["BackendResponse", "ToolBackend"]


@dataclass(slots=True)
class BackendResponse:
    """A raw, unredacted backend reply. Never returned to a caller as-is."""

    data: dict[str, JsonValue] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.OK
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ToolBackend(ABC):
    """Executes a registered tool against a real or simulated system."""

    name: str

    @abstractmethod
    async def invoke(self, spec: ToolSpec, payload: ToolPayload) -> BackendResponse:
        """Run the tool.

        Raise ``TimeoutError`` to signal a timeout; the executor also imposes its own
        bound, so a backend that hangs is still cut off.
        """

    async def aclose(self) -> None:
        return None
