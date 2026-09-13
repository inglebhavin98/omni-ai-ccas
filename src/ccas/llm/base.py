"""Provider-neutral LLM interface.

Two bindings ship and stay supported (CLAUDE.md Rule 6), so no module above this one
may import a provider SDK. A provider receives only ``LLMRequest`` -- whose every text
field is a ``RedactedText`` -- which is how Rule 2 is enforced at the egress boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, ProviderName

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "ProviderRateLimitedError",
    "ProviderUnavailableError",
]


class LLMProviderError(RuntimeError):
    """Base for provider failures that callers are expected to handle."""


class ProviderUnavailableError(LLMProviderError):
    """Provider is not configured or not reachable.

    Raised rather than silently falling back to the other binding: an unnoticed
    failover would make the parity eval meaningless.
    """


class ProviderRateLimitedError(LLMProviderError):
    """The request was never served -- quota, free-tier capacity, or a cold model.

    Distinct from a failure because nothing was measured. The parity harness drops such a
    case rather than scoring it as disagreement (``ccas.evals.parity``), and a caller on
    the call path should escalate rather than retry into the same wall.
    """


class LLMProvider(ABC):
    """Every binding implements exactly this surface."""

    name: ProviderName

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Single-shot completion. Used off the call path (labeler, judge, summarizer)."""

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Token stream. Required on the call path so TTS can start before generation ends."""

    @abstractmethod
    async def structured(self, request: LLMRequest) -> LLMResponse:
        """Schema-constrained completion.

        ``request.response_schema`` must be set; the returned ``LLMResponse.parsed`` is
        guaranteed to validate against it, so callers never parse free text.
        """

    @abstractmethod
    async def healthy(self) -> bool:
        """Cheap reachability probe used by the parity harness and readiness checks."""

    async def aclose(self) -> None:
        """Release pooled connections. Overridden where the client owns a session."""
        return None
