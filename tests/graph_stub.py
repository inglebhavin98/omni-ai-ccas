"""Deterministic router and responder stand-ins for graph tests.

Keyword-matched rather than model-driven, so a graph test asserts the *graph* and never
a model's mood. Confidence is scriptable, which is what makes the policy bands testable
from the outside.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from ccas.llm.base import LLMProvider
from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, ProviderName

__all__ = ["StubGraphProvider"]

_UTTERANCE = re.compile(r"Classify this utterance:\s*(.+)\s*$", re.DOTALL)

#: keyword -> (intent_id, confidence)
_INTENTS: tuple[tuple[str, str, float], ...] = (
    ("dispute", "disputes.chargeback.raise", 0.94),
    ("chargeback", "disputes.chargeback.raise", 0.94),
    ("return", "returns.initiate.open", 0.91),
    ("send it back", "returns.initiate.open", 0.91),
    ("change the address", "fulfilment.address.change", 0.90),
    ("redirect", "fulfilment.address.change", 0.90),
    ("where is", "fulfilment.tracking.status", 0.93),
    ("delivery", "fulfilment.tracking.status", 0.93),
    ("track", "fulfilment.tracking.status", 0.93),
    ("something else", "fulfilment.tracking.status", 0.60),  # clarify band
    ("no idea", "", 0.10),  # escalate band
)


class StubGraphProvider(LLMProvider):
    """Serves both the router and the responder; branches on the response schema."""

    name = ProviderName.VLLM

    def __init__(self, grounded: bool = True) -> None:
        self.grounded = grounded
        self.router_calls = 0
        self.responder_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:  # pragma: no cover
        raise NotImplementedError

    async def healthy(self) -> bool:
        return True

    async def structured(self, request: LLMRequest) -> LLMResponse:
        schema = request.response_schema or {}
        properties = schema.get("properties", {})
        if "intent_id" in properties:
            return self._route(request)
        return self._respond(request)

    def _route(self, request: LLMRequest) -> LLMResponse:
        self.router_calls += 1
        prompt = request.messages[0].content.text
        match = _UTTERANCE.search(prompt)
        utterance = (match.group(1) if match else prompt).lower()

        intent_id, confidence = "", 0.15
        for keyword, candidate, score in _INTENTS:
            if keyword in utterance:
                intent_id, confidence = candidate, score
                break

        alternatives = (
            [{"intent_id": "returns.initiate.open", "confidence": 0.55}]
            if 0.4 < confidence < 0.8
            else []
        )
        payload = {
            "intent_id": intent_id,
            "confidence": confidence,
            "alternatives": alternatives,
        }
        return LLMResponse(
            binding=request.binding,
            text=json.dumps(payload),
            parsed=payload,
            ttft_ms=1,
            total_ms=1,
        )

    def _respond(self, request: LLMRequest) -> LLMResponse:
        self.responder_calls += 1
        body = request.messages[0].content.text
        has_results = "FAILED" not in body and "(no tools were called)" not in body
        grounded = self.grounded and has_results
        payload = {
            "reply": (
                "That's on its way and should reach you shortly."
                if grounded
                else "I don't have that to hand."
            ),
            "grounded": grounded,
            "missing": [] if grounded else ["status"],
        }
        return LLMResponse(
            binding=request.binding,
            text=json.dumps(payload),
            parsed=payload,
            ttft_ms=1,
            total_ms=1,
        )
