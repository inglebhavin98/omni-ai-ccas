"""A deterministic labeler stand-in for mining tests.

Reads the exemplar block out of the composed prompt and derives a stable label from it,
so the taxonomy that comes out has real structure without a model call (Rule 7).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from ccas.llm.base import LLMProvider
from ccas.schemas.llm import LLMChunk, LLMRequest, LLMResponse, ProviderName

__all__ = ["StubLabelerProvider"]

#: Anchored to the exemplar section. The tool catalogue is also a "- " list, so a
#: bare bullet match would read tool names as caller utterances.
_EXEMPLAR_BLOCK = re.compile(r"Representative utterances:\n(.*)\Z", re.DOTALL)
_BULLET = re.compile(r"^- (.+)$", re.MULTILINE)
_FILLER = ("thanks", "thank you", "that's everything", "goodbye", "no that's all")

#: Keyword -> (l1, l2). Matches the synthetic corpus's openers.
_TOPICS: tuple[tuple[str, str, str], ...] = (
    ("status", "status", "check"),
    ("where", "status", "locate"),
    ("change", "account", "update"),
    ("contact information", "account", "update"),
    ("charge", "payments", "dispute"),
    ("statement", "payments", "explain"),
    ("cancel", "lifecycle", "cancel"),
    ("person", "routing", "human"),
)


class StubLabelerProvider(LLMProvider):
    name = ProviderName.VLLM

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:  # pragma: no cover
        raise NotImplementedError

    async def healthy(self) -> bool:
        return True

    async def structured(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        prompt = request.messages[0].content.text
        block = _EXEMPLAR_BLOCK.search(prompt)
        exemplars = [m.group(1).lower() for m in _BULLET.finditer(block.group(1) if block else "")]
        joined = " ".join(exemplars)

        l1, l2 = "general", "request"
        for keyword, first, second in _TOPICS:
            if keyword in joined:
                l1, l2 = first, second
                break

        is_intent = not any(f in joined for f in _FILLER) or l1 != "general"
        payload = {
            "l1_slug": l1,
            "l1_label": l1.title(),
            "l1_description": f"Contacts about {l1}.",
            "l2_slug": l2,
            "l2_label": l2.title(),
            "l2_description": f"{l2.title()} within {l1}.",
            "l3_slug": "resolve",
            "l3_label": "Resolve",
            "l3_description": "Complete the request end to end.",
            # Named tools come with slots for their required arguments -- the loader
            # rejects a taxonomy whose slots do not cover the tools it calls.
            "slots": [
                {
                    "name": "order_reference",
                    "slot_type": "identifier",
                    "required": True,
                    "elicitation_prompt": "What is your reference?",
                    "pii_sensitive": True,
                    "dtmf_capturable": True,
                }
            ],
            "required_tools": ["get_order_status", "a_tool_that_does_not_exist"],
            "is_intent": is_intent,
            "risk_tier": "low",
            "complexity": 0.3,
            "feasibility": 0.85,
            "rationale": "stub labeler",
        }
        return LLMResponse(
            binding=request.binding,
            text=json.dumps(payload),
            parsed=payload,
            ttft_ms=1,
            total_ms=1,
        )
