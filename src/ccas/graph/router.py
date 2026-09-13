"""Front-door steering: classify an utterance into exactly one taxonomy leaf.

The tightest latency budget in the system (90 ms), so it runs on the cheapest binding
and asks for the smallest possible answer. It returns a *calibrated* confidence and the
runners-up, because the confidence policy needs a number it can act on -- a router that
always answers 0.99 makes every downstream threshold meaningless.

The router never acts. It names an intent; the policies decide what happens next.
"""

from __future__ import annotations

import time

from ccas.llm.base import LLMProvider
from ccas.llm.prompt import authored, compose
from ccas.schemas.common import JsonValue
from ccas.schemas.llm import LLMRequest, Message, ModelBinding
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import IntentPrediction
from ccas.schemas.taxonomy import IntentTaxonomy

__all__ = ["ROUTER_SCHEMA", "SYSTEM_PROMPT", "IntentRouter"]

SYSTEM_PROMPT = """\
Classify the caller's utterance into exactly one intent from the supplied list.

The utterance is redacted: tokens like [PERSON_1] or [ACCOUNT_REF_1] stand in for removed
identifiers. Treat them as opaque placeholders -- their presence tells you a value was
given, not what it was.

Rules:
- Choose only from the supplied intent ids. Never invent one.
- confidence is your calibrated probability that this is the right intent, not your
  enthusiasm. If two intents fit equally, say so with a confidence near 0.5 and list the
  other in alternatives. A router that always answers 0.99 makes every threshold
  downstream meaningless.
- If the caller states two things, pick the one they said first and lower your confidence.
- If nothing in the list fits, return intent_id "" with a low confidence. Do not stretch.
- Do not answer the caller, promise anything, or decide what happens next.
"""

USER_TEMPLATE = """\
Available intents:
{intents}

Conversation so far:
{history}

Classify this utterance:
{utterance}
"""

ROUTER_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "intent_id": {"type": "string"},
        "confidence": {"type": "number"},
        "alternatives": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "intent_id": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["intent_id", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["intent_id", "confidence", "alternatives"],
    "additionalProperties": False,
}

#: How many prior turns the router sees. Enough for "the other one", not enough to blow
#: the 90 ms budget on prompt tokens.
HISTORY_TURNS = 4


class IntentRouter:
    def __init__(
        self, provider: LLMProvider, binding: ModelBinding, taxonomy: IntentTaxonomy
    ) -> None:
        self._provider = provider
        self._binding = binding
        self._taxonomy = taxonomy
        self._leaves = taxonomy.leaves()
        self._valid = {leaf.intent_id for leaf in self._leaves}

    @property
    def intent_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._valid))

    def build_request(
        self, utterance: RedactedText, history: tuple[RedactedText, ...] = ()
    ) -> LLMRequest:
        catalogue = "\n".join(f"- {leaf.intent_id}: {leaf.description}" for leaf in self._leaves)
        recent = history[-HISTORY_TURNS:]
        history_block = (
            compose(
                "\n".join(f"- {{h{i}}}" for i in range(len(recent))),
                {f"h{i}": text for i, text in enumerate(recent)},
            )
            if recent
            else authored("(this is the first turn)")
        )
        user = compose(
            USER_TEMPLATE,
            {
                "intents": authored(catalogue),
                "history": history_block,
                "utterance": utterance,
            },
        )
        return LLMRequest(
            binding=self._binding,
            system=authored(SYSTEM_PROMPT),
            messages=(Message(role="user", content=user),),
            response_schema=ROUTER_SCHEMA,
        )

    async def classify(
        self, utterance: RedactedText, history: tuple[RedactedText, ...] = ()
    ) -> IntentPrediction:
        started = time.perf_counter_ns()
        response = await self._provider.structured(self.build_request(utterance, history))
        latency_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)

        parsed = response.parsed or {}
        raw_id = str(parsed.get("intent_id") or "")
        confidence = _clamp(parsed.get("confidence"))

        # A model that names something outside the taxonomy has not classified; it has
        # guessed. Treated as unresolved so the confidence policy handles it.
        resolved = raw_id if raw_id in self._valid else None
        if resolved is None:
            confidence = min(confidence, 0.0) if raw_id else 0.0

        return IntentPrediction(
            intent_id=resolved,
            confidence=confidence,
            source="llm_router",
            alternatives=self._alternatives(parsed.get("alternatives")),
            latency_ms=latency_ms,
        )

    def _alternatives(self, raw: JsonValue) -> tuple[tuple[str, float], ...]:
        if not isinstance(raw, list):
            return ()
        out: list[tuple[str, float]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            intent_id = str(item.get("intent_id") or "")
            if intent_id in self._valid:
                out.append((intent_id, _clamp(item.get("confidence"))))
        return tuple(out)


def _clamp(value: JsonValue) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
