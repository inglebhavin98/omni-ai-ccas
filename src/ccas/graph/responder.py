"""Grounded response generation.

The one place the platform speaks a sentence it composed itself, and therefore the one
place it could invent a fact about a caller's account. The prompt forbids it, the schema
carries a ``grounded`` flag the node checks, and the tool results are the only evidence
supplied -- when they are empty or failed, the correct output is an admission and a
handoff, not a guess (CLAUDE.md Rule 4).
"""

from __future__ import annotations

import json
import time

from ccas.llm.base import LLMProvider
from ccas.llm.prompt import authored, compose
from ccas.schemas.common import JsonValue
from ccas.schemas.llm import LLMRequest, Message, ModelBinding
from ccas.schemas.pii import RedactedText
from ccas.schemas.tools import ToolRecord

__all__ = ["RESPONSE_SCHEMA", "SYSTEM_PROMPT", "GroundedReply", "Responder"]

SYSTEM_PROMPT = """\
You are speaking to a caller on a phone. Answer using only the tool results supplied.

Rules:
- Every fact you state must appear in the tool results. If a value is not there, you do
  not have it. Say so and offer to transfer -- never estimate a date, an amount or a
  status.
- Tokens like [PERSON_1] or [ACCOUNT_REF_1] are redacted placeholders. Never read one
  aloud; refer to "your reference" or "the name on the account" instead.
- One or two sentences. This is speech: no lists, no markdown, no spelling out fields.
- Do not promise a future action unless a tool result confirms it happened.
- Set grounded false if you could not answer from the results. Do not answer anyway.
"""

USER_TEMPLATE = """\
Caller's request: {utterance}

Intent: {intent}

Tool results:
{results}
"""

RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "grounded": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "grounded", "missing"],
    "additionalProperties": False,
}


class GroundedReply:
    __slots__ = ("grounded", "latency_ms", "missing", "reply")

    def __init__(
        self,
        reply: str,
        grounded: bool,
        missing: tuple[str, ...],
        latency_ms: int = 0,
    ) -> None:
        self.reply = reply
        self.grounded = grounded
        self.missing = missing
        self.latency_ms = latency_ms
        """Time to the completed reply. The ledger's `llm_ttft_ms` slice -- structured
        output cannot stream usefully, so first-token and completion coincide here."""

    def __repr__(self) -> str:
        return (
            f"GroundedReply(grounded={self.grounded}, "
            f"missing={list(self.missing)}, latency_ms={self.latency_ms})"
        )


class Responder:
    def __init__(self, provider: LLMProvider, binding: ModelBinding) -> None:
        self._provider = provider
        self._binding = binding

    def build_request(
        self, utterance: RedactedText, intent_id: str | None, records: tuple[ToolRecord, ...]
    ) -> LLMRequest:
        rendered = _render(records)
        user = compose(
            USER_TEMPLATE,
            {
                "utterance": utterance,
                "intent": authored(intent_id or "(not identified)"),
                "results": authored(rendered),
            },
        )
        return LLMRequest(
            binding=self._binding,
            system=authored(SYSTEM_PROMPT),
            messages=(Message(role="user", content=user),),
            response_schema=RESPONSE_SCHEMA,
        )

    async def reply(
        self, utterance: RedactedText, intent_id: str | None, records: tuple[ToolRecord, ...]
    ) -> GroundedReply:
        started = time.perf_counter_ns()
        response = await self._provider.structured(
            self.build_request(utterance, intent_id, records)
        )
        latency_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
        parsed = response.parsed or {}
        missing = parsed.get("missing")
        return GroundedReply(
            reply=str(parsed.get("reply") or ""),
            grounded=bool(parsed.get("grounded")),
            missing=tuple(str(m) for m in missing) if isinstance(missing, list) else (),
            latency_ms=latency_ms,
        )


def _render(records: tuple[ToolRecord, ...]) -> str:
    """Only egress-permitted results are shown. An unredacted one is reported as failed
    rather than omitted, so the model knows a gap exists instead of assuming none."""
    if not records:
        return "(no tools were called)"
    lines: list[str] = []
    for record in records:
        name = record.payload.tool_name
        result = record.result
        if result.ok and result.safe_for_model:
            lines.append(f"{name}: {json.dumps(result.data, sort_keys=True)}")
        elif result.ok:
            lines.append(f"{name}: FAILED (result could not be made safe to show)")
        else:
            lines.append(f"{name}: FAILED ({result.error_code})")
    return "\n".join(lines)
