"""Extracting JSON from a model that was merely *asked* for it.

Native structured output makes the body valid JSON by construction. Most free models do
not support it, so the platform asks in the prompt and takes what comes back -- which is
routinely a fenced block, or prose with an object in the middle, or reasoning followed
by the answer.

This is deliberately narrow. It finds a JSON object in a string; it does not repair
malformed JSON, because a model that emitted broken JSON has misunderstood the task and
guessing at its intent would put invented values into a caller's transcript.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["JSON_INSTRUCTION", "extract_json"]

#: Appended to a system prompt when a model cannot be constrained by the server.
JSON_INSTRUCTION = """\

OUTPUT FORMAT -- this overrides any other instruction about how to answer.

Your entire response must be one JSON object matching the schema below. Begin with the
character `{{` and end with `}}`. Do not think out loud, do not explain your reasoning,
do not use markdown fences, do not write anything before or after the object.

{schema}"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _balanced_object(text: str) -> str | None:
    """The first brace-balanced object, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the JSON object in ``text``, or None if there is not exactly one to find.

    Tried in order: the whole string, a fenced block, the first balanced object. None is
    a real answer -- the caller raises rather than inventing a result.
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = [text.strip()]
    candidates.extend(match.group(1) for match in _FENCE.finditer(text))
    balanced = _balanced_object(text)
    if balanced is not None:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
