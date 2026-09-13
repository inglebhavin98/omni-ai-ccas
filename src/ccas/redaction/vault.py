"""Session placeholder vault: the reverse map redaction deliberately does not keep.

Rule 2 exists to stop caller data reaching a *model* or leaving the process. It was
never meant to stop an internal system of record receiving the reference the caller just
read out -- but that is what happens when a value is replaced everywhere: the tool sees
``[ACCOUNT_REF_1]`` and cannot act.

The vault closes that gap without weakening the gate. It holds token -> original for the
life of one session, in process memory only:

- it is never a field on a Pydantic model, so it cannot be serialized into a CallLog, a
  HandoffContext, a checkpoint or a log line;
- ``__repr__`` and ``__str__`` are blind, so a traceback cannot print it;
- only the tool executor reads it, immediately before dispatch to a registered backend.

Everything a model or a human sees still carries the placeholder. See
docs/adr/0010-placeholder-vault.md.
"""

from __future__ import annotations

import re

__all__ = ["PLACEHOLDER_PATTERN", "PlaceholderVault"]

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z0-9_]+_\d+\]")


class PlaceholderVault:
    """Token -> original, scoped to one session."""

    __slots__ = ("_originals",)

    def __init__(self) -> None:
        self._originals: dict[str, str] = {}

    def store(self, token: str, original: str) -> None:
        # First write wins: the allocator hands out one token per distinct surface, so a
        # second value under the same token would mean two entities collapsed into one.
        self._originals.setdefault(token, original)

    def resolve(self, token: str) -> str | None:
        return self._originals.get(token)

    def detokenize(self, text: str) -> str:
        """Restore every placeholder this vault knows. Unknown tokens are left alone --
        a token from another session must not silently resolve to nothing."""
        return PLACEHOLDER_PATTERN.sub(lambda m: self._originals.get(m.group(0), m.group(0)), text)

    def contains_unknown(self, text: str) -> tuple[str, ...]:
        """Tokens in ``text`` this vault cannot resolve. A non-empty result means the
        value would reach a backend still masked."""
        return tuple(
            sorted(
                {
                    token
                    for token in PLACEHOLDER_PATTERN.findall(text)
                    if token not in self._originals
                }
            )
        )

    def __len__(self) -> int:
        return len(self._originals)

    def __repr__(self) -> str:
        return f"PlaceholderVault(entries={len(self._originals)})"

    __str__ = __repr__
