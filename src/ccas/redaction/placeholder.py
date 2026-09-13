"""Stable placeholder allocation.

The same entity must keep the same token for the life of a session, or the model loses
the ability to corefer -- "is that the same address you gave me?" becomes unanswerable
when the two mentions render as ``[ADDRESS_1]`` and ``[ADDRESS_2]``.

Surface forms are keyed by hash, so the allocator never retains the original text.
"""

from __future__ import annotations

from ccas.redaction.vault import PlaceholderVault
from ccas.schemas.pii import PiiEntityType, sha256_hex

__all__ = ["DEFAULT_FORMAT", "PlaceholderAllocator"]

DEFAULT_FORMAT = "[{entity}_{n}]"


class PlaceholderAllocator:
    """Assigns ``[ENTITY_N]`` tokens, stable per allocator instance.

    Scope one allocator per session (or per call log) -- sharing one across callers would
    let a token collide across conversations, and a fresh one per turn would break
    coreference within a conversation.
    """

    __slots__ = ("_assigned", "_counters", "_format", "_vault")

    def __init__(
        self,
        placeholder_format: str = DEFAULT_FORMAT,
        vault: PlaceholderVault | None = None,
    ) -> None:
        if "{entity}" not in placeholder_format or "{n}" not in placeholder_format:
            raise ValueError(
                f"placeholder format {placeholder_format!r} must contain {{entity}} and {{n}}"
            )
        self._format = placeholder_format
        self._counters: dict[str, int] = {}
        self._assigned: dict[str, str] = {}
        self._vault = vault
        """When supplied, every token is recorded against its original so the tool
        executor can restore it before dispatch. In-process only -- see vault.py."""

    def allocate(
        self,
        entity_type: PiiEntityType,
        surface: str,
        custom_label: str | None = None,
    ) -> str:
        """Token for this entity. Repeat surface forms get the token already issued."""
        label = (custom_label or entity_type.value).upper()
        key = f"{label}:{sha256_hex(surface.strip().casefold())}"
        existing = self._assigned.get(key)
        if existing is not None:
            return existing

        nth = self._counters.get(label, 0) + 1
        self._counters[label] = nth
        token = self._format.format(entity=label, n=nth)
        self._assigned[key] = token
        if self._vault is not None:
            self._vault.store(token, surface)
        return token

    @property
    def issued(self) -> int:
        return len(self._assigned)

    def counts(self) -> dict[str, int]:
        """Distinct entities seen per label. Safe to log -- no surface forms."""
        return dict(self._counters)
