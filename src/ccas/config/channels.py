"""Channel profiles -- what differs between voice, chat and the rest, as data.

Rule 1 forbids branching on domain in the core, and the same argument applies to channel:
the moment `if channel == "chat"` appears in a node, every future channel is a code change
and the core stops being channel-agnostic. So channel differences load from
``configs/channels.yaml`` the way a vertical loads from ``domains/``.

What legitimately differs is small and concrete: how hard to redact (voice cannot afford
the NER pass at 3 ms, chat can), what latency is acceptable, and whether keypad entry
means anything. See docs/adr/0019-channel-profiles.md.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from ccas.redaction.pipeline import RedactionMode
from ccas.schemas.common import Channel, Frozen, SchemaVersion

__all__ = [
    "CHANNELS_FILENAME",
    "ChannelProfile",
    "ChannelProfiles",
    "UnknownChannelError",
    "load_channel_profiles",
]

CHANNELS_FILENAME = "channels.yaml"


class UnknownChannelError(LookupError):
    """A channel with no profile. Raised rather than defaulted.

    Falling back to a default would silently hand a new channel some other channel's
    redaction mode -- and the cheap default is the weak one.
    """


class ChannelProfile(Frozen):
    """How the platform behaves on one channel."""

    redaction_mode: RedactionMode
    """REALTIME skips the NER pass to fit a 3 ms slice (ADR-0007). Only voice needs that
    trade; every text channel should be on BATCH."""

    budget_ms: int = Field(ge=1)
    budget_ref: str = Field(min_length=1)
    """Filename of the per-stage budget, resolved against the config directory."""

    supports_keypad_entry: bool = False
    """DTMF. Meaningful on a phone call and nowhere else."""


class ChannelProfiles(Frozen):
    schema_version: SchemaVersion = "1.0"
    channels: dict[Channel, ChannelProfile]

    def __contains__(self, channel: object) -> bool:
        return channel in self.channels

    def __getitem__(self, channel: Channel) -> ChannelProfile:
        return self.channels[channel]

    def require(self, channel: Channel) -> ChannelProfile:
        """The profile, or a named failure. Never a default."""
        try:
            return self.channels[channel]
        except KeyError as exc:
            known = ", ".join(sorted(c.value for c in self.channels))
            raise UnknownChannelError(
                f"no profile for channel {channel!r}; configs/{CHANNELS_FILENAME} declares {known}"
            ) from exc


def load_channel_profiles(path: Path) -> ChannelProfiles:
    """Load and validate every channel profile.

    Every member of ``Channel`` must be present: a channel the platform can be handed but
    has no profile for is a crash waiting for the first caller who uses it.
    """
    profiles = ChannelProfiles.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    missing = sorted(c.value for c in Channel if c not in profiles.channels)
    if missing:
        raise ValueError(f"{path} has no profile for: {', '.join(missing)}")
    return profiles
