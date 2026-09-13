"""Channel differences are data, not branches (ADR-0019).

Rule 1 forbids `if domain == ...` in the core; the same reasoning applies to `if channel
== "chat"`. What differs between channels is configuration -- how hard to redact, what
latency is acceptable, whether keypad entry means anything -- so it loads from a file the
way a domain pack does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ccas.config.channels import (
    CHANNELS_FILENAME,
    UnknownChannelError,
    load_channel_profiles,
)
from ccas.redaction.pipeline import RedactionMode
from ccas.schemas.common import Channel

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs"


def test_every_declared_channel_has_a_profile() -> None:
    """A channel the platform can be handed but has no profile for is a latent crash."""
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    for channel in Channel:
        assert channel in profiles, f"no profile for {channel.value}"


def test_chat_uses_batch_redaction() -> None:
    """The point of the pivot: chat has no 3 ms slice, so it can afford the NER pass.

    ADR-0007 split the pipeline because voice could not afford spaCy. Chat can, and
    leaving it on REALTIME would carry a voice constraint into a channel that never had it.
    """
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    assert profiles[Channel.CHAT].redaction_mode is RedactionMode.BATCH


def test_voice_stays_on_realtime_redaction() -> None:
    """Freezing voice must not silently change how voice behaves."""
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    assert profiles[Channel.VOICE].redaction_mode is RedactionMode.REALTIME


def test_only_voice_supports_keypad_entry() -> None:
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    assert profiles[Channel.VOICE].supports_keypad_entry
    assert not profiles[Channel.CHAT].supports_keypad_entry


def test_chat_gets_a_looser_budget_than_voice() -> None:
    """Not a relaxation of the voice budget -- a different contract for a different channel."""
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    assert profiles[Channel.CHAT].budget_ms > profiles[Channel.VOICE].budget_ms


def test_an_unknown_channel_raises_rather_than_defaulting() -> None:
    """Falling back to some default would silently redact a new channel the weak way."""
    profiles = load_channel_profiles(CONFIGS / CHANNELS_FILENAME)
    with pytest.raises(UnknownChannelError, match=r"no profile for channel"):
        profiles.require("sms_v2")  # type: ignore[arg-type]


def test_a_profile_naming_an_unknown_channel_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "channels.yaml"
    path.write_text("channels:\n  carrier_pigeon:\n    redaction_mode: batch\n    budget_ms: 900\n")
    with pytest.raises(ValidationError):
        load_channel_profiles(path)


def test_the_core_contains_no_channel_branch() -> None:
    """The invariant this whole mechanism exists to preserve."""
    offenders = []
    for path in (REPO / "src" / "ccas").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("if ", "elif ")):
                continue
            if "Channel." in stripped or "channel ==" in stripped:
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, f"channel branching in the core: {offenders}"


def test_build_context_takes_its_redaction_mode_from_the_profile() -> None:
    """The hardcoded REALTIME is gone -- this is what the profile is for."""
    from ccas.config.domain_loader import load_domain
    from ccas.graph.context import build_context
    from ccas.llm.bindings import load_bindings
    from tests.graph_stub import StubGraphProvider

    domain = load_domain(REPO / "domains", "retail")
    bindings = load_bindings(REPO / "configs" / "models.yaml")

    chat = build_context(domain, StubGraphProvider(), bindings, channel=Channel.CHAT)
    voice = build_context(domain, StubGraphProvider(), bindings, channel=Channel.VOICE)

    assert chat.redaction.mode is RedactionMode.BATCH
    assert voice.redaction.mode is RedactionMode.REALTIME
    assert chat.channel is Channel.CHAT


def test_build_context_defaults_to_chat() -> None:
    """Chat is the proving ground now (ADR-0018); voice is opt-in."""
    from ccas.config.domain_loader import load_domain
    from ccas.graph.context import build_context
    from ccas.llm.bindings import load_bindings
    from tests.graph_stub import StubGraphProvider

    domain = load_domain(REPO / "domains", "retail")
    ctx = build_context(
        domain, StubGraphProvider(), load_bindings(REPO / "configs" / "models.yaml")
    )
    assert ctx.channel is Channel.CHAT
    assert ctx.redaction.mode is RedactionMode.BATCH
