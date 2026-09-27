"""Construct a provider for a binding.

The only place that turns configuration into a concrete client. Keeping it here means a
call site names a *node* ("taxonomy_labeler") and never a provider, which is what makes
the hybrid swappable (CLAUDE.md Rule 6).
"""

from __future__ import annotations

from pydantic import SecretStr

from ccas.config.settings import Settings
from ccas.llm.anthropic_provider import AnthropicProvider
from ccas.llm.base import LLMProvider, ProviderUnavailableError
from ccas.llm.openrouter_provider import OpenRouterProvider
from ccas.llm.vllm_provider import VLLMProvider
from ccas.schemas.llm import ModelBinding, ProviderName

__all__ = ["AUTH_HELP", "build_provider", "key_is_set"]


def key_is_set(key: object) -> bool:
    """True only when a credential is present *and* non-empty.

    An empty string is not a credential. ``OPENROUTER_API_KEY=`` in .env or the shell
    otherwise reads as configured, readiness reports a provider that does not exist,
    and every call then fails at the vendor with a 401 that looks like a broken key
    rather than a missing one -- which is exactly how an exposed key got "verified" as
    live by its 429s (resume note, 2026-09-19)."""
    return bool(key is not None and getattr(key, "get_secret_value", lambda: key)())


def _key_text(key: SecretStr | None) -> str:
    """Precondition: ``key_is_set(key)``. Kept apart from it because narrowing through a
    helper is invisible to mypy -- the assert is what proves the branch to --strict."""
    assert key is not None
    return key.get_secret_value()


AUTH_HELP = """\
no Anthropic credentials found. Any one of these works:
  - `ant auth login`                     stores a profile the SDK reads automatically
  - export ANTHROPIC_API_KEY=sk-ant-...  a key in the environment or .env
  - export ANTHROPIC_AUTH_TOKEN=...      an OAuth bearer token
  - workload identity federation         ANTHROPIC_FEDERATION_RULE_ID and friends
Or run against the self-hosted binding instead: --provider vllm\
"""


def build_provider(binding: ModelBinding, settings: Settings) -> LLMProvider:
    if binding.provider is ProviderName.OPENROUTER:
        key = settings.openrouter_api_key
        if not key_is_set(key):
            raise ProviderUnavailableError(
                f"node {binding.node!r} is bound to openrouter but OPENROUTER_API_KEY "
                "is unset or empty; add it to .env or export it"
            )
        return OpenRouterProvider(
            api_key=_key_text(key),
            base_url=settings.openrouter_base_url,
            referer=settings.openrouter_referer,
            title=settings.openrouter_title,
        )

    if binding.provider is ProviderName.VLLM:
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key.get_secret_value(),
        )

    # Pass a key only when one is configured here. Otherwise construct with no arguments
    # and let the SDK run its own resolution chain -- env var, auth token, an
    # `ant auth login` profile, workload identity. Demanding ANTHROPIC_API_KEY up front
    # would reject a machine that is perfectly well authenticated by another route.
    key = settings.anthropic_api_key
    try:
        return AnthropicProvider(api_key=key.get_secret_value() if key is not None else None)
    except ProviderUnavailableError:
        raise
    except Exception as exc:
        # The SDK raises at construction when it finds no credential at all.
        raise ProviderUnavailableError(
            f"node {binding.node!r} is bound to anthropic but {AUTH_HELP}"
        ) from exc
