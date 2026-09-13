"""Per-model request-shaping rules for the Anthropic binding.

The Messages API rejects parameters a given model does not support (``effort`` on
Haiku, ``budget_tokens`` on the current generation), so the binding consults this
table instead of sending a one-size-fits-all request and hoping.
"""

from __future__ import annotations

from ccas.schemas.common import Frozen

__all__ = ["ModelCapabilities", "capabilities_for"]


class ModelCapabilities(Frozen):
    adaptive_thinking: bool
    """``thinking={"type": "adaptive"}``. Omitted entirely where unsupported."""

    effort: bool
    """``output_config.effort``. Errors on Haiku 4.5 and older Sonnet."""

    refusal_fallbacks: bool
    """Server-side fallback routing for a ``stop_reason == "refusal"`` turn."""


_OPUS_5_CLASS = ModelCapabilities(adaptive_thinking=True, effort=True, refusal_fallbacks=True)
_SONNET_5_CLASS = ModelCapabilities(adaptive_thinking=True, effort=True, refusal_fallbacks=False)
_HAIKU_CLASS = ModelCapabilities(adaptive_thinking=False, effort=False, refusal_fallbacks=False)

#: Exact ids, because prefix matching would silently mis-shape a future model.
_TABLE: dict[str, ModelCapabilities] = {
    "claude-opus-5": _OPUS_5_CLASS,
    "claude-opus-4-8": _OPUS_5_CLASS,
    "claude-opus-4-7": _OPUS_5_CLASS,
    "claude-fable-5": _OPUS_5_CLASS,
    "claude-fable-5-1": _OPUS_5_CLASS,
    "claude-sonnet-5": _SONNET_5_CLASS,
    "claude-haiku-4-5": _HAIKU_CLASS,
}

#: Anything unrecognised gets the most conservative shape: a wrong 400 at startup is
#: worse than a slightly plainer request.
_CONSERVATIVE = ModelCapabilities(adaptive_thinking=False, effort=False, refusal_fallbacks=False)


def capabilities_for(model: str) -> ModelCapabilities:
    return _TABLE.get(model, _CONSERVATIVE)
