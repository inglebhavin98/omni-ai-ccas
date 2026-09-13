"""Resolve a call site to a concrete ``ModelBinding`` from ``configs/models.yaml``.

Hard-coding a model id anywhere else is a defect (CLAUDE.md Rule 5).

Each node declares two or more **variants**. A variant is usually a provider, but need
not be: when every model lives behind one gateway, parity is between two *models* rather
than two vendors. Rule 6's intent -- that no node can silently depend on one model -- is
preserved either way, and enforced here at load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ccas.schemas.llm import ModelBinding, ProviderName

__all__ = ["MIN_VARIANTS", "BindingRegistry", "UnknownNodeError", "load_bindings"]

#: Rule 6: a node bound to exactly one model can never be parity-tested.
MIN_VARIANTS = 2


class UnknownNodeError(LookupError):
    """A call site asked for a binding that ``models.yaml`` does not declare."""


class BindingRegistry:
    """Immutable node -> variant -> binding lookup."""

    __slots__ = ("_bindings", "_default")

    def __init__(
        self, bindings: dict[str, dict[str, ModelBinding]], default_provider: ProviderName
    ) -> None:
        self._bindings = bindings
        self._default = default_provider

    @property
    def default_provider(self) -> ProviderName:
        return self._default

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def variants(self, node: str) -> tuple[str, ...]:
        return tuple(self._variants(node))

    def _variants(self, node: str) -> dict[str, ModelBinding]:
        try:
            return self._bindings[node]
        except KeyError as exc:
            raise UnknownNodeError(
                f"no binding for node {node!r}; declared nodes: {list(self.nodes)}"
            ) from exc

    def resolve(self, node: str, variant: str | ProviderName | None = None) -> ModelBinding:
        """A binding by variant name, by provider, or the node's default.

        The default is the first variant on the configured default provider, so a call
        site that names only a node still gets a deliberate choice rather than an
        arbitrary one.
        """
        variants = self._variants(node)
        if variant is None:
            preferred = next((b for b in variants.values() if b.provider is self._default), None)
            return preferred or next(iter(variants.values()))

        key = variant.value if isinstance(variant, ProviderName) else variant
        if key in variants:
            return variants[key]
        by_provider = [b for b in variants.values() if b.provider.value == key]
        if by_provider:
            return by_provider[0]
        raise UnknownNodeError(
            f"node {node!r} has no variant {key!r}; available: {sorted(variants)}"
        )

    def pair(self, node: str) -> tuple[ModelBinding, ModelBinding]:
        """Two bindings for a node, for the parity harness (Rule 6)."""
        variants = list(self._variants(node).values())
        return variants[0], variants[1]

    def models(self) -> dict[str, list[str]]:
        """Every model in play, per node. Useful in a readiness report."""
        return {
            node: [f"{b.provider.value}:{b.model}" for b in variants.values()]
            for node, variants in self._bindings.items()
        }


def load_bindings(path: Path) -> BindingRegistry:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = ProviderName(raw.get("default_provider", "openrouter"))

    bindings: dict[str, dict[str, ModelBinding]] = {}
    for node, per_variant in (raw.get("bindings") or {}).items():
        resolved: dict[str, ModelBinding] = {}
        for variant_name, spec in per_variant.items():
            spec = dict(spec)
            # The variant key names the provider unless the spec says otherwise, so the
            # common case stays terse and a second model on one provider stays possible.
            provider = ProviderName(spec.pop("provider", variant_name))
            resolved[variant_name] = ModelBinding(
                node=node, provider=provider, variant=variant_name, **spec
            )
        _check_parity_is_possible(node, resolved)
        bindings[node] = resolved

    if not bindings:
        raise ValueError(f"{path} declares no bindings")
    return BindingRegistry(bindings, default)


def _check_parity_is_possible(node: str, resolved: dict[str, ModelBinding]) -> None:
    if len(resolved) < MIN_VARIANTS:
        raise ValueError(
            f"node {node!r} declares {len(resolved)} binding(s); at least "
            f"{MIN_VARIANTS} are required so it can be parity-tested (Rule 6)"
        )
    models = [f"{b.provider.value}:{b.model}" for b in resolved.values()]
    if len(set(models)) < MIN_VARIANTS:
        raise ValueError(
            f"node {node!r} declares {len(resolved)} variants but only "
            f"{len(set(models))} distinct model(s): {sorted(set(models))}. "
            "Comparing a model against itself measures nothing."
        )
