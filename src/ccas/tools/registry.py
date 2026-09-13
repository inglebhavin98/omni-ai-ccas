"""Tool registry.

Two layers, the same pattern redaction uses: a global set in ``configs/tools.yaml`` that
means the same thing in every vertical, and per-pack tools declared in
``domains/<pack>/pack.yaml``. A pack may add; it may not shadow (CLAUDE.md Rule 1).

The registry is the authority on what exists. A model chooses *which* registered tool to
call and never constructs one, so a name that is not here cannot be dispatched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, Slug
from ccas.schemas.domain import DomainPack
from ccas.schemas.taxonomy import IntentNode
from ccas.schemas.tools import ToolSpec

__all__ = [
    "GLOBAL_TOOLS_PATH",
    "GlobalToolSet",
    "ToolNotRegisteredError",
    "ToolRegistry",
    "build_registry",
]

GLOBAL_TOOLS_PATH = Path("configs/tools.yaml")

#: Reserved domain for tools that belong to no vertical.
CORE_DOMAIN = "core"


class ToolNotRegisteredError(LookupError):
    """A call named a tool the registry does not hold."""


class GlobalToolSet(Frozen):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    tools: tuple[ToolSpec, ...] = ()

    @model_validator(mode="after")
    def _check_tools_are_core(self) -> Self:
        for tool in self.tools:
            if tool.domain != CORE_DOMAIN:
                raise ValueError(
                    f"global tool {tool.name!r} declares domain {tool.domain!r}; "
                    f"only {CORE_DOMAIN!r} belongs in the global set -- a vertical's "
                    "tool goes in its pack"
                )
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool name in the global set")
        return self


class ToolRegistry:
    """Immutable lookup over the global set plus one pack."""

    __slots__ = ("_by_name", "_domain", "_global_names")

    def __init__(self, global_tools: tuple[ToolSpec, ...], pack: DomainPack) -> None:
        self._global_names = frozenset(t.name for t in global_tools)
        clashing = self._global_names & {t.name for t in pack.tools}
        if clashing:
            raise ValueError(
                f"pack {pack.domain!r} shadows global tool(s) {sorted(clashing)}; "
                "a pack may add tools, never redefine a shared one"
            )
        self._by_name: dict[str, ToolSpec] = {t.name: t for t in (*global_tools, *pack.tools)}
        self._domain = pack.domain

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    @property
    def global_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._global_names))

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._by_name[name] for name in self.names)

    def is_global(self, name: str) -> bool:
        return name in self._global_names

    def resolve(self, name: str) -> ToolSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(
                f"tool {name!r} is not registered for domain {self._domain!r}; "
                f"registered: {list(self.names)}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._by_name

    def for_intent(self, node: IntentNode) -> tuple[ToolSpec, ...]:
        """Specs an intent declares. Raises if the taxonomy names an unregistered tool."""
        return tuple(self.resolve(name) for name in node.required_tools)

    def unresolved(self, names: tuple[Slug, ...]) -> tuple[str, ...]:
        return tuple(name for name in names if name not in self._by_name)


def load_global_tools(path: Path = GLOBAL_TOOLS_PATH) -> tuple[ToolSpec, ...]:
    if not path.is_file():
        return ()
    return GlobalToolSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).tools


def build_registry(pack: DomainPack, global_path: Path = GLOBAL_TOOLS_PATH) -> ToolRegistry:
    return ToolRegistry(load_global_tools(global_path), pack)
