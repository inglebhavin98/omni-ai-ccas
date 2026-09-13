"""Domain pack loading.

The pack is the *only* place a vertical's vocabulary is allowed to live (CLAUDE.md
Rule 1). Everything the orchestrator, redactor and miner need about a domain arrives
through this module.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from ccas.schemas.domain import DomainPack
from ccas.schemas.taxonomy import IntentTaxonomy

__all__ = [
    "DomainPackNotFoundError",
    "LoadedDomain",
    "available_domains",
    "load_domain",
    "load_domain_cached",
    "load_pack",
    "load_taxonomy",
]

PACK_FILENAME = "pack.yaml"


class DomainPackNotFoundError(LookupError):
    """Raised instead of falling back to a default -- a wrong pack is worse than a stop."""


class LoadedDomain:
    """A pack plus its mined taxonomy, resolved against one directory."""

    __slots__ = ("_taxonomy", "pack", "root")

    def __init__(self, pack: DomainPack, root: Path, taxonomy: IntentTaxonomy | None) -> None:
        self.pack = pack
        self.root = root
        self._taxonomy = taxonomy

    @property
    def domain(self) -> str:
        return self.pack.domain

    @property
    def has_taxonomy(self) -> bool:
        return self._taxonomy is not None

    @property
    def taxonomy(self) -> IntentTaxonomy:
        if self._taxonomy is None:
            raise DomainPackNotFoundError(
                f"domain {self.pack.domain!r} has no taxonomy at "
                f"{self.root / self.pack.taxonomy_ref}; run scripts/mine_taxonomy.py first"
            )
        return self._taxonomy

    def __repr__(self) -> str:
        return f"LoadedDomain(domain={self.pack.domain!r}, has_taxonomy={self.has_taxonomy})"


def available_domains(domains_dir: Path) -> tuple[str, ...]:
    if not domains_dir.is_dir():
        return ()
    return tuple(sorted(p.name for p in domains_dir.iterdir() if (p / PACK_FILENAME).is_file()))


def load_pack(domains_dir: Path, domain: str) -> DomainPack:
    path = domains_dir / domain / PACK_FILENAME
    if not path.is_file():
        known = available_domains(domains_dir)
        raise DomainPackNotFoundError(
            f"no domain pack at {path}; available packs: {list(known) or 'none'}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pack = DomainPack.model_validate(data)
    if pack.domain != domain:
        raise ValueError(
            f"pack at {path} declares domain {pack.domain!r} "
            f"but lives in a directory named {domain!r}"
        )
    return pack


def load_taxonomy(domains_dir: Path, pack: DomainPack) -> IntentTaxonomy | None:
    """Return the mined taxonomy, or ``None`` before Module 3 has run for this pack."""
    path = domains_dir / pack.domain / pack.taxonomy_ref
    if not path.is_file():
        return None
    taxonomy = IntentTaxonomy.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if taxonomy.domain != pack.domain:
        raise ValueError(
            f"taxonomy at {path} is for domain {taxonomy.domain!r}, not {pack.domain!r}"
        )
    _check_tools_resolve(pack, taxonomy)
    _check_slots_cover_tools(pack, taxonomy)
    return taxonomy


def _check_tools_resolve(pack: DomainPack, taxonomy: IntentTaxonomy) -> None:
    """An intent that needs an undeclared tool would fail mid-call; fail at load instead."""
    declared = set(pack.tools_by_name)
    for node in taxonomy.nodes:
        missing = set(node.required_tools) - declared
        if missing:
            raise ValueError(
                f"intent {node.intent_id!r} requires tools {sorted(missing)} "
                f"that pack {pack.domain!r} does not declare"
            )


def _check_slots_cover_tools(pack: DomainPack, taxonomy: IntentTaxonomy) -> None:
    """Every argument a tool requires must be collectable from a declared slot.

    Without this the graph asks for what the taxonomy lists, calls the tool, and gets a
    schema violation -- a failure that looks like a backend problem and is actually a
    mis-specified intent. Caught at load, it is one line of YAML.
    """
    specs = pack.tools_by_name
    for node in taxonomy.nodes:
        slot_names = {slot.name for slot in node.slots}
        for tool_name in node.required_tools:
            required = specs[tool_name].input_schema.get("required")
            if not isinstance(required, list):
                continue
            uncovered = sorted({str(a) for a in required} - slot_names)
            if uncovered:
                raise ValueError(
                    f"intent {node.intent_id!r} calls {tool_name!r}, which requires "
                    f"{uncovered}, but declares no slot for them -- the graph would "
                    "collect nothing and the call would fail on a schema violation"
                )


def load_domain(domains_dir: Path, domain: str) -> LoadedDomain:
    root = domains_dir / domain
    pack = load_pack(domains_dir, domain)
    return LoadedDomain(pack=pack, root=root, taxonomy=load_taxonomy(domains_dir, pack))


@lru_cache(maxsize=8)
def _cached(domains_dir: str, domain: str) -> LoadedDomain:
    return load_domain(Path(domains_dir), domain)


def load_domain_cached(domains_dir: Path, domain: str) -> LoadedDomain:
    """Process-lifetime cache. Packs are immutable at runtime; reloading is a restart."""
    return _cached(str(domains_dir), domain)
