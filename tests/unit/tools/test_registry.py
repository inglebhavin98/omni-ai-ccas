from __future__ import annotations

from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.schemas.domain import DomainPack
from ccas.schemas.tools import ToolSpec
from ccas.tools.registry import (
    GlobalToolSet,
    ToolNotRegisteredError,
    ToolRegistry,
    build_registry,
    load_global_tools,
)

REPO = Path(__file__).resolve().parents[3]
DOMAINS = REPO / "domains"
GLOBAL = REPO / "configs" / "tools.yaml"

STRICT = {"type": "object", "properties": {}, "additionalProperties": False}


def spec(name: str, domain: str = "core") -> ToolSpec:
    return ToolSpec(name=name, description="d", input_schema=STRICT, domain=domain)


@pytest.fixture(scope="module")
def retail() -> DomainPack:
    return load_pack(DOMAINS, "retail")


def test_the_shipped_global_set_loads() -> None:
    names = {t.name for t in load_global_tools(GLOBAL)}
    assert {"transfer_to_human", "schedule_callback", "send_link", "verify_identity"} <= names


def test_global_tools_must_be_domain_agnostic() -> None:
    """A vertical's tool in the global set would put its vocabulary in the core."""
    with pytest.raises(ValueError, match="only 'core' belongs in the global set"):
        GlobalToolSet(version="1.0.0", tools=(spec("track_order", domain="retail"),))


def test_a_registry_holds_global_and_pack_tools(retail: DomainPack) -> None:
    registry = build_registry(retail, GLOBAL)
    assert "transfer_to_human" in registry.names
    assert "get_order_status" in registry.names
    assert registry.is_global("transfer_to_human")
    assert not registry.is_global("get_order_status")


def test_a_pack_may_not_shadow_a_global_tool(retail: DomainPack) -> None:
    shadowing = retail.model_copy(
        update={"tools": (*retail.tools, spec("transfer_to_human", domain="retail"))}
    )
    with pytest.raises(ValueError, match="shadows global tool"):
        build_registry(shadowing, GLOBAL)


def test_both_packs_build_a_registry() -> None:
    """Rule 1: the same core serves both, differing only in data."""
    for domain in ("retail", "healthcare"):
        registry = build_registry(load_pack(DOMAINS, domain), GLOBAL)
        assert registry.domain == domain
        assert len(registry.names) > len(registry.global_names)


def test_an_unregistered_tool_names_what_is_available(retail: DomainPack) -> None:
    registry = build_registry(retail, GLOBAL)
    with pytest.raises(ToolNotRegisteredError, match="registered:"):
        registry.resolve("delete_everything")


def test_for_intent_resolves_a_taxonomy_node(retail: DomainPack) -> None:
    from ccas.schemas.taxonomy import AutomationScore, IntentNode, VolumeStats

    node = IntentNode(
        intent_id="billing",
        level=1,
        label="Billing",
        description="d",
        required_tools=("get_order_status", "transfer_to_human"),
        volume=VolumeStats(utterance_count=1, call_count=1, share_of_total=0.1),
        automation=AutomationScore(
            feasibility=0.5, complexity=0.5, confidence=0.5, rationale="r", volume_share=0.1
        ),
    )
    specs = build_registry(retail, GLOBAL).for_intent(node)
    assert [s.name for s in specs] == ["get_order_status", "transfer_to_human"]


def test_unresolved_reports_missing_names(retail: DomainPack) -> None:
    registry = build_registry(retail, GLOBAL)
    assert registry.unresolved(("get_order_status", "nope")) == ("nope",)


def test_a_missing_global_file_is_not_fatal(retail: DomainPack, tmp_path: Path) -> None:
    """A deployment without shared tools still runs on its pack's own."""
    registry = ToolRegistry(load_global_tools(tmp_path / "absent.yaml"), retail)
    assert registry.global_names == ()
    assert "get_order_status" in registry.names


def test_every_registered_tool_has_a_closed_schema(retail: DomainPack) -> None:
    """An open schema lets a model smuggle unvalidated arguments to a backend."""
    for tool in build_registry(retail, GLOBAL).specs:
        assert tool.input_schema["additionalProperties"] is False
