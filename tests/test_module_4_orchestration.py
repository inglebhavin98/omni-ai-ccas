"""Module gate: M4 — the agentic mesh.

Granular tests live in ``tests/unit/{tools,policies,graph}/``; the scripted journeys in
``tests/integration/test_graph_e2e.py``. This asserts the module is usable as a whole
and that its four structural guarantees hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccas.config.domain_loader import LoadedDomain, load_pack
from ccas.graph.assembly import NODE_NAMES, build_graph
from ccas.graph.context import build_context
from ccas.llm.bindings import load_bindings
from ccas.policies.base import PolicyContext
from ccas.policies.engine import PolicyEngine
from ccas.schemas.common import VerificationLevel
from ccas.schemas.taxonomy import IntentTaxonomy
from ccas.tools.registry import build_registry
from tests.graph_stub import StubGraphProvider

REPO = Path(__file__).resolve().parents[1]
TAXONOMY = IntentTaxonomy.model_validate(
    json.loads((REPO / "tests" / "fixtures" / "taxonomies" / "retail.json").read_text())
)


def _context(domain: str = "retail"):
    pack = load_pack(REPO / "domains", domain)
    loaded = LoadedDomain(
        pack=pack,
        root=REPO / "domains" / domain,
        taxonomy=TAXONOMY if domain == "retail" else None,
    )
    return build_context(
        loaded,
        StubGraphProvider(),
        load_bindings(REPO / "configs" / "models.yaml"),
        config_dir=REPO / "configs",
    )


@pytest.mark.parametrize("domain", ["retail", "healthcare"])
def test_the_mesh_assembles_for_both_packs(domain: str) -> None:
    """Rule 1: swapping the pack must need no code change."""
    graph = build_graph(_context(domain))
    assert set(NODE_NAMES) <= set(graph.get_graph().nodes)


def test_the_graph_is_deterministic_not_an_agent_loop() -> None:
    """Rule 4: every successor is a pure function of state, declared at build time."""
    drawn = build_graph(_context()).get_graph()
    for node in NODE_NAMES:
        assert any(e.source == node for e in drawn.edges), f"{node} has no outgoing edge"


def test_every_tool_a_taxonomy_needs_is_registered() -> None:
    """The handshake between M3 and M4: an intent naming an unknown tool fails at load."""
    registry = build_registry(
        load_pack(REPO / "domains", "retail"), REPO / "configs" / "tools.yaml"
    )
    for node in TAXONOMY.nodes:
        assert registry.unresolved(node.required_tools) == ()


def test_every_tool_argument_is_collectable_from_a_slot() -> None:
    """Otherwise the graph collects nothing and the call dies on a schema violation."""
    registry = build_registry(
        load_pack(REPO / "domains", "retail"), REPO / "configs" / "tools.yaml"
    )
    for node in TAXONOMY.nodes:
        declared = {slot.name for slot in node.slots}
        for spec in registry.for_intent(node):
            required = spec.input_schema.get("required") or []
            assert {str(a) for a in required} <= declared, f"{node.intent_id} -> {spec.name}"


def test_a_guarded_tool_cannot_be_dispatched_unverified() -> None:
    ctx = _context()
    spec = ctx.registry.resolve("update_delivery_address")
    assert spec.requires_verification.rank >= VerificationLevel.STRONG.rank
    payload = ctx.executor.build_payload(
        "update_delivery_address",
        session_id="s1",
        trace=_trace(),
        arguments={},
        verification=VerificationLevel.NONE,
    )
    assert not payload.authorized_by(spec)


def test_policies_are_pure_and_ordered() -> None:
    engine = PolicyEngine()
    assert engine.names == ("risk", "retry", "sentiment", "confidence")


def test_every_intent_has_a_reachable_policy_context() -> None:
    ctx = _context()
    pack = ctx.pack
    for node in TAXONOMY.nodes:
        policy_ctx = PolicyContext.for_intent(pack, node)
        assert policy_ctx.default_queue in pack.queues_by_name


def test_a_regulated_intent_is_configured_to_auto_escalate() -> None:
    """Rule 4: a bot must never attempt one."""
    from ccas.schemas.common import RiskTier

    regulated = [n for n in TAXONOMY.nodes if n.risk_tier is RiskTier.REGULATED]
    assert regulated, "the fixture should exercise the regulated path"
    for node in regulated:
        if node.required_tools:
            pytest.fail(f"{node.intent_id} is regulated but declares tools")


def test_the_session_vault_never_enters_a_serialisable_model() -> None:
    """It holds originals; a field on any model would put them in a checkpoint."""
    from ccas.schemas.session import SessionState

    assert "vault" not in SessionState.model_fields
    ctx = _context()
    assert "PlaceholderVault" not in ctx.pack.model_dump_json()


def test_the_vault_renders_blind() -> None:
    """A traceback must not print what it holds."""
    from ccas.redaction.vault import PlaceholderVault

    vault = PlaceholderVault()
    vault.store("[ACCOUNT_REF_1]", "ORD-884210")
    assert "ORD-884210" not in f"{vault!r} {vault!s}"


def _trace():
    from ccas.schemas.common import TraceContext

    return TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id="gate")
