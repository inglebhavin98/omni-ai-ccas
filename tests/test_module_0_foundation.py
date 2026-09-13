"""Module gate: Phase 1 foundation (contracts, config, LLM bindings, observability).

One file per module, asserting the module is *usable as a whole* rather than testing
any single unit -- the granular tests live in ``tests/unit/`` mirroring ``src/ccas/``.
A module is not complete until its gate file is green.

Gate files, in order:
    tests/test_module_0_foundation.py   schemas + config + llm + observability   [green]
    tests/test_module_1_ingestion.py    adapters -> CallLog                      (phase 2)
    tests/test_module_2_redaction.py    regex + presidio + leak detector         (phase 2)
    tests/test_module_3_mining.py       embed -> cluster -> taxonomy             (phase 3)
    tests/test_module_4_orchestration.py  StateGraph + tools + policies          (phase 4)
    tests/test_module_5_voice.py        LiveKit + VAD + STT/TTS + barge-in       (phase 5)
    tests/test_module_6_copilot_evals.py  handoff, summary, judge, parity        (phase 6)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

CONTRACTS = [
    "CallLog",
    "IntentTaxonomy",
    "SessionState",
    "ToolPayload",
    "ToolResult",
    "HandoffContext",
    "DomainPack",
    "LLMRequest",
    "EvalReport",
]


@pytest.mark.parametrize("name", CONTRACTS)
def test_every_core_contract_is_exported(name: str) -> None:
    schemas = importlib.import_module("ccas.schemas")
    assert hasattr(schemas, name)
    assert name in schemas.__all__


def test_the_contract_layer_has_no_upward_dependencies() -> None:
    """`schemas` is imported by everything, so a dependency here would be a cycle."""
    package = REPO / "src" / "ccas" / "schemas"
    banned = (
        "ccas.config",
        "ccas.llm",
        "ccas.ingestion",
        "ccas.redaction",
        "ccas.mining",
        "ccas.orchestration",
        "ccas.voice",
        "ccas.copilot",
        "ccas.evals",
        "ccas.observability",
    )
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for module in banned:
            assert f"import {module}" not in source, f"{path.name} imports {module}"


def test_the_configured_packs_load() -> None:
    from ccas.config.domain_loader import available_domains, load_domain

    domains = available_domains(REPO / "domains")
    assert domains, "no domain pack is installed"
    for domain in domains:
        assert load_domain(REPO / "domains", domain).pack.domain == domain


def test_the_latency_budget_is_loadable_and_tight() -> None:
    from ccas.config.budget import load_budget

    budget = load_budget(REPO / "configs" / "latency_budget.yaml")
    assert budget.budget_ms == 800
    assert budget.stages.total_ms > 0


def test_every_node_has_at_least_two_distinct_models() -> None:
    """Rule 6: a node bound to one model can never be parity-tested."""
    from ccas.llm.bindings import MIN_VARIANTS, load_bindings

    registry = load_bindings(REPO / "configs" / "models.yaml")
    assert registry.nodes
    for node in registry.nodes:
        variants = registry.variants(node)
        assert len(variants) >= MIN_VARIANTS
        models = {registry.resolve(node, v).model for v in variants}
        assert len(models) >= MIN_VARIANTS, node


def test_the_provider_abstraction_admits_both_implementations() -> None:
    from ccas.llm.anthropic_provider import AnthropicProvider
    from ccas.llm.base import LLMProvider
    from ccas.llm.vllm_provider import VLLMProvider

    assert issubclass(AnthropicProvider, LLMProvider)
    assert issubclass(VLLMProvider, LLMProvider)


def test_observability_is_importable_without_side_effects() -> None:
    """Importing a tracer must not configure a global provider."""
    importlib.import_module("ccas.observability.tracing")
    ctx = importlib.import_module("ccas.observability.tracing").current_trace_context()
    assert len(ctx.trace_id) == 32


def test_the_dataset_roles_are_declared_for_every_source() -> None:
    from ccas.ingestion.datasets import DATASET_ROLES
    from ccas.schemas.call_log import DatasetSource

    assert set(DATASET_ROLES) == set(DatasetSource)


def test_the_demo_console_runs_against_the_shipped_packs() -> None:
    from cli.demo import main

    assert main(["--quiet-log", "pipeline", "hello", "--domain", "retail"]) == 0
