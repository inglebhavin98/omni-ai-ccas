from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ccas.llm.bindings import MIN_VARIANTS, UnknownNodeError, load_bindings
from ccas.schemas.llm import ProviderName, StructuredMode

REPO = Path(__file__).resolve().parents[3]
MODELS_FILE = REPO / "configs" / "models.yaml"


def test_shipped_bindings_load() -> None:
    registry = load_bindings(MODELS_FILE)
    assert registry.default_provider is ProviderName.OPENROUTER
    assert "router" in registry.nodes


def test_every_node_can_be_parity_tested() -> None:
    """Rule 6's intent: no node may silently depend on a single model."""
    registry = load_bindings(MODELS_FILE)
    for node in registry.nodes:
        assert len(registry.variants(node)) >= MIN_VARIANTS
        first, second = registry.pair(node)
        assert first.model != second.model, node


def test_parity_may_be_between_two_models_on_one_provider() -> None:
    """With one gateway in front of many models, that is the useful comparison."""
    registry = load_bindings(MODELS_FILE)
    first, second = registry.pair("router")
    assert first.provider is second.provider is ProviderName.OPENROUTER
    assert first.model != second.model


def test_a_single_variant_node_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    raw["bindings"]["router"] = {"openrouter": raw["bindings"]["router"]["openrouter"]}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2 are required"):
        load_bindings(path)


def test_two_variants_naming_the_same_model_are_rejected(tmp_path: Path) -> None:
    """Comparing a model against itself measures nothing."""
    raw = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    router = raw["bindings"]["router"]
    model = router["openrouter"]["model"]
    router["openrouter_alt"]["model"] = model
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct model"):
        load_bindings(path)


def test_the_router_is_the_most_latency_constrained_node() -> None:
    registry = load_bindings(MODELS_FILE)
    router = registry.resolve("router")
    task = registry.resolve("task_agent")
    assert router.latency_budget_ms is not None
    assert task.latency_budget_ms is not None
    assert router.latency_budget_ms <= task.latency_budget_ms


def test_the_router_does_not_reason_inline() -> None:
    """Measured: a model that reasons before answering cannot meet a 90ms budget, and
    truncates before emitting JSON when the allowance is small."""
    for binding in load_bindings(MODELS_FILE).pair("router"):
        assert not binding.reasoning


def test_the_primary_router_binding_uses_native_structured_output() -> None:
    """The server constrains decoding, so the model cannot prepend prose."""
    assert load_bindings(MODELS_FILE).resolve("router").structured_mode is (
        StructuredMode.RESPONSE_FORMAT
    )


def test_offline_nodes_do_not_stream() -> None:
    registry = load_bindings(MODELS_FILE)
    for node in ("judge", "taxonomy_labeler", "summarizer"):
        assert not registry.resolve(node).stream


def test_structured_mode_is_declared_for_every_binding() -> None:
    """It is a property of the model, and getting it wrong fails silently."""
    registry = load_bindings(MODELS_FILE)
    for node in registry.nodes:
        for variant in registry.variants(node):
            assert registry.resolve(node, variant).structured_mode in set(StructuredMode)


def test_unknown_node_raises_with_the_declared_list() -> None:
    with pytest.raises(UnknownNodeError, match="declared nodes"):
        load_bindings(MODELS_FILE).resolve("no_such_node")


def test_unknown_variant_raises_with_the_available_list() -> None:
    with pytest.raises(UnknownNodeError, match="available"):
        load_bindings(MODELS_FILE).resolve("router", "no_such_variant")


def test_a_variant_resolves_by_name_or_by_provider() -> None:
    registry = load_bindings(MODELS_FILE)
    assert registry.resolve("router", "openrouter").variant == "openrouter"
    assert registry.resolve("router", ProviderName.OPENROUTER).provider is (ProviderName.OPENROUTER)


def test_the_default_variant_prefers_the_default_provider() -> None:
    registry = load_bindings(MODELS_FILE)
    assert registry.resolve("router").provider is registry.default_provider


def test_models_lists_everything_in_play() -> None:
    models = load_bindings(MODELS_FILE).models()
    assert set(models) == set(load_bindings(MODELS_FILE).nodes)
    assert all(len(v) >= MIN_VARIANTS for v in models.values())


def test_an_empty_binding_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("default_provider: openrouter\nbindings: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no bindings"):
        load_bindings(path)
