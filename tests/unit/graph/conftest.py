from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccas.config.domain_loader import LoadedDomain, load_pack
from ccas.graph.context import GraphContext, build_context
from ccas.graph.responder import Responder
from ccas.graph.router import IntentRouter
from ccas.llm.bindings import load_bindings
from ccas.schemas.taxonomy import IntentTaxonomy
from tests.graph_stub import StubGraphProvider

REPO = Path(__file__).resolve().parents[3]
FIXTURE_TAXONOMY = REPO / "tests" / "fixtures" / "taxonomies" / "retail.json"


@pytest.fixture(scope="session")
def taxonomy() -> IntentTaxonomy:
    return IntentTaxonomy.model_validate(json.loads(FIXTURE_TAXONOMY.read_text()))


@pytest.fixture
def provider() -> StubGraphProvider:
    return StubGraphProvider()


@pytest.fixture
def loaded(taxonomy: IntentTaxonomy) -> LoadedDomain:
    pack = load_pack(REPO / "domains", "retail")
    return LoadedDomain(pack=pack, root=REPO / "domains" / "retail", taxonomy=taxonomy)


@pytest.fixture
def ctx(loaded: LoadedDomain, provider: StubGraphProvider) -> GraphContext:
    return build_context(
        loaded,
        provider,
        load_bindings(REPO / "configs" / "models.yaml"),
        config_dir=REPO / "configs",
    )


@pytest.fixture
def router(
    ctx: GraphContext, provider: StubGraphProvider, taxonomy: IntentTaxonomy
) -> IntentRouter:
    return IntentRouter(provider, ctx.bindings.resolve("router"), taxonomy)


@pytest.fixture
def responder(ctx: GraphContext, provider: StubGraphProvider) -> Responder:
    return Responder(provider, ctx.bindings.resolve("task_agent"))
