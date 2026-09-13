"""Module gate: M3 — intent mining and taxonomy.

Granular tests live in ``tests/unit/mining/``. This asserts the chain works as a whole:
ingested logs in, a contract-valid ``IntentTaxonomy`` out, loadable by a domain pack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack, load_taxonomy
from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.datasets import DatasetRole, DatasetRoleError, require_role
from ccas.ingestion.normalizer import Normalizer
from ccas.mining.build_taxonomy import TaxonomyBuilder
from ccas.mining.cluster import HdbscanClusterer
from ccas.mining.embedder import HashingEmbedder
from ccas.mining.labeler import IntentLabeler
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.redaction.policy import load_policy
from ccas.redaction.regex_engine import RegexEngine
from ccas.schemas.call_log import CallLog, DatasetSource
from ccas.schemas.llm import ModelBinding, ProviderName
from ccas.schemas.taxonomy import IntentTaxonomy, Quadrant, TaxonomyProvenance
from tests.mining_stub import StubLabelerProvider

REPO = Path(__file__).resolve().parents[1]
BINDING = ModelBinding(
    node="taxonomy_labeler", provider=ProviderName.VLLM, model="stub", stream=False
)


@pytest.fixture(scope="module")
def logs() -> list[CallLog]:
    policy = load_policy(REPO / "configs" / "redaction_policy.yaml")
    relaxed = policy.model_copy(
        update={
            "engines": tuple(
                e.model_copy(update={"enabled": False}) if e.name == "presidio_onnx" else e
                for e in policy.engines
            )
        }
    )
    normalizer = Normalizer(
        RedactionPipeline(relaxed, RegexEngine.from_path(), presidio=None, mode=RedactionMode.BATCH)
    )
    return [
        o.call_log
        for o in normalizer.normalize_many(SyntheticAdapter(count=200, seed=31).read())
        if o.call_log
    ]


@pytest.fixture(scope="module")
async def taxonomy(logs: list[CallLog]) -> IntentTaxonomy:
    pack = load_pack(REPO / "domains", "retail")
    builder = TaxonomyBuilder(
        embedder=HashingEmbedder(),
        clusterer=HdbscanClusterer(min_cluster_size=10),
        labeler=IntentLabeler(StubLabelerProvider(), BINDING, tool_names=list(pack.tools_by_name)),
        thresholds=pack.quadrant_thresholds,
    )
    return await builder.build(logs, domain="retail", version="0.1.0")


def test_mining_is_gated_on_corpus_role() -> None:
    """The corpus separation is the difference between a real taxonomy and a tautology."""
    require_role(DatasetSource.AIXBLOCK, DatasetRole.INTENT_MINING)
    for supervision in (DatasetSource.BITEXT, DatasetSource.NATCS):
        with pytest.raises(DatasetRoleError):
            require_role(supervision, DatasetRole.INTENT_MINING)


def test_the_pipeline_produces_a_usable_taxonomy(taxonomy: IntentTaxonomy) -> None:
    assert taxonomy.nodes
    assert taxonomy.leaves()
    assert 0.0 <= taxonomy.coverage <= 1.0


def test_every_node_carries_a_quadrant(taxonomy: IntentTaxonomy) -> None:
    """The 2x2 is what sequences the migration; a node without one cannot be scheduled."""
    for node in taxonomy.nodes:
        assert isinstance(node.automation.quadrant, Quadrant)


def test_regulated_intents_auto_escalate(taxonomy: IntentTaxonomy) -> None:
    """CLAUDE.md Rule 4: a bot must never attempt a regulated intent."""
    from ccas.schemas.common import RiskTier

    for node in taxonomy.nodes:
        if node.risk_tier is RiskTier.REGULATED:
            assert node.escalation.auto_escalate


def test_the_taxonomy_loads_through_the_pack_loader(
    taxonomy: IntentTaxonomy, tmp_path: Path
) -> None:
    """The handshake with M4: the orchestrator builds its graph from this file."""
    pack = load_pack(REPO / "domains", "retail")
    root = tmp_path / "retail"
    root.mkdir()
    (root / "taxonomy.json").write_text(taxonomy.model_dump_json(), encoding="utf-8")

    loaded = load_taxonomy(tmp_path, pack)
    assert loaded is not None
    assert loaded.domain == "retail"
    assert len(loaded.nodes) == len(taxonomy.nodes)


def test_required_tools_resolve_against_the_pack(taxonomy: IntentTaxonomy) -> None:
    """An intent needing an undeclared tool would fail mid-call; it must fail at load."""
    declared = set(load_pack(REPO / "domains", "retail").tools_by_name)
    for node in taxonomy.nodes:
        assert set(node.required_tools) <= declared


def test_mined_slots_cover_the_arguments_of_the_tools_they_name(
    taxonomy: IntentTaxonomy,
) -> None:
    """Otherwise the graph collects nothing and the call dies on a schema violation.

    The labeler is told each tool's required arguments precisely so this holds; the
    loader rejects a taxonomy where it does not.
    """
    specs = load_pack(REPO / "domains", "retail").tools_by_name
    for node in taxonomy.nodes:
        slot_names = {slot.name for slot in node.slots}
        for tool_name in node.required_tools:
            required = specs[tool_name].input_schema.get("required") or []
            assert {str(a) for a in required} <= slot_names, f"{node.intent_id} -> {tool_name}"


def test_the_taxonomy_carries_no_unredacted_content(taxonomy: IntentTaxonomy) -> None:
    """Exemplars and labels derive from call text; none of it may be raw."""
    import re

    serialized = json.dumps(json.loads(taxonomy.model_dump_json()))
    for pattern in (r"\b4\d{3} \d{4} \d{4} \d{4}\b", r"[a-z]+@example\.com", r"ORD-\d{6}"):
        assert not re.search(pattern, serialized)


def test_provenance_is_complete(taxonomy: IntentTaxonomy) -> None:
    """A taxonomy nobody can reproduce is not an asset."""
    assert taxonomy.embedding_model
    assert taxonomy.labeler_model
    assert taxonomy.clusterer_params
    assert taxonomy.source_call_ids


def test_no_committed_taxonomy_was_mined_by_a_stub() -> None:
    """Any taxonomy on disk must be reproducible from a real corpus and a real model.

    Replaces the Phase 3 guard that simply forbade committing one. Now that a pack ships
    a mined taxonomy the question is no longer *whether* one exists but whether it can be
    traced: a taxonomy mined by the hashing embedder or the stub labeler would look
    exactly as authoritative as a real one, and describe nothing.
    """
    committed = sorted((REPO / "domains").glob("*/taxonomy.json"))
    for path in committed:
        taxonomy = IntentTaxonomy.model_validate(json.loads(path.read_text()))
        name = path.parent.name
        if taxonomy.provenance is TaxonomyProvenance.ADOPTED:
            # Nothing was clustered, so there is no embedder or labeller to vouch for.
            # What must be traceable is the corpus and which half of it was used, because
            # an evaluator has to know what to hold out (ADR-0020).
            assert taxonomy.adopted_from, f"{name}: adopted from nowhere"
            assert taxonomy.derivation_split, f"{name}: no derivation split recorded"
            continue
        assert "hashing" not in taxonomy.embedding_model.lower(), (
            f"{name}: mined from lexical overlap, not semantics "
            f"({taxonomy.embedding_model}). --allow-hashing-embedder is for throwaways"
        )
        assert "stub" not in taxonomy.labeler_model.lower(), (
            f"{name}: labelled by a stub ({taxonomy.labeler_model})"
        )
        assert taxonomy.source_call_ids, f"{name}: no source calls recorded"
        assert taxonomy.clusterer_params, f"{name}: no clusterer params recorded"


def test_a_pack_that_declares_a_taxonomy_has_one() -> None:
    """`taxonomy_ref` pointing at a missing file fails at the first caller, not at load."""
    domains = REPO / "domains"
    for pack_yaml in sorted(domains.glob("*/pack.yaml")):
        pack = load_pack(domains, pack_yaml.parent.name)
        if not (domains / pack.domain / pack.taxonomy_ref).is_file():
            continue
        loaded = load_taxonomy(domains, pack)
        assert loaded is not None
        assert loaded.domain == pack.domain
