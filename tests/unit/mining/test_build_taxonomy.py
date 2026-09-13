from __future__ import annotations

from pathlib import Path

import pytest

from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.datasets import DatasetRoleError
from ccas.ingestion.normalizer import Normalizer
from ccas.mining.build_taxonomy import TaxonomyBuilder, collect_utterances
from ccas.mining.cluster import HdbscanClusterer
from ccas.mining.embedder import HashingEmbedder
from ccas.mining.labeler import IntentLabeler
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.redaction.policy import load_pattern_set, load_policy
from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH, RegexEngine
from ccas.schemas.call_log import CallLog, DatasetSource
from ccas.schemas.llm import ModelBinding, ProviderName
from ccas.schemas.taxonomy import IntentTaxonomy, QuadrantThresholds
from tests.mining_stub import StubLabelerProvider

REPO = Path(__file__).resolve().parents[3]
BINDING = ModelBinding(
    node="taxonomy_labeler", provider=ProviderName.VLLM, model="stub", stream=False
)


def _normalizer() -> Normalizer:
    policy = load_policy(REPO / "configs" / "redaction_policy.yaml")
    relaxed = policy.model_copy(
        update={
            "engines": tuple(
                e.model_copy(update={"enabled": False}) if e.name == "presidio_onnx" else e
                for e in policy.engines
            )
        }
    )
    patterns = load_pattern_set(GENERIC_PATTERNS_PATH)
    return Normalizer(
        RedactionPipeline(relaxed, RegexEngine(patterns), presidio=None, mode=RedactionMode.BATCH)
    )


@pytest.fixture(scope="module")
def logs() -> list[CallLog]:
    outcomes = _normalizer().normalize_many(SyntheticAdapter(count=180, seed=17).read())
    return [o.call_log for o in outcomes if o.call_log]


def builder(**kw: object) -> TaxonomyBuilder:
    defaults: dict[str, object] = {
        "embedder": HashingEmbedder(),
        "clusterer": HdbscanClusterer(min_cluster_size=8),
        "labeler": IntentLabeler(StubLabelerProvider(), BINDING, tool_names=["get_order_status"]),
        "thresholds": QuadrantThresholds(),
    }
    defaults.update(kw)
    return TaxonomyBuilder(**defaults)  # type: ignore[arg-type]


def test_only_caller_turns_are_mined(logs: list[CallLog]) -> None:
    """Agent lines describe the resolution, not the intent."""
    from ccas.schemas.common import Speaker

    mined = collect_utterances(logs)
    by_call = {log.call_id: log for log in logs}
    for utterance in mined:
        source = by_call[utterance.call_id].utterances[utterance.index]
        assert source.speaker is Speaker.CALLER


def test_very_short_turns_are_excluded(logs: list[CallLog]) -> None:
    """'yes' and 'okay' would otherwise form the largest cluster in any real corpus."""
    assert all(len(u.text.strip()) >= 8 for u in collect_utterances(logs))


def test_the_length_floor_is_raisable(logs: list[CallLog]) -> None:
    """A pause-segmented corpus needs a higher floor: its segments are sub-utterance.

    See ADR-0013 -- AIxBlock splits on silence, so one spoken sentence arrives as several
    fragments and the default eight-character floor admits all of them.
    """
    strict = collect_utterances(logs, min_chars=50)
    assert all(len(u.text.strip()) >= 50 for u in strict)
    assert len(strict) < len(collect_utterances(logs))


async def test_the_builders_length_floor_reaches_collection(logs: list[CallLog]) -> None:
    """The floor is a builder setting, not just an argument to a free function."""
    built = builder(min_chars=10_000)
    with pytest.raises(ValueError, match="no caller utterances"):
        await built.build(logs, domain="retail")


async def test_a_taxonomy_is_built_and_validates(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    assert isinstance(taxonomy, IntentTaxonomy)
    assert taxonomy.domain == "retail"
    assert taxonomy.nodes


async def test_the_hierarchy_is_well_formed(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    by_id = taxonomy.by_id
    for node in taxonomy.nodes:
        assert node.intent_id.count(".") + 1 == node.level
        if node.parent_id is not None:
            assert node.parent_id in by_id
            assert by_id[node.parent_id].level == node.level - 1


async def test_every_leaf_has_a_full_path(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    for leaf in taxonomy.leaves():
        path = taxonomy.path(leaf.intent_id)
        assert [n.level for n in path] == [1, 2, 3]


async def test_parent_volumes_are_the_sum_of_their_children(logs: list[CallLog]) -> None:
    """An independently guessed parent volume would make the 2x2 meaningless."""
    taxonomy = await builder().build(logs, domain="retail")
    for node in taxonomy.nodes:
        children = taxonomy.children(node.intent_id)
        if not children:
            continue
        assert node.volume.utterance_count == sum(c.volume.utterance_count for c in children)


async def test_conversational_filler_is_excluded(logs: list[CallLog]) -> None:
    """The closing line is the biggest cluster in a real corpus and is not an intent.

    Left in, it would be the taxonomy's highest-volume node and would dominate the
    migration sequencing that the 2x2 exists to drive.
    """
    taxonomy = await builder().build(logs, domain="retail")
    params = taxonomy.clusterer_params
    assert params["clusters_labelled_filler"] >= 1
    assert isinstance(params["filler_utterances"], int)
    assert params["filler_utterances"] > 0

    mined = len(collect_utterances(logs))
    counted = sum(n.volume.utterance_count for n in taxonomy.nodes if n.level == 1)
    assert counted + params["filler_utterances"] <= mined


async def test_share_is_measured_against_intent_bearing_utterances(
    logs: list[CallLog],
) -> None:
    """Filler is not signal we failed to cluster -- it is signal we declined to name it."""
    taxonomy = await builder().build(logs, domain="retail")
    l1_share = sum(n.volume.share_of_total for n in taxonomy.nodes if n.level == 1)
    assert 0.0 < l1_share <= 1.0 + 1e-6


async def test_only_declared_tools_survive(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    for node in taxonomy.nodes:
        assert set(node.required_tools) <= {"get_order_status"}


async def test_coverage_and_noise_sum_to_one(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    assert taxonomy.coverage + taxonomy.noise_ratio == pytest.approx(1.0, abs=1e-6)


async def test_provenance_is_recorded(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    assert taxonomy.embedding_model == "hashing-bow"
    assert taxonomy.labeler_model == "vllm:stub"
    assert taxonomy.clusterer == "hdbscan"
    assert taxonomy.clusterer_params["min_cluster_size"] == 8
    assert len(taxonomy.source_call_ids) == len(logs)


async def test_the_taxonomy_round_trips(logs: list[CallLog]) -> None:
    taxonomy = await builder().build(logs, domain="retail")
    assert IntentTaxonomy.model_validate_json(taxonomy.model_dump_json()) == taxonomy


async def test_the_taxonomy_id_is_deterministic(logs: list[CallLog]) -> None:
    first = await builder().build(logs, domain="retail")
    second = await builder().build(logs, domain="retail")
    assert first.taxonomy_id == second.taxonomy_id


async def test_mining_a_supervision_corpus_is_refused(logs: list[CallLog]) -> None:
    """ADR-0004: clustering a labelled set rediscovers its own labels."""
    relabelled = [log.model_copy(update={"source": DatasetSource.BITEXT}) for log in logs[:20]]
    with pytest.raises(DatasetRoleError, match="not a valid corpus for 'intent_mining'"):
        await builder().build(relabelled, domain="retail")


async def test_an_unmineable_corpus_says_why() -> None:
    empty = _normalizer().normalize(next(iter(SyntheticAdapter(count=1, seed=1).read()))).call_log
    assert empty is not None
    with pytest.raises(ValueError, match="found no structure"):
        await builder(clusterer=HdbscanClusterer(min_cluster_size=50)).build(
            [empty], domain="retail"
        )


async def test_the_taxonomy_persists_the_space_its_centroids_live_in(
    logs: list[CallLog],
) -> None:
    """A centroid without the mean that defines its space cannot be compared against.

    Clustering happens in centred space (ADR-0017). If the stored centroid is computed
    from raw vectors, membership and description disagree, and the transform is gone --
    a live utterance cannot centre itself, because the mean of one vector is itself.
    """
    taxonomy = await builder().build(logs, domain="retail")
    assert taxonomy.clusterer_params["centre"] is True
    assert taxonomy.embedding_mean, "the corpus mean must ship with the taxonomy"
    assert len(taxonomy.embedding_mean) == HashingEmbedder().dimensions


async def test_an_uncentred_taxonomy_stores_no_mean(logs: list[CallLog]) -> None:
    """Nothing was transformed, so there is nothing to record or undo."""
    taxonomy = await builder(clusterer=HdbscanClusterer(min_cluster_size=8, centre=False)).build(
        logs, domain="retail"
    )
    assert taxonomy.clusterer_params["centre"] is False
    assert taxonomy.embedding_mean == ()
