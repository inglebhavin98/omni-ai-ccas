from __future__ import annotations

import pytest

from ccas.ingestion.datasets import (
    DATASET_ROLES,
    DatasetRole,
    DatasetRoleError,
    raw_dir_name,
    require_role,
    roles_for,
    sources_for,
)
from ccas.schemas.call_log import DatasetSource

GROUND_TRUTH = (DatasetSource.AIXBLOCK, DatasetSource.BITEXT, DatasetSource.NATCS)


def test_every_source_has_a_declared_role_set() -> None:
    assert set(DATASET_ROLES) == set(DatasetSource)


def test_aixblock_is_the_mining_corpus() -> None:
    roles = roles_for(DatasetSource.AIXBLOCK)
    assert DatasetRole.INTENT_MINING in roles
    assert DatasetRole.TRANSCRIPT_INGESTION in roles
    assert DatasetRole.REDACTION_CORPUS in roles


def test_bitext_is_supervision_only() -> None:
    """Mining a taxonomy from labelled data rediscovers its own labels."""
    roles = roles_for(DatasetSource.BITEXT)
    assert DatasetRole.NLU_GROUND_TRUTH in roles
    assert DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH in roles
    assert DatasetRole.JUDGE_EVALUATION in roles
    assert DatasetRole.INTENT_MINING not in roles


def test_natcs_is_natural_language_supervision() -> None:
    """Measured, not assumed (ADR-0015).

    NatCS publishes 4,205 labelled turns over 82 intents -- so it *is* supervision, which
    ADR-0004 had wrong. It is not a trajectory: only 8 of 1,088 multi-turn dialogues have
    consecutive turn ids, because the release publishes labelled turns rather than whole
    conversations. It therefore cannot benchmark what ADR-0004 assigned it.
    """
    roles = roles_for(DatasetSource.NATCS)
    assert DatasetRole.NLU_GROUND_TRUTH in roles
    assert DatasetRole.TRANSCRIPT_INGESTION in roles
    # Labelled, so mining it would rediscover its own labels -- the Bitext reasoning.
    assert DatasetRole.INTENT_MINING not in roles
    # Sampled turns, not trajectories.
    assert DatasetRole.STATE_GRAPH_BENCHMARK not in roles
    assert DatasetRole.DIALOGUE_CONTEXT_BENCHMARK not in roles
    # No slot annotations anywhere in the release.
    assert DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH not in roles


def test_no_corpus_can_benchmark_state_graph_transitions_yet() -> None:
    """An honest empty set beats a role nothing on disk can support.

    Leaving STATE_GRAPH_BENCHMARK assigned to NatCS would let Phase 6 build a benchmark on
    sampled turns and report a transition-accuracy number that means nothing.
    """
    real = (DatasetSource.AIXBLOCK, DatasetSource.BITEXT, DatasetSource.NATCS)
    assert all(s not in real for s in sources_for(DatasetRole.STATE_GRAPH_BENCHMARK))


def test_mining_and_supervision_never_share_a_corpus() -> None:
    """The firewall that actually matters.

    Replaces a disjoint-primary-roles assertion that ADR-0015 made false: Bitext and NatCS
    are *both* NLU ground truth now, and that overlap is a feature. Bitext is templated
    ({{Order Number}} with generated typos); NatCS is human speech. A router graded only on
    templates scores better than it is. What must never overlap is mining and supervision
    -- clustering a labelled corpus rediscovers its own labels.
    """
    supervision = {
        DatasetRole.NLU_GROUND_TRUTH,
        DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH,
        DatasetRole.JUDGE_EVALUATION,
    }
    for source in GROUND_TRUTH:
        roles = roles_for(source)
        assert not (DatasetRole.INTENT_MINING in roles and roles & supervision), (
            f"{source.value} is admitted for both mining and supervision"
        )


def test_both_supervision_corpora_grade_intent_classification() -> None:
    """Templated and natural phrasing, deliberately kept as two sources (ADR-0015)."""
    graders = set(sources_for(DatasetRole.NLU_GROUND_TRUTH))
    assert {DatasetSource.BITEXT, DatasetSource.NATCS} <= graders
    assert DatasetSource.AIXBLOCK not in graders
    # Only Bitext annotates parameters, so among the real corpora only Bitext can grade
    # extraction. Fixture sources are unrestricted and are not part of this claim.
    extraction = set(sources_for(DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH)) & set(GROUND_TRUTH)
    assert extraction == {DatasetSource.BITEXT}


@pytest.mark.parametrize(
    ("source", "role"),
    [
        (DatasetSource.BITEXT, DatasetRole.INTENT_MINING),
        (DatasetSource.NATCS, DatasetRole.INTENT_MINING),
        (DatasetSource.NATCS, DatasetRole.STATE_GRAPH_BENCHMARK),
        (DatasetSource.NATCS, DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH),
        (DatasetSource.AIXBLOCK, DatasetRole.NLU_GROUND_TRUTH),
        (DatasetSource.BITEXT, DatasetRole.DIALOGUE_CONTEXT_BENCHMARK),
    ],
)
def test_misuse_is_refused_at_the_point_of_the_mistake(
    source: DatasetSource, role: DatasetRole
) -> None:
    with pytest.raises(DatasetRoleError, match="is not a valid corpus for"):
        require_role(source, role)


def test_the_refusal_names_the_corpora_that_would_work() -> None:
    with pytest.raises(DatasetRoleError, match="aixblock"):
        require_role(DatasetSource.BITEXT, DatasetRole.INTENT_MINING)


@pytest.mark.parametrize("source", GROUND_TRUTH)
def test_permitted_roles_pass_silently(source: DatasetSource) -> None:
    for role in roles_for(source):
        require_role(source, role)


def test_fixture_sources_are_unrestricted() -> None:
    for source in (DatasetSource.SYNTHETIC, DatasetSource.GENERIC_CSV):
        assert roles_for(source) == frozenset(DatasetRole)


def test_sources_for_inverts_the_mapping() -> None:
    assert DatasetSource.BITEXT in sources_for(DatasetRole.JUDGE_EVALUATION)
    assert DatasetSource.AIXBLOCK not in sources_for(DatasetRole.JUDGE_EVALUATION)


@pytest.mark.parametrize("source", GROUND_TRUTH)
def test_raw_dir_matches_the_documented_layout(source: DatasetSource) -> None:
    assert raw_dir_name(source) == source.value
