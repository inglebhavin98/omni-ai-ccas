"""Dataset role registry.

The three ground-truth corpora answer different questions and must not be pooled into
one undifferentiated training set. Mixing them silently is easy and the damage is
invisible until an eval number is quietly meaningless -- so the separation is enforced
here in code rather than described in a README.

Each corpus is admitted only for the roles it can actually support:

``aixblock``
    92k call-centre scripts. Unlabelled, high volume, conversational. The only corpus
    large and unbiased enough to mine a taxonomy from, and the right target for
    redaction throughput work.

``bitext``
    Labelled intent/category pairs with slot annotations. Small, clean, single-turn.
    This is supervision -- it grades the router and the judge. Mining a taxonomy from
    it would rediscover its own label set and prove nothing.

``natcs``
    4,205 human-spoken turns labelled with 82 intents across insurance, banking and
    finance. Supervision, like Bitext -- but *natural* phrasing where Bitext is templated,
    which is why both grade the router rather than one replacing the other.

    It is not a trajectory. The release publishes labelled turns, not whole conversations:
    only 8 of its 1,088 multi-turn dialogues have consecutive turn ids. It therefore
    cannot benchmark context retention or state-graph transitions, whatever its
    description suggests. Measured, not assumed -- see ADR-0015.
"""

from __future__ import annotations

from enum import StrEnum

from ccas.schemas.call_log import DatasetSource

__all__ = [
    "DATASET_ROLES",
    "DatasetRole",
    "DatasetRoleError",
    "raw_dir_name",
    "require_role",
    "roles_for",
    "sources_for",
]


class DatasetRole(StrEnum):
    """What a corpus is permitted to be used *for*."""

    TRANSCRIPT_INGESTION = "transcript_ingestion"
    """Raw multi-turn text normalised into ``CallLog``."""

    REDACTION_CORPUS = "redaction_corpus"
    """Volume input for PII redaction development and throughput measurement."""

    INTENT_MINING = "intent_mining"
    """Unsupervised embedding + HDBSCAN clustering into an ``IntentTaxonomy``."""

    NLU_GROUND_TRUTH = "nlu_ground_truth"
    """Labelled intents that grade the router. Never an input to mining."""

    SLOT_EXTRACTION_GROUND_TRUTH = "slot_extraction_ground_truth"
    """Labelled parameters that grade tool-argument extraction."""

    JUDGE_EVALUATION = "judge_evaluation"
    """Offline LLM-as-a-Judge scoring against known-correct answers."""

    DIALOGUE_CONTEXT_BENCHMARK = "dialogue_context_benchmark"
    """Multi-turn context retention across a conversation.

    No corpus on disk supports this today (ADR-0015). The role stays declared so the
    requirement stays visible and a future corpus can be admitted for it.
    """

    STATE_GRAPH_BENCHMARK = "state_graph_benchmark"
    """Node-to-node transition correctness over a real dialogue trajectory.

    Also unsourced. Replaying a trajectory needs consecutive turns from both speakers,
    which no available release publishes.
    """


#: The authoritative mapping. Adding a role to a corpus is a documented decision, not a
#: convenience -- see docs/adr/0004-dataset-role-separation.md.
DATASET_ROLES: dict[DatasetSource, frozenset[DatasetRole]] = {
    DatasetSource.AIXBLOCK: frozenset(
        {
            DatasetRole.TRANSCRIPT_INGESTION,
            DatasetRole.REDACTION_CORPUS,
            DatasetRole.INTENT_MINING,
        }
    ),
    DatasetSource.BITEXT: frozenset(
        {
            DatasetRole.NLU_GROUND_TRUTH,
            DatasetRole.SLOT_EXTRACTION_GROUND_TRUTH,
            DatasetRole.JUDGE_EVALUATION,
        }
    ),
    DatasetSource.NATCS: frozenset(
        {
            DatasetRole.TRANSCRIPT_INGESTION,
            DatasetRole.NLU_GROUND_TRUTH,
            DatasetRole.JUDGE_EVALUATION,
        }
    ),
    # Fixtures and live capture are unrestricted; they carry no corpus-level bias.
    DatasetSource.SYNTHETIC: frozenset(DatasetRole),
    DatasetSource.GENERIC_CSV: frozenset(DatasetRole),
    DatasetSource.LIVE_CAPTURE: frozenset(DatasetRole),
}


class DatasetRoleError(ValueError):
    """A corpus was used for something it cannot legitimately support."""


def roles_for(source: DatasetSource) -> frozenset[DatasetRole]:
    return DATASET_ROLES[source]


def sources_for(role: DatasetRole) -> tuple[DatasetSource, ...]:
    return tuple(sorted((s for s, r in DATASET_ROLES.items() if role in r), key=str))


def require_role(source: DatasetSource, role: DatasetRole) -> None:
    """Gate a pipeline stage on corpus suitability.

    Call this at the top of any stage that consumes a corpus, so a misuse fails at the
    point of the mistake rather than silently skewing a downstream metric.
    """
    permitted = DATASET_ROLES[source]
    if role not in permitted:
        allowed = ", ".join(sorted(sources_for(role))) or "none"
        raise DatasetRoleError(
            f"{source.value!r} is not a valid corpus for {role.value!r}; "
            f"permitted roles for {source.value!r} are "
            f"{sorted(r.value for r in permitted)}. "
            f"Corpora admitted for {role.value!r}: {allowed}."
        )


def raw_dir_name(source: DatasetSource) -> str:
    """Directory under ``data/raw/`` holding this corpus."""
    return source.value
