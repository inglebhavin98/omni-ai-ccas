"""``IntentTaxonomy`` -- the L1/L2/L3 hierarchy mined by Module 3.

The taxonomy is data, not code. Module 4 builds its graph from whatever this object
contains, so a new vertical is a new taxonomy file rather than a new branch.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, computed_field, model_validator

from ccas.schemas.common import Frozen, JsonValue, RiskTier, SchemaVersion, Slug, utcnow
from ccas.schemas.pii import PiiEntityType

__all__ = [
    "AutomationScore",
    "CorpusHomogeneity",
    "EscalationPolicy",
    "IntentNode",
    "IntentTaxonomy",
    "Quadrant",
    "QuadrantThresholds",
    "SlotSpec",
    "SlotType",
    "TaxonomyProvenance",
    "VolumeStats",
]


class SlotType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    CURRENCY = "currency"
    IDENTIFIER = "identifier"
    PHONE = "phone"
    EMAIL = "email"
    FREE_TEXT = "free_text"


class SlotSpec(Frozen):
    name: Slug
    slot_type: SlotType
    required: bool = True
    elicitation_prompt: str = Field(min_length=1)
    reprompt: str | None = None
    validation_regex: str | None = None
    enum_values: tuple[str, ...] | None = None
    pii_entity: PiiEntityType | None = None
    """Non-None means this slot's captured value is redacted before any egress."""

    dtmf_capturable: bool = False
    max_attempts: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _check_enum_values(self) -> Self:
        is_enum = self.slot_type is SlotType.ENUM
        if is_enum and not self.enum_values:
            raise ValueError(f"slot {self.name!r}: ENUM requires non-empty enum_values")
        if not is_enum and self.enum_values:
            raise ValueError(f"slot {self.name!r}: enum_values only valid for ENUM")
        return self


class Quadrant(StrEnum):
    """Volume x complexity migration matrix."""

    IMMEDIATE_MIGRATION = "immediate_migration"
    PHASED_AGENTIC = "phased_agentic"
    SELF_SERVICE_SCRIPTED = "self_service_scripted"
    DIRECT_AGENT_ROUTE = "direct_agent_route"


class QuadrantThresholds(Frozen):
    """Cut points for the 2x2. Pack-configurable -- never hardcoded in the miner."""

    volume_share: float = Field(default=0.05, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)


class VolumeStats(Frozen):
    utterance_count: int = Field(ge=0)
    call_count: int = Field(ge=0)
    share_of_total: float = Field(ge=0.0, le=1.0)
    avg_turns_to_resolve: float | None = Field(default=None, ge=0.0)
    historical_escalation_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class AutomationScore(Frozen):
    feasibility: float = Field(ge=0.0, le=1.0)
    complexity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    volume_share: float = Field(ge=0.0, le=1.0)
    thresholds: QuadrantThresholds = QuadrantThresholds()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quadrant(self) -> Quadrant:
        high_volume = self.volume_share >= self.thresholds.volume_share
        high_complexity = self.complexity >= self.thresholds.complexity
        if high_volume and not high_complexity:
            return Quadrant.IMMEDIATE_MIGRATION
        if high_volume:
            return Quadrant.PHASED_AGENTIC
        if not high_complexity:
            return Quadrant.SELF_SERVICE_SCRIPTED
        return Quadrant.DIRECT_AGENT_ROUTE


class EscalationPolicy(Frozen):
    min_intent_confidence: float = Field(default=0.82, ge=0.0, le=1.0)
    max_clarifications: int = Field(default=2, ge=0, le=5)
    max_tool_failures: int = Field(default=2, ge=0, le=5)
    frustration_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_escalate: bool = False
    """True for intents a bot must never attempt -- typically RiskTier.REGULATED."""

    target_queue: Slug | None = None
    required_skills: tuple[Slug, ...] = ()


class IntentNode(Frozen):
    intent_id: Slug
    """Dotted slug whose depth equals ``level``, e.g. ``billing.dispute.status``."""

    level: Literal[1, 2, 3]
    parent_id: Slug | None = None
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    exemplars: tuple[str, ...] = Field(default=(), max_length=20)
    cluster_id: int | None = None
    centroid: tuple[float, ...] | None = None
    volume: VolumeStats
    automation: AutomationScore
    slots: tuple[SlotSpec, ...] = ()
    required_tools: tuple[Slug, ...] = ()
    risk_tier: RiskTier = RiskTier.LOW
    escalation: EscalationPolicy = EscalationPolicy()

    @model_validator(mode="after")
    def _check_hierarchy(self) -> Self:
        depth = self.intent_id.count(".") + 1
        if depth != self.level:
            raise ValueError(
                f"intent_id {self.intent_id!r} has depth {depth} but level is {self.level}"
            )
        if self.level == 1 and self.parent_id is not None:
            raise ValueError(f"L1 node {self.intent_id!r} must not have a parent")
        if self.level > 1:
            if self.parent_id is None:
                raise ValueError(f"L{self.level} node {self.intent_id!r} requires a parent")
            if not self.intent_id.startswith(f"{self.parent_id}."):
                raise ValueError(
                    f"intent_id {self.intent_id!r} is not nested under {self.parent_id!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_slot_names_unique(self) -> Self:
        names = [s.name for s in self.slots]
        if len(names) != len(set(names)):
            raise ValueError(f"intent {self.intent_id!r} has duplicate slot names")
        return self

    @property
    def required_slots(self) -> tuple[SlotSpec, ...]:
        return tuple(s for s in self.slots if s.required)


class TaxonomyProvenance(StrEnum):
    """How a taxonomy came to exist. The two are not interchangeable."""

    MINED = "mined"
    """Discovered by clustering an unlabelled corpus (Module 3). Describes what callers
    actually said, and only as well as the clustering covered them."""

    ADOPTED = "adopted"
    """Taken from a labelled corpus's published label set. No clustering happened, so
    coverage of the *corpus* is total by construction -- and coverage of real traffic is
    entirely unknown, because the labels were somebody else's design."""


class CorpusHomogeneity(Frozen):
    """How much callers repeat each other. Context for ``coverage``, never a correction.

    Density-based clustering needs density, so a corpus of near-identical calls scores high
    coverage and yields few useful intents. Coverage read alone is therefore not a quality
    signal, and this travels beside it.

    Read ``lift``, not ``observed_overlap``. Raw overlap is dominated by vocabulary size --
    identical calls drawn from a 245-word vocabulary score 0.44 where identical calls from a
    160-word one score 0.88 -- so two corpora's raw numbers are not comparable. The null
    absorbs that, and the ratio is what carries meaning.
    """

    @property
    def lift(self) -> float | None:
        """Observed over null. 1.0 is a corpus no more repetitive than chance."""
        if self.observed_overlap is None or not self.null_overlap:
            return None
        return round(self.observed_overlap / self.null_overlap, 4)

    observed_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    """Mean binary cosine between random pairs of calls. None when unmeasurable."""

    null_overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    """The same statistic on a corpus whose vocabularies are redrawn at the same sizes
    from the same word distribution. Absorbs the size effect that defeated two earlier
    attempts at a size *correction*."""

    min_vocabulary: int = Field(ge=1)
    qualifying_share: float = Field(ge=0.0, le=1.0)
    """Fraction of calls with enough distinct vocabulary to be measured. Low means the
    number describes the corpus's wordy tail rather than the corpus."""

    calls_measured: int = Field(ge=0)


class IntentTaxonomy(Frozen):
    #: 1.1 -- added embedding_mean. See docs/tech-spec.md 1.4 for the migration.
    schema_version: SchemaVersion = "1.1"
    taxonomy_id: str = Field(min_length=1, max_length=128)
    domain: Slug
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    nodes: tuple[IntentNode, ...] = Field(min_length=1)
    provenance: TaxonomyProvenance = TaxonomyProvenance.MINED
    adopted_from: str = ""
    """The corpus whose published labels these are. Set iff ADOPTED."""

    derivation_split: str = ""
    """Which half of that corpus defined this taxonomy, so an evaluator knows what to hold
    out. Grading on the rows that defined it would measure nothing."""

    embedding_model: str = ""
    #: The corpus mean subtracted before clustering, empty when clustering was raw.
    #: Every ``IntentNode.centroid`` lives in the space this defines, so a vector compared
    #: against one has to be mapped in with it first. It is stored rather than recomputed
    #: because it is a property of the corpus that was mined -- re-deriving it later from
    #: a different corpus, or from a sample, silently moves the centroids.
    embedding_mean: tuple[float, ...] = ()
    #: Reported beside `coverage` so nobody reads coverage as a quality score.
    homogeneity: CorpusHomogeneity | None = None
    clusterer: Literal["hdbscan", "bertopic", "none"] = "none"
    clusterer_params: dict[str, JsonValue] = Field(default_factory=dict)
    labeler_model: str = ""
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    """Fraction of utterances the clustering assigned rather than left as noise.

    ``None`` on an adopted taxonomy, because nothing was clustered and the field has no
    value to report. Reporting 1.0 there would read as a perfect score beside a mined
    taxonomy's 0.197 and invite a comparison between numbers that do not mean the same
    thing."""

    noise_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    source_call_ids: tuple[str, ...] = ()
    generated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _check_provenance_is_complete(self) -> Self:
        """Each provenance requires its own evidence, and forbids the other's.

        A mined taxonomy nobody can reproduce is not an asset; an adopted one that asserts
        an embedding model is lying about where it came from.
        """
        if self.provenance is TaxonomyProvenance.MINED:
            missing = [
                name
                for name, value in (
                    ("embedding_model", self.embedding_model),
                    ("labeler_model", self.labeler_model),
                )
                if not value
            ]
            if missing or self.clusterer == "none":
                raise ValueError(
                    f"mined taxonomy {self.taxonomy_id!r} is not reproducible: "
                    f"missing {missing or ['clusterer']}"
                )
            if self.adopted_from:
                raise ValueError("a mined taxonomy has no adopted_from")
            if self.coverage is None:
                raise ValueError(
                    f"mined taxonomy {self.taxonomy_id!r} must report coverage -- it is how "
                    "well the clustering covered the corpus, and nothing else says"
                )
            return self

        if not self.adopted_from:
            raise ValueError(
                f"adopted taxonomy {self.taxonomy_id!r} must name the corpus it came from"
            )
        if self.embedding_model or self.labeler_model or self.clusterer != "none":
            raise ValueError(
                f"adopted taxonomy {self.taxonomy_id!r} claims mining provenance it does not have"
            )
        return self

    @model_validator(mode="after")
    def _check_centroids_match_the_space(self) -> Self:
        """A centroid and the mean that defines its space must have the same width."""
        if not self.embedding_mean:
            return self
        width = len(self.embedding_mean)
        for node in self.nodes:
            if node.centroid is not None and len(node.centroid) != width:
                raise ValueError(
                    f"intent {node.intent_id!r} has a {len(node.centroid)}-d centroid but "
                    f"embedding_mean is {width}-d; they describe different spaces"
                )
        return self

    @model_validator(mode="after")
    def _check_graph_integrity(self) -> Self:
        by_id = {node.intent_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("duplicate intent_id in taxonomy")
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in by_id:
                raise ValueError(
                    f"intent {node.intent_id!r} references missing parent {node.parent_id!r}"
                )
        return self

    @model_validator(mode="after")
    def _check_coverage_consistency(self) -> Self:
        if self.coverage is None and self.noise_ratio is None:
            return self
        if self.coverage is None or self.noise_ratio is None:
            raise ValueError("coverage and noise_ratio are reported together or not at all")
        total = self.coverage + self.noise_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"coverage + noise_ratio must equal 1.0, got {total}")
        return self

    @property
    def by_id(self) -> dict[str, IntentNode]:
        return {node.intent_id: node for node in self.nodes}

    def resolve(self, intent_id: str) -> IntentNode:
        try:
            return self.by_id[intent_id]
        except KeyError as exc:
            raise KeyError(f"unknown intent {intent_id!r} in taxonomy {self.domain}") from exc

    def path(self, intent_id: str) -> tuple[IntentNode, ...]:
        """Walk L1 -> L2 -> L3 for an intent."""
        chain: list[IntentNode] = []
        current: IntentNode | None = self.resolve(intent_id)
        while current is not None:
            chain.append(current)
            current = self.by_id[current.parent_id] if current.parent_id else None
        return tuple(reversed(chain))

    def children(self, intent_id: str | None) -> tuple[IntentNode, ...]:
        return tuple(n for n in self.nodes if n.parent_id == intent_id)

    def leaves(self) -> tuple[IntentNode, ...]:
        parents = {n.parent_id for n in self.nodes if n.parent_id}
        return tuple(n for n in self.nodes if n.intent_id not in parents)
