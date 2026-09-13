"""Assemble an ``IntentTaxonomy`` from redacted call logs (Module 3).

    utterances -> embed -> HDBSCAN -> LLM labelling -> feasibility -> taxonomy

Only *caller* utterances are mined. Agent lines describe the resolution, not the intent,
and including them would produce clusters of stock phrases like "let me look that up".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ccas.ingestion.datasets import DatasetRole, require_role
from ccas.mining.cluster import Clusterer, centroids, representatives
from ccas.mining.embedder import Embedder, EmbeddingCache
from ccas.mining.feasibility import score_cluster, volume_stats
from ccas.mining.labeler import ClusterBrief, ClusterLabel, IntentLabeler, MinedSlot
from ccas.schemas.call_log import CallLog
from ccas.schemas.common import RiskTier, utcnow
from ccas.schemas.pii import PiiEntityType, RedactedText
from ccas.schemas.taxonomy import (
    EscalationPolicy,
    IntentNode,
    IntentTaxonomy,
    QuadrantThresholds,
    SlotSpec,
)

__all__ = ["MinedUtterance", "TaxonomyBuilder", "collect_utterances"]


@dataclass(slots=True)
class MinedUtterance:
    call_id: str
    index: int
    content: RedactedText

    @property
    def text(self) -> str:
        return self.content.text


#: Characters below which a turn carries no reason for contact. Calibrated for a corpus
#: whose turns are real turns. A pause-segmented corpus wants a much higher floor --
#: see ADR-0013 and ``TaxonomyBuilder(min_chars=...)``.
DEFAULT_MIN_CHARS = 8


def collect_utterances(
    logs: Sequence[CallLog], min_chars: int = DEFAULT_MIN_CHARS
) -> list[MinedUtterance]:
    """Caller turns worth clustering.

    Very short turns ("yes", "okay", "mm") carry no intent and would otherwise form the
    largest cluster in any real corpus.
    """
    collected: list[MinedUtterance] = []
    for log in logs:
        for utterance in log.caller_utterances:
            if len(utterance.content.text.strip()) < min_chars:
                continue
            collected.append(MinedUtterance(log.call_id, utterance.index, utterance.content))
    return collected


@dataclass(slots=True)
class _Aggregate:
    """A taxonomy node under construction, before volumes are known."""

    label: ClusterLabel
    utterance_count: int = 0
    call_ids: set[str] = field(default_factory=set)
    cluster_ids: set[int] = field(default_factory=set)
    centroid: tuple[float, ...] | None = None


class TaxonomyBuilder:
    def __init__(
        self,
        embedder: Embedder,
        clusterer: Clusterer,
        labeler: IntentLabeler,
        thresholds: QuadrantThresholds | None = None,
        exemplars_per_cluster: int = 8,
        cache: EmbeddingCache | None = None,
        min_chars: int = DEFAULT_MIN_CHARS,
    ) -> None:
        self._embedder = embedder
        self._clusterer = clusterer
        self._labeler = labeler
        self._thresholds = thresholds or QuadrantThresholds()
        self._exemplars = exemplars_per_cluster
        self._cache = cache
        self._min_chars = min_chars

    async def build(
        self, logs: Sequence[CallLog], domain: str, version: str = "0.1.0"
    ) -> IntentTaxonomy:
        # Gate 1: this corpus must be admitted for mining (ADR-0004).
        for source in {log.source for log in logs}:
            require_role(source, DatasetRole.INTENT_MINING)

        utterances = collect_utterances(logs, min_chars=self._min_chars)
        if not utterances:
            raise ValueError("no caller utterances long enough to mine")

        texts = [u.text for u in utterances]
        vectors = (
            self._cache.encode(self._embedder, texts)
            if self._cache is not None
            else self._embedder.encode(texts)
        )

        result = self._clusterer.fit_predict(vectors)
        if result.n_clusters == 0:
            raise ValueError(
                f"clustering found no structure in {len(texts)} utterances "
                f"(all noise). Lower min_cluster_size, or ingest more data."
            )

        # Describe the clusters in the space they were found in. Centroids computed
        # from raw vectors would be crowded together by the shared direction centring
        # removed -- averaging concentrates it, because the orthogonal parts cancel and
        # the common part does not (ADR-0017).
        space = result.in_cluster_space(vectors)
        cluster_centroids = centroids(space, result)
        briefs = tuple(
            ClusterBrief(
                cluster_id=cluster_id,
                size=len(result.indices_for(cluster_id)),
                share=len(result.indices_for(cluster_id)) / len(texts),
                exemplars=tuple(
                    utterances[i].content
                    for i in representatives(space, result, cluster_id, self._exemplars)
                ),
            )
            for cluster_id in result.cluster_ids
        )

        labels = await self._labeler.label_all(briefs)

        # Conversational filler is not an intent. In a real corpus the closing line is
        # the single largest cluster; keeping it would make it the taxonomy's top node.
        intents = tuple(label for label in labels if label.is_intent)
        if not intents:
            raise ValueError(
                f"the labeler marked all {len(labels)} clusters as conversational "
                "filler; check the exemplars or the min_chars filter"
            )
        filler_utterances = sum(
            int(result.indices_for(label.cluster_id).size)
            for label in labels
            if not label.is_intent
        )
        intent_utterances = len(texts) - filler_utterances

        nodes = self._assemble(intents, result, utterances, cluster_centroids, intent_utterances)
        # Coverage is stated against intent-bearing utterances: filler is not signal we
        # failed to cluster, it is signal we correctly declined to name.
        assigned = sum(int(result.indices_for(label.cluster_id).size) for label in intents)
        coverage = assigned / intent_utterances if intent_utterances else 0.0

        return IntentTaxonomy(
            taxonomy_id=_taxonomy_id(domain, logs),
            domain=domain,
            version=version,
            nodes=nodes,
            embedding_model=self._embedder.name,
            embedding_mean=result.embedding_mean or (),
            clusterer="hdbscan",
            clusterer_params={
                **self._clusterer.params,
                "clusters_found": result.n_clusters,
                "clusters_labelled_filler": len(labels) - len(intents),
                "filler_utterances": filler_utterances,
            },
            labeler_model=self._labeler.model_name,
            coverage=round(coverage, 6),
            noise_ratio=round(1.0 - round(coverage, 6), 6),
            source_call_ids=tuple(sorted({log.call_id for log in logs})),
            generated_at=utcnow(),
        )

    def _assemble(
        self,
        labels: Sequence[ClusterLabel],
        result: object,
        utterances: Sequence[MinedUtterance],
        cluster_centroids: dict[int, np.ndarray],
        corpus_size: int,
    ) -> tuple[IntentNode, ...]:
        from ccas.mining.cluster import ClusterResult

        assert isinstance(result, ClusterResult)

        # Two clusters can be labelled into the same leaf -- the clusterer split
        # something the labeler considers one intent. Merge rather than disambiguate:
        # inventing `..._2` would put a fiction into the taxonomy.
        leaves: dict[str, _Aggregate] = {}
        for label in labels:
            aggregate = leaves.setdefault(label.l3_id, _Aggregate(label=label))
            indices = result.indices_for(label.cluster_id)
            aggregate.utterance_count += int(indices.size)
            aggregate.call_ids.update(utterances[int(i)].call_id for i in indices)
            aggregate.cluster_ids.add(label.cluster_id)
            if aggregate.centroid is None and label.cluster_id in cluster_centroids:
                aggregate.centroid = tuple(float(x) for x in cluster_centroids[label.cluster_id])

        nodes: list[IntentNode] = []
        nodes.extend(self._rollup_parents(leaves, corpus_size))
        for intent_id, aggregate in sorted(leaves.items()):
            volume = volume_stats(aggregate.utterance_count, len(aggregate.call_ids), corpus_size)
            nodes.append(
                IntentNode(
                    intent_id=intent_id,
                    level=3,
                    parent_id=aggregate.label.l2_id,
                    label=aggregate.label.l3_label,
                    description=aggregate.label.l3_description,
                    exemplars=(),
                    cluster_id=min(aggregate.cluster_ids),
                    centroid=aggregate.centroid,
                    volume=volume,
                    automation=score_cluster(aggregate.label, volume, self._thresholds),
                    slots=tuple(_to_slot_spec(s) for s in aggregate.label.slots),
                    required_tools=aggregate.label.required_tools,
                    risk_tier=aggregate.label.risk_tier,
                    escalation=_escalation_for(aggregate.label.risk_tier),
                )
            )
        return tuple(nodes)

    def _rollup_parents(self, leaves: dict[str, _Aggregate], corpus_size: int) -> list[IntentNode]:
        """L1 and L2 volumes are the sum of their children, never an independent guess."""
        parents: dict[str, _Aggregate] = {}
        for aggregate in leaves.values():
            for intent_id in (aggregate.label.l1_id, aggregate.label.l2_id):
                parent = parents.setdefault(intent_id, _Aggregate(label=aggregate.label))
                parent.utterance_count += aggregate.utterance_count
                parent.call_ids.update(aggregate.call_ids)

        nodes: list[IntentNode] = []
        for intent_id, aggregate in sorted(parents.items()):
            level = intent_id.count(".") + 1
            is_l1 = level == 1
            volume = volume_stats(aggregate.utterance_count, len(aggregate.call_ids), corpus_size)
            nodes.append(
                IntentNode(
                    intent_id=intent_id,
                    level=1 if is_l1 else 2,
                    parent_id=None if is_l1 else intent_id.rsplit(".", 1)[0],
                    label=aggregate.label.l1_label if is_l1 else aggregate.label.l2_label,
                    description=(
                        aggregate.label.l1_description if is_l1 else aggregate.label.l2_description
                    ),
                    volume=volume,
                    automation=score_cluster(aggregate.label, volume, self._thresholds),
                    risk_tier=aggregate.label.risk_tier,
                )
            )
        return nodes


def _to_slot_spec(slot: MinedSlot) -> SlotSpec:
    return SlotSpec(
        name=slot.name,
        slot_type=slot.slot_type,
        required=slot.required,
        elicitation_prompt=slot.elicitation_prompt,
        pii_entity=PiiEntityType.ACCOUNT_REF if slot.pii_sensitive else None,
        dtmf_capturable=slot.dtmf_capturable,
    )


def _escalation_for(risk_tier: RiskTier) -> EscalationPolicy:
    """A regulated intent must never be attempted by a bot (CLAUDE.md Rule 4)."""
    if risk_tier is RiskTier.REGULATED:
        return EscalationPolicy(auto_escalate=True, min_intent_confidence=0.0)
    return EscalationPolicy()


def _taxonomy_id(domain: str, logs: Sequence[CallLog]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode())
    for call_id in sorted({log.call_id for log in logs}):
        digest.update(call_id.encode())
    return f"{domain}-{digest.hexdigest()[:16]}"
