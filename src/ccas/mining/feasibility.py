"""Volume x complexity feasibility scoring (Module 3).

The 2x2 is what sequences a migration: which intents move first, which need a phased
agentic build, which stay scripted, and which should go straight to a human. Thresholds
come from the pack so two verticals can disagree about what counts as high volume.
"""

from __future__ import annotations

from ccas.mining.labeler import ClusterLabel
from ccas.schemas.taxonomy import AutomationScore, QuadrantThresholds, VolumeStats

__all__ = ["score_cluster", "volume_stats"]


def volume_stats(
    utterance_count: int,
    call_count: int,
    corpus_utterances: int,
    avg_turns: float | None = None,
) -> VolumeStats:
    share = utterance_count / corpus_utterances if corpus_utterances else 0.0
    return VolumeStats(
        utterance_count=utterance_count,
        call_count=call_count,
        share_of_total=min(share, 1.0),
        avg_turns_to_resolve=avg_turns,
    )


def score_cluster(
    label: ClusterLabel,
    volume: VolumeStats,
    thresholds: QuadrantThresholds,
) -> AutomationScore:
    """Combine the model's judgement with the corpus's measured volume.

    Complexity and feasibility are the model's read of the utterances; volume is a fact
    about the corpus. Keeping them separate matters -- an LLM asked to estimate volume
    would simply guess, and the quadrant would stop meaning anything.
    """
    return AutomationScore(
        feasibility=label.feasibility,
        complexity=label.complexity,
        confidence=_confidence(volume),
        rationale=label.rationale,
        volume_share=volume.share_of_total,
        thresholds=thresholds,
    )


def _confidence(volume: VolumeStats) -> float:
    """How much to trust this cluster's scoring.

    A cluster of 8 utterances is a guess; one of 800 is evidence. Saturates at 200 so a
    very large cluster does not read as certainty about its *labelling*.
    """
    return round(min(volume.utterance_count / 200.0, 1.0), 4)
