"""Sentiment policy.

Two triggers, because an absolute threshold alone fires too late. A caller whose
frustration climbs steadily across turns has already decided the bot is not helping;
waiting for them to cross a fixed line costs another turn they did not want to spend.

    absolute   frustration >= threshold
    slope      two consecutive rises, ending above half the threshold

Sentiment is read from ``SessionState.sentiment_trail``, which Module 5 populates. With
no readings the policy abstains rather than guessing.
"""

from __future__ import annotations

from itertools import pairwise

from ccas.policies.base import PolicyAction, PolicyContext, PolicyDecision, proceed
from ccas.schemas.common import Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.session import SessionState

__all__ = ["MIN_TREND_READINGS", "NAME", "evaluate"]

NAME = "sentiment"

#: A trend needs three points; two is a single change, which is noise in speech.
MIN_TREND_READINGS = 3


def evaluate(state: SessionState, ctx: PolicyContext) -> PolicyDecision:
    trail = state.sentiment_trail
    if not trail:
        return proceed(NAME)

    threshold = ctx.escalation.frustration_threshold
    latest = trail[-1]

    if latest.frustration_index >= threshold:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            policy=NAME,
            reason=HandoffReason.NEGATIVE_SENTIMENT,
            urgency=Urgency.HIGH,
            detail=(f"frustration {latest.frustration_index:.2f} at or above {threshold:.2f}"),
        )

    if _rising(state) and latest.frustration_index >= threshold / 2:
        return PolicyDecision(
            action=PolicyAction.ESCALATE,
            policy=NAME,
            reason=HandoffReason.NEGATIVE_SENTIMENT,
            urgency=Urgency.NORMAL,
            detail=(
                f"frustration rising across {MIN_TREND_READINGS} turns to "
                f"{latest.frustration_index:.2f}"
            ),
        )

    return proceed(NAME)


def _rising(state: SessionState) -> bool:
    recent = state.sentiment_trail[-MIN_TREND_READINGS:]
    if len(recent) < MIN_TREND_READINGS:
        return False
    return all(
        later.frustration_index > earlier.frustration_index for earlier, later in pairwise(recent)
    )
