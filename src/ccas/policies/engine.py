"""Ordered policy evaluation.

First decisive verdict wins, and the winner's name is carried through to
``EscalationDecision.triggered_by`` -- so a handoff always states which rule fired
rather than implying one.

Order is a design decision, not an implementation detail:

1. ``risk``       a regulated intent must not be attempted, whatever else is true
2. ``retry``      an exhausted call cannot be repaired by asking again
3. ``sentiment``  a furious caller must not be asked a clarifying question
4. ``confidence`` only then does it matter whether we understood them

Putting ``confidence`` before ``sentiment`` would produce the worst turn in the system:
asking an already-frustrated caller to rephrase.
"""

from __future__ import annotations

from collections.abc import Sequence

from ccas.policies import confidence, retry, risk, sentiment
from ccas.policies.base import PROCEED, Policy, PolicyContext, PolicyDecision
from ccas.schemas.session import SessionState

__all__ = ["DEFAULT_POLICIES", "PolicyEngine", "evaluate"]

DEFAULT_POLICIES: tuple[Policy, ...] = (
    risk.evaluate,
    retry.evaluate,
    sentiment.evaluate,
    confidence.evaluate,
)


def evaluate(
    state: SessionState,
    ctx: PolicyContext,
    policies: Sequence[Policy] = DEFAULT_POLICIES,
) -> PolicyDecision:
    for policy in policies:
        decision = policy(state, ctx)
        if decision.decisive:
            return decision
    return PROCEED


class PolicyEngine:
    """Bound to one policy order. Holding it as an object keeps the graph nodes free of
    the ordering question -- they ask for a verdict, not for a sequence."""

    __slots__ = ("_policies",)

    def __init__(self, policies: Sequence[Policy] = DEFAULT_POLICIES) -> None:
        self._policies = tuple(policies)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(getattr(p, "__module__", "?").rsplit(".", 1)[-1] for p in self._policies)

    def evaluate(self, state: SessionState, ctx: PolicyContext) -> PolicyDecision:
        return evaluate(state, ctx, self._policies)

    def evaluate_all(self, state: SessionState, ctx: PolicyContext) -> tuple[PolicyDecision, ...]:
        """Every verdict, for diagnostics. The graph uses ``evaluate``; this is for the
        demo console and the copilot, where seeing *why* matters."""
        return tuple(policy(state, ctx) for policy in self._policies)
