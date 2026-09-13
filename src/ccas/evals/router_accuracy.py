"""Does the router assign the right intent? (docs/future-scoped-work.md 9.18)

Until this runs, "the chat channel works" means "it routes without crashing", which is a
different statement from "it routes correctly", and much weaker than it sounds.

Two rules are enforced here rather than left to the caller, because both have already
caught this project out once.

**A row that defined the taxonomy cannot grade it.** `domains/retail/taxonomy.json` is
adopted from Bitext's published labels (ADR-0020), so scoring against the rows that
produced those labels measures the label set agreeing with itself. `score_router` raises
on a derivation row rather than trusting whoever assembled the batch to have filtered it.

**A row nobody routed is unmeasured, not wrong.** The distinction ADR-0014 forced on the
parity gate. A 429 is not a misclassification, and scoring it as one reports a broken
router when the provider was out of quota.

Pure: calling the router is the caller's job, which is what lets the same report come from
a live run, a cassette, or a stub.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ccas.mining.adopt import DERIVATION_SPLIT, split_of

__all__ = [
    "IntentScore",
    "RouterCase",
    "RouterOutcome",
    "RouterReport",
    "naturalise",
    "score_router",
]


#: Bitext marks parameters inline: "cancel order {{Order Number}}".
_TEMPLATE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def naturalise(utterance: str) -> str:
    """Strip template markup from a corpus utterance, keeping the words.

    A quarter of Bitext's caller lines carry `{{Order Number}}`-style syntax. Nobody types
    that, so grading on it measures how a model reacts to synthetic markup rather than
    whether it understood a request. Substituting a plausible value would be worse -- it
    would invent data the corpus does not contain and quietly change what is being tested.
    """
    return re.sub(r"\s{2,}", " ", _TEMPLATE.sub(lambda m: m.group(1).lower(), utterance)).strip()


@dataclass(frozen=True, slots=True)
class RouterCase:
    """One held-out utterance and the intent its corpus says it is."""

    row_id: str
    utterance: str
    expected_intent: str

    @property
    def expected_category(self) -> str:
        return self.expected_intent.split(".", 1)[0]


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    """What the router said, or why it said nothing."""

    predicted: str | None = None
    confidence: float | None = None
    error: str | None = None
    unavailable: bool = False
    """The request was never served -- quota, outage. Excluded from the denominator."""

    latency_ms: int = 0

    @property
    def answered(self) -> bool:
        return self.predicted is not None and not self.unavailable


@dataclass(frozen=True, slots=True)
class IntentScore:
    intent_id: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class RouterReport:
    measured: int
    unmeasured: int
    exact: int
    category_correct: int
    per_intent: tuple[tuple[str, IntentScore], ...]
    confusions: tuple[tuple[str, str, int], ...]
    """(expected, predicted, count), most frequent first. Where the router actually goes."""

    p95_latency_ms: int = 0

    @property
    def exact_accuracy(self) -> float:
        # Zero measured is zero accuracy, not perfect. An all-throttled run must not read
        # as a green router (ADR-0014).
        return self.exact / self.measured if self.measured else 0.0

    @property
    def category_accuracy(self) -> float:
        """Right L1, whatever the leaf. Separates "lost" from "nearly right"."""
        return self.category_correct / self.measured if self.measured else 0.0

    def meets(self, threshold: float, min_cases: int = 1) -> bool:
        return self.measured >= max(1, min_cases) and self.exact_accuracy >= threshold

    def worst_intents(self, k: int = 5) -> tuple[tuple[str, IntentScore], ...]:
        """Lowest accuracy first. One collapsed intent is invisible in an overall number."""
        return tuple(sorted(self.per_intent, key=lambda kv: (kv[1].accuracy, -kv[1].total))[:k])

    def summary(self) -> str:
        return (
            f"{self.exact}/{self.measured} exact ({self.exact_accuracy:.1%}), "
            f"category {self.category_accuracy:.1%}, "
            f"{self.unmeasured} unmeasured, p95 {self.p95_latency_ms} ms"
        )


Outcomes = Sequence[tuple[RouterCase, RouterOutcome]]


def score_router(outcomes: Outcomes) -> RouterReport:
    """Fold per-row outcomes into one report, refusing any row that is not held out."""
    offenders = [c.row_id for c, _ in outcomes if split_of(c.row_id) == DERIVATION_SPLIT]
    if offenders:
        raise ValueError(
            f"{len(offenders)} row(s) are from the derivation split and defined this "
            f"taxonomy's labels; only held out rows may grade it (first: {offenders[0]!r})"
        )

    measured = unmeasured = exact = category_correct = 0
    correct_by_intent: dict[str, int] = {}
    total_by_intent: dict[str, int] = {}
    confusions: dict[tuple[str, str], int] = {}
    latencies: list[int] = []

    for case, outcome in outcomes:
        if outcome.unavailable:
            unmeasured += 1
            continue
        measured += 1
        latencies.append(outcome.latency_ms)
        total_by_intent[case.expected_intent] = total_by_intent.get(case.expected_intent, 0) + 1

        predicted = outcome.predicted or ""
        if predicted == case.expected_intent:
            exact += 1
            category_correct += 1
            correct_by_intent[case.expected_intent] = (
                correct_by_intent.get(case.expected_intent, 0) + 1
            )
            continue
        if predicted.split(".", 1)[0] == case.expected_category:
            category_correct += 1
        key = (case.expected_intent, predicted or "<none>")
        confusions[key] = confusions.get(key, 0) + 1

    per_intent = tuple(
        (
            intent,
            IntentScore(intent, correct_by_intent.get(intent, 0), total),
        )
        for intent, total in sorted(total_by_intent.items())
    )
    ranked = tuple(
        (expected, predicted, n)
        for (expected, predicted), n in sorted(confusions.items(), key=lambda kv: -kv[1])
    )
    return RouterReport(
        measured=measured,
        unmeasured=unmeasured,
        exact=exact,
        category_correct=category_correct,
        per_intent=per_intent,
        confusions=ranked,
        p95_latency_ms=_p95(latencies),
    )


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, -(-95 * len(ordered) // 100))
    return ordered[rank - 1]
