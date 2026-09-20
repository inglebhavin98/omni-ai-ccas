"""What a confidence cutoff actually buys, measured rather than assumed.

The pack routes when ``intent_confidence >= 0.82`` and escalates below it. That number
was picked for one model's self-reported confidence. It is not portable: a different
model reports a different statistic, and TypeSafe ``jev`` reports the *shape* of a
probability distribution rather than the winning option's probability at all -- its
documentation says outright not to carry a threshold between question formats.

So a cutoff is not a config value to copy. It is a measurement, and this is the
instrument: sweep candidate cutoffs over rows whose answer is known and report, for each,
how much traffic clears it and how often the cleared traffic is right.

The deferred slice is scored twice -- at the leaf and at the L1 category -- because
"unsure" is not the same as "no idea". A row the router will not commit to may still have
the right category, and routing it to that category's queue is a cheaper answer than
sending the caller to a human. Whether that is true here is exactly what the second
number settles.

Pure. The caller supplies outcomes from a live run, a cassette or a stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from ccas.evals.router_accuracy import Outcomes, measured_rows

__all__ = ["DEFAULT_THRESHOLDS", "ConfidenceBand", "recommend", "sweep", "sweep_table"]

#: Coarse enough to read in one screen, fine enough to bracket the pack's 0.82.
DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95)


@dataclass(frozen=True, slots=True)
class ConfidenceBand:
    """One candidate cutoff, and the run split in two by it."""

    threshold: float
    auto: int
    """Rows at or above the cutoff -- what would be handled without a human."""

    auto_exact: int
    deferred: int
    deferred_exact: int
    deferred_category: int
    """Deferred rows whose L1 category was right even though the leaf was not."""

    ungated: int = 0
    """Answered, but with no confidence to gate on. Counted apart rather than folded into
    either slice: a provider that reports no confidence has not said the row is safe, and
    it has not said the row is doubtful either."""

    @property
    def gated(self) -> int:
        return self.auto + self.deferred

    @property
    def coverage(self) -> float:
        """Share of gateable traffic that clears the cutoff."""
        return self.auto / self.gated if self.gated else 0.0

    @property
    def auto_accuracy(self) -> float:
        # An empty slice is 0%, not 100% -- the same reading ADR-0014 forced on parity.
        return self.auto_exact / self.auto if self.auto else 0.0

    @property
    def deferred_accuracy(self) -> float:
        return self.deferred_exact / self.deferred if self.deferred else 0.0

    @property
    def deferred_category_accuracy(self) -> float:
        return self.deferred_category / self.deferred if self.deferred else 0.0


def sweep(
    outcomes: Outcomes, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> tuple[ConfidenceBand, ...]:
    """One band per candidate cutoff, over the rows that were actually measured."""
    rows = [
        (case, outcome)
        for case, outcome in measured_rows(outcomes)
        if outcome.answered and outcome.predicted is not None
    ]
    bands: list[ConfidenceBand] = []
    for threshold in thresholds:
        auto = auto_exact = deferred = deferred_exact = deferred_category = ungated = 0
        for case, outcome in rows:
            predicted = outcome.predicted or ""
            exact = predicted == case.expected_intent
            if outcome.confidence is None:
                ungated += 1
                continue
            if outcome.confidence >= threshold:
                auto += 1
                auto_exact += int(exact)
                continue
            deferred += 1
            deferred_exact += int(exact)
            deferred_category += int(predicted.split(".", 1)[0] == case.expected_category)
        bands.append(
            ConfidenceBand(
                threshold=threshold,
                auto=auto,
                auto_exact=auto_exact,
                deferred=deferred,
                deferred_exact=deferred_exact,
                deferred_category=deferred_category,
                ungated=ungated,
            )
        )
    return tuple(bands)


def recommend(
    bands: tuple[ConfidenceBand, ...], min_auto_accuracy: float, min_auto: int = 5
) -> ConfidenceBand | None:
    """The cheapest cutoff that still clears the accuracy floor, or nothing.

    Lowest rather than highest: every point of cutoff beyond what the floor requires is
    coverage given away, and each deferred row costs a human. ``min_auto`` stops the
    sweep recommending a cutoff whose evidence is two rows -- a perfect slice of two
    measures nothing, which is the argument ADR-0014 makes about a parity run that
    measured nothing. Returning ``None`` is a real answer: no cutoff in this sweep is
    supportable on this evidence.
    """
    for band in sorted(bands, key=lambda b: b.threshold):
        if band.auto >= min_auto and band.auto_accuracy >= min_auto_accuracy:
            return band
    return None


def sweep_table(bands: tuple[ConfidenceBand, ...]) -> str:
    """The sweep as one block of text, so two harnesses print the same shape.

    Formatting rather than presentation logic: the whole point of the study is comparing
    a jev cutoff with a chat model's, and two tables with different columns would make
    that comparison by eye harder than it needs to be.
    """
    lines = [
        f"{'cutoff':>7}  {'auto':>5}  {'auto ok':>8}  {'covered':>8}  "
        f"{'deferred':>9}  {'def ok':>7}  {'def cat':>8}"
    ]
    for band in bands:
        lines.append(
            f"{band.threshold:>7.2f}  {band.auto:>5}  {band.auto_accuracy:>7.1%}  "
            f"{band.coverage:>7.1%}  {band.deferred:>9}  {band.deferred_accuracy:>6.1%}  "
            f"{band.deferred_category_accuracy:>7.1%}"
        )
    return "\n".join(lines)
