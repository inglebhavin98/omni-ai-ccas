"""Corpus homogeneity -- the context `coverage` needs to not mislead.

Density-based clustering needs density, so a corpus of near-identical calls scores superb
coverage and yields one useful intent. Coverage alone is therefore not a quality metric,
and a migration decision made on it would be made on the wrong number.

The sampling rule is part of the contract, not a docstring: sampling K *tokens* rather
than K *distinct* words moved the measured effect by a factor of five between two
implementations, because a repetitive call yields fewer distinct words from K tokens --
which is exactly the quantity being measured. A control that encodes the treatment.
"""

from __future__ import annotations

import pytest

from ccas.mining.homogeneity import (
    MIN_VOCABULARY,
    content_vocabulary,
    measure_homogeneity,
)


def _identical(n: int, words: int = 200) -> list[set[str]]:
    vocab = {f"w{i}" for i in range(words)}
    return [set(vocab) for _ in range(n)]


def _disjoint(n: int, words: int = 200) -> list[set[str]]:
    return [{f"c{c}w{i}" for i in range(words)} for c in range(n)]


def test_identical_calls_score_maximum_overlap() -> None:
    """Nothing is subsampled, so identical calls score 1.0 -- not a size-dependent ceiling."""
    report = measure_homogeneity(_identical(60))
    assert report.observed_overlap == pytest.approx(1.0, abs=1e-6)
    assert report.qualifying_share == 1.0


def test_disjoint_calls_score_zero_overlap() -> None:
    report = measure_homogeneity(_disjoint(60))
    assert report.observed_overlap == pytest.approx(0.0, abs=1e-6)


def test_raw_overlap_is_not_comparable_across_vocabulary_sizes() -> None:
    """Why `lift` is the number to read.

    Two corpora with the same real structure -- a shared core plus per-call noise -- where
    calls differ only in length. Raw overlap moves with call size; lift should not, because
    the null moves with it too.
    """

    def corpus(unique_per_call: int) -> list[set[str]]:
        shared = {f"s{i}" for i in range(60)}
        return [shared | {f"u{c}w{i}" for i in range(unique_per_call)} for c in range(80)]

    terse = measure_homogeneity(corpus(20))
    wordy = measure_homogeneity(corpus(200))
    assert terse.observed_overlap is not None and wordy.observed_overlap is not None
    assert terse.observed_overlap > wordy.observed_overlap * 2, "raw overlap tracks call size"
    assert terse.lift is not None and wordy.lift is not None
    assert terse.lift > 1.5 and wordy.lift > 1.5, "both corpora are genuinely structured"


def test_lift_is_one_when_a_corpus_is_no_more_repetitive_than_chance() -> None:
    """The null's own corpus must score ~1.0, or the baseline is not a baseline."""
    rng = __import__("numpy").random.default_rng(5)
    words = [f"w{i}" for i in range(4000)]
    corpus = [set(rng.choice(words, size=120, replace=False)) for _ in range(120)]
    report = measure_homogeneity(corpus)
    assert report.lift is not None
    assert 0.8 < report.lift < 1.25


def test_a_repetitive_corpus_lifts_clearly_above_a_random_one() -> None:
    """The discrimination that matters, against a measured baseline rather than a guess.

    Two-thirds of every call is a shared core: lift measures 1.89. A corpus drawn at random
    from the same vocabulary measures ~1.0. The threshold sits between them.
    """
    shared = {f"s{i}" for i in range(80)}
    structured = measure_homogeneity([shared | {f"u{c}w{i}" for i in range(40)} for c in range(80)])

    rng = __import__("numpy").random.default_rng(3)
    words = [f"w{i}" for i in range(3200)]
    unstructured = measure_homogeneity(
        [set(rng.choice(words, size=120, replace=False)) for _ in range(80)]
    )

    assert structured.lift is not None and unstructured.lift is not None
    assert structured.lift > 1.5
    assert unstructured.lift < 1.25
    assert structured.lift > unstructured.lift * 1.5


def test_the_default_floor_is_the_one_recorded() -> None:
    assert measure_homogeneity(_identical(60)).min_vocabulary == MIN_VOCABULARY


def test_the_floor_is_recorded_and_thin_calls_are_counted_out() -> None:
    vocabs = _identical(40) + [{"a", "b"} for _ in range(60)]
    report = measure_homogeneity(vocabs, min_vocabulary=100)
    assert report.min_vocabulary == 100
    assert report.calls_measured == 40
    assert report.qualifying_share == pytest.approx(0.4)


def test_too_few_qualifying_calls_is_unmeasurable_not_zero() -> None:
    """Reporting 0.0 would read as 'no repetition', which is the opposite of unknown."""
    report = measure_homogeneity([{"a", "b"} for _ in range(50)], min_vocabulary=100)
    assert report.observed_overlap is None
    assert report.lift is None
    assert report.calls_measured == 0


def test_measurement_is_deterministic_for_a_given_seed() -> None:
    corpus = [{f"w{(i * 7 + j) % 300}" for j in range(150)} for i in range(50)]
    assert measure_homogeneity(corpus, seed=11).lift == measure_homogeneity(corpus, seed=11).lift


def test_content_vocabulary_drops_stopwords_and_short_tokens() -> None:
    vocab = content_vocabulary("The caller said the policy is ok and I do not know")
    assert "caller" in vocab and "policy" in vocab
    for word in ("the", "is", "and", "not", "ok", "do", "i"):
        assert word not in vocab


def test_content_vocabulary_ignores_placeholders() -> None:
    """Redaction placeholders are identical across every call and would inflate overlap."""
    vocab = content_vocabulary("my reference is [ACCOUNT_REF_1] and [PERSON_2] called")
    assert "account_ref" not in vocab
    assert "person" not in vocab
    assert "reference" in vocab and "called" in vocab
