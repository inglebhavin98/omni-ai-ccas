"""CLAUDE.md Rule 6 -- no node depends on a single model.

Two halves, deliberately separated:

* The **structural** half runs everywhere. It reads ``configs/models.yaml`` and asserts
  every node declares at least two variants naming distinct models. It needs no network,
  so a change that quietly pins a node to one model fails in CI, not in production.
* The **behavioural** half is marked ``integration``. It puts the same utterances to both
  router variants and asserts they agree on the intent. It needs real credentials and is
  skipped without them -- but a skip is visible, and `make evals` runs it.

Agreement is on the decision, never the wording. See ``ccas.evals.parity``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from ccas.config.settings import Settings
from ccas.evals.parity import ParityCase, VariantOutcome, compare_variants
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProvider, LLMProviderError, ProviderRateLimitedError
from ccas.llm.bindings import MIN_VARIANTS, load_bindings
from ccas.llm.factory import build_provider
from ccas.observability.logging import get_logger
from ccas.schemas.taxonomy import IntentTaxonomy
from tests.factories import redacted

REPO = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO / "configs" / "models.yaml"
LOG = get_logger("evals.parity")

#: Two different models will not agree on every borderline utterance, and demanding that
#: they do would mean tuning the threshold until the gate is meaningless. This is the
#: level below which the graph's behaviour depends on which variant served the turn.
MIN_AGREEMENT = 0.75

#: Below this many *measured* cases the run says nothing, whatever its agreement rate.
#: Free-tier throttling drops cases silently, and a one-case run at 100% must not go green.
MIN_MEASURED_CASES = 5

#: Free-tier models are rate limited per minute. Firing every case at once turns a parity
#: run into a throttling test, so the cases are serialised with a gap between them.
PACING_S = 3.0

#: Unambiguous cases. A parity gate built on genuinely ambiguous utterances measures
#: coin flips; these have one defensible answer each, so a divergence is a real defect.
ROUTER_CASES = (
    ParityCase("tracking", "where is my delivery, it was due on Tuesday"),
    ParityCase("tracking_2", "I still have not received the parcel I paid for"),
    ParityCase("address", "I need to send it to a different address before it ships"),
    ParityCase("returns", "I want to send this back, it does not fit"),
    ParityCase("returns_2", "how do I return something I bought last week"),
    ParityCase("dispute", "there is a payment on my statement I did not authorise"),
    ParityCase("dispute_2", "I want to formally dispute that charge"),
    ParityCase("nonsense", "what is the weather like where you are"),
)


# ------------------------------------------------------------------ structural


def test_every_node_declares_at_least_two_variants() -> None:
    registry = load_bindings(MODELS_YAML)
    for node in registry.nodes:
        variants = registry.variants(node)
        assert len(variants) >= MIN_VARIANTS, (
            f"node {node!r} declares {len(variants)} variant(s); Rule 6 needs "
            f"{MIN_VARIANTS}. A node bound to one model can never be parity-tested"
        )


def test_a_nodes_variants_name_distinct_models() -> None:
    """Two variants pointing at the same model is parity theatre."""
    registry = load_bindings(MODELS_YAML)
    for node in registry.nodes:
        models = {registry.resolve(node, v).model for v in registry.variants(node)}
        assert len(models) >= MIN_VARIANTS, (
            f"node {node!r} declares multiple variants but only {len(models)} distinct "
            f"model(s): {sorted(models)}"
        )


def test_the_parity_gate_covers_every_node_the_graph_uses() -> None:
    """A node added to the graph without a binding would silently escape Rule 6."""
    registry = load_bindings(MODELS_YAML)
    assert "router" in registry.nodes
    assert {"task_agent", "judge", "summarizer"} <= set(registry.nodes)


# ----------------------------------------------------------------- behavioural


def _credentials_present() -> bool:
    settings = Settings()
    return settings.openrouter_api_key is not None or bool(os.environ.get("CCAS_VLLM_BASE_URL"))


async def _route(
    router: IntentRouter, case: ParityCase, _taxonomy: IntentTaxonomy | None = None
) -> VariantOutcome:
    started = time.perf_counter_ns()
    try:
        prediction = await router.classify(redacted(case.utterance), history=())
    except ProviderRateLimitedError as exc:
        # Nobody served this. Not a disagreement -- see ccas.evals.parity.
        return VariantOutcome(error=f"rate limited: {exc}"[:160], unavailable=True)
    except (LLMProviderError, ValueError, TimeoutError) as exc:
        return VariantOutcome(error=f"{type(exc).__name__}: {exc}"[:160])
    return VariantOutcome(
        decision=prediction.intent_id or "<none>",
        confidence=prediction.confidence,
        latency_ms=(time.perf_counter_ns() - started) // 1_000_000,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _credentials_present(), reason="no LLM credentials configured")
async def test_router_variants_agree_on_the_intent() -> None:
    taxonomy = IntentTaxonomy.model_validate(
        json.loads((REPO / "tests" / "fixtures" / "taxonomies" / "retail.json").read_text())
    )
    registry = load_bindings(MODELS_YAML)
    left, right = registry.variants("router")[:MIN_VARIANTS]
    settings = Settings()

    providers: dict[str, LLMProvider] = {}
    routers: dict[str, IntentRouter] = {}
    for variant in (left, right):
        binding = registry.resolve("router", variant)
        providers[variant] = build_provider(binding, settings)
        routers[variant] = IntentRouter(providers[variant], binding, taxonomy)

    outcomes = []
    try:
        for index, case in enumerate(ROUTER_CASES):
            if index:
                await asyncio.sleep(PACING_S)
            a, b = await asyncio.gather(_route(routers[left], case), _route(routers[right], case))
            outcomes.append((case, a, b))
    finally:
        for provider in providers.values():
            await provider.aclose()

    report = compare_variants(node="router", left=left, right=right, outcomes=outcomes)
    detail = "\n".join(
        f"  {d.case_id}: {d.left} vs {d.right}{f' ({d.detail})' if d.detail else ''}"
        for d in report.divergences
    )
    LOG.info(
        "parity.router",
        correlation_id="parity-router",
        left=report.left,
        right=report.right,
        cases=report.cases,
        agreement=round(report.agreement_rate, 3),
        divergent=len(report.divergences),
        unmeasured=report.skipped,
        left_p95_ms=report.left_p95_ms,
        right_p95_ms=report.right_p95_ms,
    )
    if report.inconclusive(min_cases=MIN_MEASURED_CASES):
        # A quota is not a divergence. Skipping is visible in the run summary; passing
        # would not be, and this is the gate that stops a node from depending on one model.
        pytest.skip(
            f"parity unmeasured: {report.skipped}/{len(ROUTER_CASES)} cases were rate "
            f"limited. Rule 6 is unverified for 'router' until this runs green"
        )

    assert report.meets(MIN_AGREEMENT, min_cases=MIN_MEASURED_CASES), (
        f"{report.summary()}\n{detail}\n"
        "Rule 6: fix the prompt so both models agree, or record the divergence in an ADR. "
        "Do not pin the node to one model."
    )
