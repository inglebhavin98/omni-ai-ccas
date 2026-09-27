"""LLM-backed demo stages: live routing and the M6b judge.

Split out of ``stages.py`` rather than growing it past the 400-line cap (Rule 8). Both
stages call a real provider when one is configured and both refuse loudly when one is
not -- a stage that invented a routing or a verdict would be worse than a pending badge
(CLAUDE.md Rule 11). Each costs one LLM call, and the output says so.
"""

from __future__ import annotations

import asyncio

from ccas.config.settings import Settings
from ccas.evals.judge import SessionJudge
from ccas.graph.router import IntentRouter
from ccas.llm.base import LLMProvider, LLMProviderError
from ccas.llm.bindings import load_bindings
from ccas.llm.factory import build_provider
from ccas.observability.logging import get_logger
from ccas.schemas.common import Channel, Speaker
from ccas.schemas.eval import JudgeVerdict
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import (
    CallerContext,
    IntentPrediction,
    LatencyLedger,
    SessionState,
    Turn,
)
from ccas.schemas.taxonomy import IntentTaxonomy
from cli.stages import StageContext, StageResult, StageStatus

__all__ = ["DEMO_RULES", "intent_stage", "judge_stage", "route_live"]

LOG = get_logger("cli.stages_llm")

#: Authored and generic (Rule 1): the judge grades against the rules the exchange ran
#: under, and the demo's rules are the platform's own, not a vertical's.
DEMO_RULES = (
    "Be concise. Answer only from the tool results; escalate when the data is missing. "
    "Never state a value the record does not contain."
)


def intent_stage(ctx: StageContext) -> StageResult:
    """The taxonomy summary, then one live router call against the redacted turn.

    Without a mined taxonomy the stage names its phase and the command that delivers it,
    exactly as it always has. With one, the routing is real: one LLM call, and a pending
    badge rather than an invented classification when nobody served it.
    """
    if not ctx.domain.has_taxonomy:
        return StageResult(
            "Intent detection",
            "orchestration.router",
            StageStatus.PENDING,
            [
                "taxonomy        not mined yet for this pack",
                "mine it         uv run python scripts/ingest.py --source aixblock "
                f"--domain {ctx.domain.domain}",
                "                uv run python scripts/mine_taxonomy.py "
                f"--domain {ctx.domain.domain}",
                "needs           the AIxBlock corpus in data/raw/aixblock (ADR-0004: it is",
                "                the only corpus a taxonomy may be mined from)",
            ],
            phase=3,
        )

    taxonomy = ctx.domain.taxonomy
    assert taxonomy is not None
    leaves = taxonomy.leaves()
    quadrants: dict[str, int] = {}
    for node in leaves:
        key = node.automation.quadrant.value
        quadrants[key] = quadrants.get(key, 0) + 1

    lines = [
        f"taxonomy        {taxonomy.taxonomy_id}  v{taxonomy.version}",
        f"mined by        {taxonomy.embedding_model} -> {taxonomy.clusterer} "
        f"-> {taxonomy.labeler_model}",
        f"nodes           {len(taxonomy.nodes)}  ({len(leaves)} leaves)",
        (
            f"coverage        {taxonomy.coverage:.1%}  (noise {taxonomy.noise_ratio:.1%})"
            if taxonomy.coverage is not None
            else "coverage        n/a (adopted, nothing was clustered)"
        ),
        f"quadrants       {quadrants}",
        f"route threshold {ctx.domain.pack.confidence.route}",
        "",
    ]
    for node in sorted(taxonomy.nodes, key=lambda n: n.intent_id)[:8]:
        indent = "  " * (node.level - 1)
        lines.append(
            f"  {indent}{node.intent_id:<40} {node.volume.share_of_total:>6.1%}  "
            f"{node.automation.quadrant.value}"
        )
    if len(taxonomy.nodes) > 8:
        lines.append(f"  ... and {len(taxonomy.nodes) - 8} more")

    status, routed = route_live(ctx, taxonomy)
    lines += ["", *routed]
    # A pending stage names the phase that delivers it (Rule 11): routing not served is
    # phase 3's gap, exactly like a taxonomy that has not been mined.
    phase = 3 if status is StageStatus.PENDING else None
    return StageResult("Intent detection", "orchestration.router", status, lines, phase=phase)


async def _classify_once(
    provider: LLMProvider, router: IntentRouter, text: RedactedText
) -> tuple[IntentPrediction | None, str | None]:
    """One classification, then the provider's connections are released."""
    try:
        return await router.classify(text), None
    except LLMProviderError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        await provider.aclose()


def route_live(
    ctx: StageContext, taxonomy: IntentTaxonomy, settings: Settings | None = None
) -> tuple[StageStatus, list[str]]:
    """Classify the redacted turn through the pack taxonomy. One router call.

    Returns the stage status and the lines to append: a prediction with the policy band
    it lands in, or the reason nothing was predicted. Never fabricates a classification.
    """
    settings = settings or Settings()
    if ctx.redacted is None:
        return StageStatus.BLOCKED, ["redaction did not run; nothing may be routed"]

    bindings = load_bindings(settings.models_config)
    binding = bindings.resolve("router")
    try:
        provider = build_provider(binding, settings)
    except LLMProviderError as exc:
        return StageStatus.PENDING, [
            f"router          not configured ({exc})",
            "cost            0 calls this stage",
        ]

    router = IntentRouter(provider, binding, taxonomy)
    prediction, error = asyncio.run(_classify_once(provider, router, ctx.redacted))
    if prediction is None:
        rate_limited = isinstance(error, str) and error.startswith("ProviderRateLimitedError")
        return StageStatus.PENDING, [
            "router          not served ("
            + ("quota" if rate_limited else "provider error")
            + ") -- no classification is recorded, none is invented",
            "cost            0 calls this stage",
        ]

    threshold = ctx.domain.pack.confidence.route
    floor = ctx.domain.pack.confidence.clarify_floor
    intent_id = prediction.intent_id
    confidence = prediction.confidence
    band = (
        "route"
        if intent_id and confidence >= threshold
        else "clarify"
        if confidence >= floor
        else "escalate"
    )
    lines = [
        f"prediction      {intent_id or '<none>'}  confidence {confidence:.2f} "
        f"({prediction.source}, {prediction.latency_ms} ms)"
        if intent_id
        else f"prediction      <none>  confidence {confidence:.2f} -- below every band",
        f"bands           route>={threshold:.2f}  clarify>={floor:.2f}  -> {band}",
        "cost            1 router call (live)",
    ]
    return StageStatus.LIVE, lines


def judge_stage(ctx: StageContext, settings: Settings | None = None) -> StageResult:
    """Grade the demo exchange with the M6b judge. One judge call.

    The demo exchange is the single caller turn -- the demo does not run the graph, so
    there is no assistant reply, and the judge scores that honestly: task success is
    expected at the bottom of its rubric because nothing was completed. That is the
    demonstration -- a judge asked a real question, not a rubber stamp.
    """
    settings = settings or Settings()
    if ctx.redacted is None:
        return StageResult(
            "Judge",
            "evals.judge",
            StageStatus.BLOCKED,
            ["redaction did not run, so there is nothing that may be judged"],
            phase=6,
        )

    state = _demo_session(ctx, ctx.redacted)

    async def _once() -> tuple[JudgeVerdict | None, str | None]:
        judge: SessionJudge | None = None
        try:
            judge = SessionJudge(load_bindings(settings.models_config), settings)
            return await judge.judge_session(DEMO_RULES, state, sampled=False), None
        except LLMProviderError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            if judge is not None:
                await judge.aclose()

    verdict, error = asyncio.run(_once())
    if verdict is None:
        rate_limited = isinstance(error, str) and error.startswith("ProviderRateLimitedError")
        return StageResult(
            "Judge",
            "evals.judge",
            StageStatus.PENDING,
            [
                "not served      ("
                + ("quota" if rate_limited else "provider error")
                + ") -- unmeasured, not a zero",
                "the judge leaves the case out of the denominator rather than scoring it",
            ],
            phase=6,
        )

    lines = [
        f"judge           {verdict.judge_provider.value}:{verdict.judge_model}",
        "exchange        1 caller turn, no assistant reply -- task success is expected",
        "                at the bottom of its rubric, because nothing was completed",
        "",
    ]
    for score in verdict.scores:
        lines.append(
            f"  {score.dimension.value:<18} {score.score:>5.2f}  "
            f"{'PASS' if score.passed else 'FAIL'}  {score.rationale[:84]}"
        )
    lines += [
        "",
        "verdict         "
        + ("passed" if verdict.passed else "failed")
        + " at the default pass mark 0.75 -- uncalibrated (6.12)",
        "cost            1 judge call (live)",
    ]
    return StageResult("Judge", "evals.judge", StageStatus.LIVE, lines)


def _demo_session(ctx: StageContext, redacted: RedactedText) -> SessionState:
    """The demo exchange as a SessionState: one redacted caller turn, no reply.

    ``redacted`` arrives as a parameter rather than being read off ``ctx`` because the
    caller's guard has already refused a ``None`` and --strict tracks that narrowing no
    further than the call boundary.
    """
    correlation = ctx.trace.correlation_id
    return SessionState(
        session_id=correlation,
        trace=ctx.trace,
        domain=ctx.domain.domain,
        channel=Channel.CHAT,
        caller=CallerContext(caller_ref=correlation.ljust(8, "0")[:32]),
        turns=[Turn(index=0, speaker=Speaker.CALLER, content=redacted)],
        latency=LatencyLedger(),
    )
