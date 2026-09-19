"""Demo pipeline stages.

Each stage is either backed by a shipped module or explicitly ``PENDING`` with the
phase that delivers it. Stages never fake output -- a demo that invents a redaction it
did not perform would be worse than no demo at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ccas.config.budget import load_budget
from ccas.config.domain_loader import LoadedDomain
from ccas.config.settings import Settings
from ccas.copilot.crm.mock import MockCrmAdapter
from ccas.graph.assembly import NODE_NAMES
from ccas.ingestion.base import RawRecord, RawTurn
from ccas.ingestion.datasets import DatasetRole, roles_for
from ccas.llm.prompt import authored
from ccas.policies.base import PolicyContext
from ccas.policies.engine import PolicyEngine
from ccas.redaction.pipeline import RedactionMode
from ccas.redaction.pipeline import build_pipeline as build_redaction_pipeline
from ccas.schemas.call_log import CallLog, DatasetSource, Utterance
from ccas.schemas.common import Channel, Speaker, TraceContext, Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.handoff import HandoffContext
from ccas.schemas.pii import RedactedText
from ccas.schemas.session import LatencyLedger, Turn
from ccas.tools.registry import build_registry
from ccas.voice.stt.deepgram import DeepgramStt
from ccas.voice.tts.cartesia import CartesiaTts
from ccas.voice.vad import SileroVad

__all__ = ["Stage", "StageResult", "StageStatus", "build_pipeline", "run_pipeline"]


class StageStatus(StrEnum):
    LIVE = "live"
    PENDING = "pending"
    BLOCKED = "blocked"
    """An upstream stage is pending, so this one cannot run even in principle."""


@dataclass(slots=True)
class StageResult:
    name: str
    module: str
    status: StageStatus
    lines: list[str] = field(default_factory=list)
    phase: int | None = None
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status is StageStatus.LIVE


@dataclass(slots=True)
class Stage:
    name: str
    module: str
    run: Callable[[StageContext], StageResult]


@dataclass(slots=True)
class StageContext:
    text: str
    domain: LoadedDomain
    trace: TraceContext
    ledger: LatencyLedger
    config_dir: Path
    source: DatasetSource = DatasetSource.LIVE_CAPTURE
    mode: RedactionMode = RedactionMode.REALTIME
    redacted: RedactedText | None = None
    """Set by the redaction stage. Downstream stages read this, never ``text`` --
    which is the same rule the platform follows (CLAUDE.md Rule 2)."""

    call_log: CallLog | None = None
    """Set by the normalisation stage."""


def _roles_summary(source: DatasetSource) -> str:
    roles = roles_for(source)
    if len(roles) == len(DatasetRole):
        return "unrestricted corpus"
    return "roles: " + ", ".join(sorted(r.value for r in roles))


def _pending(name: str, module: str, phase: int, *lines: str) -> StageResult:
    return StageResult(
        name=name, module=module, status=StageStatus.PENDING, phase=phase, lines=list(lines)
    )


def _stage_pack(ctx: StageContext) -> StageResult:
    pack = ctx.domain.pack
    lines = [
        f"domain          {pack.domain}  (v{pack.version})",
        f"greeting        {pack.greeting.strip()[:72]}",
        f"tools           {', '.join(pack.tools_by_name) or 'none declared'}",
        f"queues          {', '.join(pack.queues_by_name)}",
        f"route threshold {pack.confidence.route}",
        f"compliance      {', '.join(r.value for r in pack.compliance.regimes)}",
        f"extra patterns  {len(pack.redaction_patterns)} pack-specific redaction rules",
    ]
    return StageResult("Domain pack", "config.domain_loader", StageStatus.LIVE, lines)


def _stage_redact(ctx: StageContext) -> StageResult:
    pipeline = build_redaction_pipeline(
        ctx.config_dir / "redaction_policy.yaml", pack=ctx.domain.pack, mode=ctx.mode
    )
    result = pipeline.redact(ctx.text)
    ctx.redacted = result
    ctx.ledger.redact_us = result.report.elapsed_us

    report = result.report
    budget_us = load_budget(ctx.config_dir / "latency_budget.yaml").stages.redact_ms * 1000
    verdict = "within budget" if report.elapsed_us <= budget_us else "OVER BUDGET"
    lines = [
        f"mode            {ctx.mode.value}  (engines: {', '.join(report.engines_run)})",
        f"status          {report.status.value}  "
        f"(egress {'permitted' if report.egress_permitted else 'BLOCKED'})",
        f"elapsed         {report.elapsed_us} us / {budget_us} us budget -- {verdict}",
        f"entities        {report.entity_counts or 'none found'}",
        f"pack patterns   {len(ctx.domain.pack.redaction_patterns)} layered on the global set",
        "",
        f"  in   {ctx.text}",
        f"  out  {result.text or '<withheld -- not egress-permitted>'}",
    ]
    if report.residual_patterns:
        lines.append(f"residual        {', '.join(report.residual_patterns)}")
    status = StageStatus.LIVE if report.egress_permitted else StageStatus.BLOCKED
    return StageResult("PII redaction", "redaction.pipeline", status, lines)


def _stage_normalize(ctx: StageContext) -> StageResult:
    if ctx.redacted is None or not ctx.redacted.egress_permitted:
        return StageResult(
            "Normalisation",
            "ingestion.normalizer",
            StageStatus.BLOCKED,
            ["upstream redaction did not clear; a CallLog cannot be built from this turn"],
        )

    record = RawRecord(
        source=ctx.source,
        record_id=ctx.trace.correlation_id,
        channel=Channel.VOICE,
        turns=(RawTurn(speaker=Speaker.CALLER, text=ctx.text),),
        domain_hint=ctx.domain.domain,
    )
    call_id = CallLog.make_call_id(record.source, record.record_id)

    # A CallLog is built from the already-redacted turn, which is the only thing the
    # contract will accept. Passing ctx.text here would raise, by design.
    log = CallLog(
        call_id=call_id,
        source=record.source,
        source_record_id=record.record_id,
        channel=record.channel,
        domain_hint=record.domain_hint,
        utterances=(Utterance(index=0, speaker=Speaker.CALLER, content=ctx.redacted),),
        redaction=ctx.redacted.report,
    )
    ctx.call_log = log

    lines = [
        f"call_id         {log.call_id}",
        f"source          {log.source.value}  ({_roles_summary(log.source)})",
        f"turns           {log.turn_count}   caller turns {len(log.caller_utterances)}",
        f"redaction       {log.redaction.status.value}, {len(log.redaction.spans)} spans",
        f"schema          CallLog v{log.schema_version}  -- validated, serialisable",
        "",
        "  an unredacted CallLog is not constructible; try `cli.demo gate`",
    ]
    return StageResult("Normalisation", "ingestion.normalizer", StageStatus.LIVE, lines)


def _stage_intent(ctx: StageContext) -> StageResult:
    if not ctx.domain.has_taxonomy:
        return _pending(
            "Intent detection",
            "orchestration.router",
            3,
            "taxonomy        not mined yet for this pack",
            "mine it         uv run python scripts/ingest.py --source aixblock "
            f"--domain {ctx.domain.domain}",
            f"                uv run python scripts/mine_taxonomy.py --domain {ctx.domain.domain}",
            "needs           the AIxBlock corpus in data/raw/aixblock (ADR-0004: it is",
            "                the only corpus a taxonomy may be mined from)",
        )

    taxonomy = ctx.domain.taxonomy
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
        "  routing itself lands in phase 4; this stage shows what it will route into",
    ]
    for node in sorted(taxonomy.nodes, key=lambda n: n.intent_id)[:8]:
        indent = "  " * (node.level - 1)
        lines.append(
            f"  {indent}{node.intent_id:<40} {node.volume.share_of_total:>6.1%}  "
            f"{node.automation.quadrant.value}"
        )
    if len(taxonomy.nodes) > 8:
        lines.append(f"  ... and {len(taxonomy.nodes) - 8} more")
    return StageResult(
        "Intent detection", "orchestration.router", StageStatus.PENDING, lines, phase=4
    )


def _stage_tool(ctx: StageContext) -> StageResult:
    """Show the mesh: what would route, what the policies say, what could dispatch."""
    pack = ctx.domain.pack
    registry = build_registry(pack, ctx.config_dir / "tools.yaml")
    engine = PolicyEngine()

    lines = [
        f"registry        {len(registry.names)} tools "
        f"({len(registry.global_names)} shared, "
        f"{len(registry.names) - len(registry.global_names)} from the pack)",
        f"policy order    {' -> '.join(engine.names)}",
        f"graph nodes     {' -> '.join(NODE_NAMES)}",
        "",
    ]
    for name in registry.names:
        spec = registry.resolve(name)
        required = spec.input_schema.get("required")
        params = ", ".join(str(r) for r in required) if isinstance(required, list) else "-"
        scope = "shared" if registry.is_global(name) else "pack  "
        lines.append(
            f"  {scope} {name:<24} verify={spec.requires_verification.value:<7} "
            f"risk={spec.risk_tier.value:<7} args=[{params}]"
        )

    if not ctx.domain.has_taxonomy:
        lines += [
            "",
            "  no taxonomy: the router has nothing to classify into, so the graph",
            "  escalates every turn. Mine one to see routing and dispatch.",
        ]
        return StageResult("Agentic mesh", "graph.assembly", StageStatus.PENDING, lines, phase=3)

    taxonomy = ctx.domain.taxonomy
    lines += ["", f"  {len(taxonomy.leaves())} routable intents:"]
    for leaf in taxonomy.leaves():
        policy_ctx = PolicyContext.for_intent(pack, leaf)
        threshold = max(policy_ctx.confidence.route, policy_ctx.escalation.min_intent_confidence)
        gate = (
            "auto-escalates"
            if policy_ctx.escalation.auto_escalate
            else f"needs conf >= {threshold:.2f}"
        )
        lines.append(
            f"    {leaf.intent_id:<34} {leaf.risk_tier.value:<10} {gate}; "
            f"tools={list(leaf.required_tools) or 'none'}"
        )
    lines += [
        "",
        "  routing a live utterance runs in the graph, not here -- see",
        "  tests/integration/test_graph_e2e.py for the scripted journeys",
    ]
    return StageResult("Agentic mesh", "graph.assembly", StageStatus.LIVE, lines)


def _stage_voice(ctx: StageContext) -> StageResult:
    """Show the voice stack: which driver, which engines, and what is missing."""
    settings = Settings()
    vad = SileroVad()
    stt = DeepgramStt(
        settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else None
    )
    tts = CartesiaTts(
        settings.cartesia_api_key.get_secret_value() if settings.cartesia_api_key else None,
        voice_id=settings.cartesia_voice_id,
    )

    def mark(ok: bool) -> str:
        return "ready" if ok else "NOT CONFIGURED"

    lines = [
        f"transport       livekit ({mark(bool(settings.livekit_url))}) | scripted (ready)",
        f"vad             silero ({mark(vad.available)}) | energy (ready)",
        f"stt             deepgram nova-3 ({mark(stt.available)}) | scripted (ready)",
        f"tts             cartesia sonic ({mark(tts.available)}) | scripted (ready)",
        "barge-in        clear playback -> cancel tts -> cancel turn",
        "dtmf            terminator, max-length and inter-digit timeout",
        "",
        "  the scripted driver runs the same turn loop with no media server:",
        "    uv run python scripts/bench_latency.py",
        "    uv run pytest tests/latency/test_e2e_rtt.py",
        "",
        "  a real call needs credentials; check with:",
        "    uv run python -m ccas.voice.worker preflight",
    ]
    live = vad.available and stt.available and tts.available and bool(settings.livekit_url)
    status = StageStatus.LIVE if live else StageStatus.PENDING
    return StageResult("Voice engine", "voice.session", status, lines, phase=None if live else 5)


def _stage_copilot(ctx: StageContext) -> StageResult:
    """Build the CTI payload and push it, so the last hop out is visible.

    This is the only stage whose failure mode is a privacy breach rather than a wrong
    answer, so what it demonstrates is the refusal: ``HandoffContext`` will not construct
    around unredacted text, and ``attached_data`` will not unwrap it.

    The demo does not run the graph, so there is no routed session and no policy verdict
    behind this handoff -- it is built from this turn alone, and the output says so. What
    is real is the contract, the redaction gate and the adapter; none of it is mocked out.
    """
    if ctx.redacted is None:
        return StageResult(
            "Agent copilot",
            "copilot.crm",
            StageStatus.BLOCKED,
            ["redaction did not run, so there is nothing that may leave the process"],
            phase=6,
        )

    pack = ctx.domain.pack
    queue = pack.default_queue
    entities = ctx.redacted.report.entity_counts
    summary = authored(
        f"Caller reached a handoff after 1 turn on the {ctx.domain.domain} pack. "
        f"Intent: not identified (the demo does not route). "
        f"Redaction replaced {sum(entities.values())} entity value(s)."
    )

    handoff = HandoffContext(
        handoff_id=f"{ctx.trace.correlation_id[:12]}-handoff",
        session_id=ctx.trace.correlation_id,
        trace=ctx.trace,
        domain=ctx.domain.domain,
        reason=HandoffReason.UNSUPPORTED_INTENT,
        urgency=Urgency.NORMAL,
        target_queue=queue.name,
        required_skills=queue.skills,
        intent_confidence=0.0,
        summary=summary,
        transcript=(Turn(index=0, speaker=Speaker.CALLER, content=ctx.redacted),),
        cti_attributes={"triggered_by": authored("demo"), "turns": authored("1")},
    )

    adapter = MockCrmAdapter()
    record = asyncio.run(adapter.push(handoff))

    lines = [
        f"handoff        {handoff.handoff_id}  (HandoffContext v{handoff.schema_version})",
        f"reason         {handoff.reason.value}, urgency {handoff.urgency.value}",
        f"queue          {handoff.target_queue}  skills={list(handoff.required_skills) or '-'}",
        f"transcript     {len(handoff.transcript)} turn(s), all egress-permitted",
        f"crm            {record.system} -> {record.record_id}",
        f"attached data  {len(record.attributes)} flat key/value pairs",
        "",
    ]
    for key in sorted(record.attributes):
        lines.append(f"    {key:<22} {record.attributes[key]}")
    lines += [
        "",
        "  the summary and transcript are deliberately NOT attached data -- a vendor",
        "  retains that for the life of the interaction (copilot/crm/base.py)",
        "",
        "  no routed session behind this one: the demo builds the payload from this",
        "  turn alone. The contract, the redaction gate and the adapter are all real;",
        "  try `cli.demo gate` to watch the same gate refuse.",
    ]
    return StageResult("Agent copilot", "copilot.crm", StageStatus.LIVE, lines)


def _stage_latency(ctx: StageContext) -> StageResult:
    budget = load_budget(ctx.config_dir / "latency_budget.yaml")
    stages = budget.stages.as_dict()
    lines = [f"{name:<16}{value:>5} ms" for name, value in stages.items()]
    lines.append(f"{'declared total':<16}{budget.stages.total_ms:>5} ms")
    lines.append(f"{'ceiling':<16}{budget.budget_ms:>5} ms (p{budget.percentile})")
    lines.append(f"{'slack':<16}{budget.slack_ms:>5} ms")
    lines.append(f"{'observed so far':<16}{ctx.ledger.total_rtt_ms:>5} ms")
    return StageResult("Latency budget", "config.budget", StageStatus.LIVE, lines)


def build_pipeline() -> tuple[Stage, ...]:
    return (
        Stage("Domain pack", "config.domain_loader", _stage_pack),
        Stage("PII redaction", "redaction.pipeline", _stage_redact),
        Stage("Normalisation", "ingestion.normalizer", _stage_normalize),
        Stage("Intent detection", "orchestration.router", _stage_intent),
        Stage("Agentic mesh", "graph.assembly", _stage_tool),
        Stage("Voice engine", "voice.session", _stage_voice),
        Stage("Agent copilot", "copilot.crm", _stage_copilot),
        Stage("Latency budget", "config.budget", _stage_latency),
    )


def run_pipeline(ctx: StageContext) -> list[StageResult]:
    return [stage.run(ctx) for stage in build_pipeline()]
