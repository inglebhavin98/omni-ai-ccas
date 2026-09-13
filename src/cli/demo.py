"""Manual review console for ai-ccas.

Run a transcript through the pipeline and see exactly what each module does -- and,
for modules that are not built yet, exactly which phase delivers them. Every run also
appends structured JSON events to ``logs/execution.log``.

    uv run python -m cli.demo pipeline "where is my delivery"
    uv run python -m cli.demo pipeline            # interactive
    uv run python -m cli.demo pack healthcare
    uv run python -m cli.demo datasets
    uv run python -m cli.demo budget
    uv run python -m cli.demo gate               # show the Rule 2 egress gate refusing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ccas.config.domain_loader import DomainPackNotFoundError, available_domains, load_domain
from ccas.config.settings import Settings
from ccas.ingestion.datasets import DATASET_ROLES, DatasetRole, require_role
from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import current_trace_context
from ccas.schemas.call_log import CallLog, DatasetSource, Utterance
from ccas.schemas.common import Channel, Speaker, Urgency
from ccas.schemas.escalation import HandoffReason
from ccas.schemas.handoff import HandoffContext
from ccas.schemas.pii import RedactedText, RedactionReport, RedactionStatus, sha256_hex
from ccas.schemas.session import LatencyLedger
from cli.stages import StageContext, StageStatus, run_pipeline

RULE = "─" * 78
LOG = get_logger("cli.demo")


def _first_error(exc: Exception) -> str:
    """Pydantic wraps validator messages; show only the message we wrote."""
    for line in str(exc).splitlines():
        stripped = line.strip()
        if stripped.startswith("Value error,"):
            return stripped.removeprefix("Value error,").split(" [type=")[0].strip()
    return str(exc).splitlines()[0].strip()


def _hr(title: str = "") -> None:
    print(f"\n{RULE}" if not title else f"\n{title}\n{RULE}")


def _badge(status: StageStatus, phase: int | None) -> str:
    if status is StageStatus.LIVE:
        return "[LIVE]   "
    return f"[PHASE {phase}]" if phase else "[PENDING]"


def cmd_pipeline(args: argparse.Namespace, settings: Settings) -> int:
    text = args.text or _read_interactive()
    if not text:
        print("no input", file=sys.stderr)
        return 1

    try:
        domain = load_domain(settings.domains_dir, args.domain)
    except DomainPackNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    trace = current_trace_context(tenant_id=args.tenant)
    ctx = StageContext(
        text=text,
        domain=domain,
        trace=trace,
        ledger=LatencyLedger(),
        config_dir=settings.config_dir,
        source=DatasetSource(args.source),
    )

    LOG.info(
        "demo.pipeline.start",
        correlation_id=trace.correlation_id,
        domain=domain.domain,
        source=args.source,
        input_chars=len(text),
    )

    _hr(f"ai-ccas pipeline  ::  domain={domain.domain}  corr={trace.correlation_id[:12]}")
    started = time.perf_counter()
    results = run_pipeline(ctx)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    for result in results:
        print(f"\n{_badge(result.status, result.phase)} {result.name}  ({result.module})")
        for line in result.lines:
            print(f"          {line}")
        LOG.info(
            "demo.stage",
            correlation_id=trace.correlation_id,
            stage=result.name,
            module=result.module,
            status=result.status.value,
            phase=result.phase,
        )

    live = sum(1 for r in results if r.ok)
    _hr()
    print(f"{live}/{len(results)} stages live  ::  wall {elapsed_ms} ms")
    print("pending stages name the phase that delivers them; none of them fabricate output.")
    LOG.info(
        "demo.pipeline.end",
        correlation_id=trace.correlation_id,
        stages_live=live,
        stages_total=len(results),
        wall_ms=elapsed_ms,
    )
    return 0


def _read_interactive() -> str:
    print("Enter a transcript line (blank to cancel):")
    try:
        return input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def cmd_pack(args: argparse.Namespace, settings: Settings) -> int:
    try:
        loaded = load_domain(settings.domains_dir, args.domain)
    except DomainPackNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    pack = loaded.pack

    _hr(f"domain pack :: {pack.domain} v{pack.version}  ({pack.display_name})")
    print(f"locales            {', '.join(pack.locales)}")
    print(f"max turns          {pack.max_turns}")
    print(f"taxonomy           {'loaded' if loaded.has_taxonomy else 'not mined yet'}")
    print(
        f"confidence         route={pack.confidence.route} "
        f"clarify_floor={pack.confidence.clarify_floor} "
        f"slot_accept={pack.confidence.slot_accept}"
    )
    print(
        f"compliance         {', '.join(r.value for r in pack.compliance.regimes)} "
        f"(transcripts {pack.compliance.transcript_retention_days}d, "
        f"audio {pack.compliance.audio_retention_days}d)"
    )

    _hr("queues")
    for queue in pack.queues:
        print(
            f"  {queue.name:<14} {queue.display_name:<24} "
            f"min_verification={queue.min_verification.value:<7} "
            f"max_risk={queue.max_risk_tier.value}"
        )

    _hr("tools")
    for tool in pack.tools:
        print(f"  {tool.name}")
        print(f"    {tool.description}")
        print(
            f"    verify={tool.requires_verification.value}  risk={tool.risk_tier.value}  "
            f"timeout={tool.timeout_ms}ms  side_effecting={tool.side_effecting}  "
            f"idempotency_key={'required' if tool.requires_idempotency_key else 'no'}"
        )
    if not pack.tools:
        print("  (none declared)")

    _hr("pack redaction patterns")
    for pattern in pack.redaction_patterns:
        print(f"  {pattern.name:<22} -> {pattern.entity_type.value:<14} score={pattern.score}")
    if not pack.redaction_patterns:
        print("  (none declared)")

    LOG.info("demo.pack", domain=pack.domain, tools=len(pack.tools), queues=len(pack.queues))
    return 0


def cmd_datasets(_args: argparse.Namespace, _settings: Settings) -> int:
    _hr("dataset roles  ::  src/ccas/ingestion/datasets.py")
    print("Corpora are not interchangeable. require_role() fails a stage that misuses one.\n")
    for source, roles in DATASET_ROLES.items():
        if source in {
            DatasetSource.SYNTHETIC,
            DatasetSource.GENERIC_CSV,
            DatasetSource.LIVE_CAPTURE,
        }:
            continue
        print(f"  data/raw/{source.value}")
        for role in sorted(roles):
            print(f"      + {role.value}")
        denied = sorted(set(DatasetRole) - roles)
        print(f"      - denied: {', '.join(r.value for r in denied)}\n")

    _hr("live check")
    try:
        require_role(DatasetSource.BITEXT, DatasetRole.INTENT_MINING)
    except ValueError as exc:
        print("  require_role(bitext, intent_mining) -> refused")
        print(f"    {exc}")
    require_role(DatasetSource.AIXBLOCK, DatasetRole.INTENT_MINING)
    print("  require_role(aixblock, intent_mining) -> permitted")
    return 0


def cmd_budget(_args: argparse.Namespace, settings: Settings) -> int:
    from ccas.config.budget import load_budget
    from ccas.llm.bindings import load_bindings

    budget = load_budget(settings.latency_budget_config)
    _hr(f"latency budget  ::  {budget.budget_ms} ms ceiling at p{budget.percentile}")
    for name, value in budget.stages.as_dict().items():
        bar = "#" * max(1, round(value / 5))
        print(f"  {name:<14}{value:>5} ms  {bar}")
    print(f"\n  {'total':<14}{budget.stages.total_ms:>5} ms   slack {budget.slack_ms} ms")

    _hr("model bindings  ::  configs/models.yaml")
    registry = load_bindings(settings.models_config)
    print(f"  default provider  {registry.default_provider.value}")
    for node in registry.nodes:
        print(f"\n  {node}")
        for variant in registry.variants(node):
            binding = registry.resolve(node, variant)
            budget_ms = binding.latency_budget_ms
            marks = [binding.structured_mode.value]
            if binding.reasoning:
                marks.append("reasoning")
            if budget_ms:
                marks.append(f"ttft<={budget_ms}ms")
            print(
                f"    {variant:<20} {binding.provider.value}:{binding.model:<40} {', '.join(marks)}"
            )
    return 0


def cmd_gate(_args: argparse.Namespace, _settings: Settings) -> int:
    """Demonstrate the Rule 2 egress gate refusing to build a payload."""
    _hr("zero-leakage egress gate  ::  CLAUDE.md Rule 2")
    raw = "my card is 4111 1111 1111 1111"
    unverified = RedactedText(
        text=raw,
        report=RedactionReport.unverified(
            policy_version="demo", reason="presidio_onnx_model_missing"
        ),
        source_sha256=sha256_hex(raw),
    )
    print(f"  redaction status     {unverified.report.status.value}")
    print(f"  egress permitted     {unverified.egress_permitted}")

    print("\n  attempting RedactedText.require_egress() ...")
    try:
        unverified.require_egress()
    except PermissionError as exc:
        print(f"    refused: {exc}")

    print("\n  attempting to build a CallLog from it ...")
    try:
        CallLog(
            call_id=CallLog.make_call_id(DatasetSource.SYNTHETIC, "demo"),
            source=DatasetSource.SYNTHETIC,
            source_record_id="demo",
            channel=Channel.VOICE,
            utterances=(Utterance(index=0, speaker=Speaker.CALLER, content=unverified),),
            redaction=RedactionReport.clean(
                engines_run=("regex",), policy_version="demo", elapsed_us=1
            ),
        )
    except ValueError as exc:
        print(f"    refused: {_first_error(exc)}")

    print("\n  attempting to build a HandoffContext from it ...")
    try:
        HandoffContext(
            handoff_id="h-demo",
            session_id="s-demo",
            trace=current_trace_context(),
            domain="retail",
            reason=HandoffReason.SYSTEM_ERROR,
            urgency=Urgency.NORMAL,
            target_queue="tier-1",
            intent_confidence=0.0,
            summary=unverified,
        )
    except ValueError as exc:
        print(f"    refused: {_first_error(exc)}")

    print("\n  The gate is structural: there is no code path that emits unredacted text.")
    LOG.info("demo.gate", outcome="all_refused", status=RedactionStatus.UNVERIFIED.value)
    return 0


def build_parser(default_domain: str, domains: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.demo", description="ai-ccas manual review console")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--quiet-log", action="store_true", help="do not echo logs to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    choices = list(domains) or [default_domain]

    run = sub.add_parser("pipeline", help="run a transcript through every stage")
    run.add_argument("text", nargs="?", help="transcript line; omit for interactive input")
    run.add_argument("--domain", default=default_domain, choices=choices)
    run.add_argument("--tenant", default="demo")
    run.add_argument(
        "--source",
        default=DatasetSource.LIVE_CAPTURE.value,
        choices=[s.value for s in DatasetSource],
    )
    run.set_defaults(func=cmd_pipeline)

    pack = sub.add_parser("pack", help="inspect a loaded domain pack")
    pack.add_argument("domain", nargs="?", default=default_domain, choices=choices)
    pack.set_defaults(func=cmd_pack)

    sub.add_parser("datasets", help="show the dataset role matrix").set_defaults(func=cmd_datasets)
    sub.add_parser("budget", help="show latency budget and model bindings").set_defaults(
        func=cmd_budget
    )
    sub.add_parser("gate", help="demonstrate the zero-leakage egress gate").set_defaults(
        func=cmd_gate
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    domains = available_domains(settings.domains_dir)
    args = build_parser(settings.default_domain, domains).parse_args(argv)
    configure_logging(Path("logs/execution.log"), level=args.log_level, console=not args.quiet_log)
    result: int = args.func(args, settings)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
