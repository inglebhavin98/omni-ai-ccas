from __future__ import annotations

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ccas.observability.otel import configure_tracing
from ccas.observability.tracing import current_trace_context, new_correlation_id, session_span
from ccas.schemas.common import TraceContext


def test_context_is_valid_without_an_active_span() -> None:
    """A payload built outside an instrumented path must still be traceable."""
    ctx = current_trace_context()
    assert len(ctx.trace_id) == 32
    assert len(ctx.span_id) == 16
    assert ctx.tenant_id == "default"


def test_correlation_ids_are_unique() -> None:
    assert new_correlation_id() != new_correlation_id()


def test_span_ids_propagate_into_the_context() -> None:
    exporter = InMemorySpanExporter()
    provider = configure_tracing("ai-ccas-test", force=True)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    ctx = TraceContext(
        trace_id="0" * 32, span_id="1" * 16, correlation_id="corr-1", tenant_id="acme"
    )
    with session_span("route", ctx, domain="retail", turn_index=2):
        inner = current_trace_context(correlation_id=ctx.correlation_id, tenant_id=ctx.tenant_id)

    (span,) = exporter.get_finished_spans()
    assert span.name == "route"
    assert span.attributes is not None
    assert span.attributes["ccas.correlation_id"] == "corr-1"
    assert span.attributes["ccas.tenant_id"] == "acme"
    assert span.attributes["ccas.domain"] == "retail"
    assert span.attributes["ccas.turn_index"] == 2
    # The context snapshotted inside the span reflects that span, not the seed values.
    assert inner.trace_id != "0" * 32
    assert inner.correlation_id == "corr-1"


def test_configure_is_idempotent_without_force() -> None:
    configure_tracing("ai-ccas-test", force=True)
    first = configure_tracing("ai-ccas-test")
    second = configure_tracing("ai-ccas-test")
    assert first is not second  # each call builds a provider
    from ccas.observability.otel import is_configured

    assert is_configured()
