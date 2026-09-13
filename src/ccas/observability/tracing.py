"""Trace context plumbing.

``TraceContext`` travels on every payload that crosses a boundary, so a redacted
transcript in an agent desktop can still be joined to the spans that produced it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span

from ccas.schemas.common import TraceContext

__all__ = ["current_trace_context", "new_correlation_id", "session_span", "tracer"]

_INVALID_TRACE_ID = 0


def tracer(name: str = "ccas") -> trace.Tracer:
    return trace.get_tracer(name)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def current_trace_context(
    correlation_id: str | None = None, tenant_id: str = "default"
) -> TraceContext:
    """Snapshot the active span as a serializable context.

    Falls back to a freshly generated id pair when no span is active, so a payload is
    always traceable even if it originated outside an instrumented path.
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.trace_id == _INVALID_TRACE_ID:
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
    else:
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        correlation_id=correlation_id or new_correlation_id(),
        tenant_id=tenant_id,
    )


@contextmanager
def session_span(name: str, ctx: TraceContext, **attributes: str | int | float) -> Iterator[Span]:
    """Open a span carrying the identifiers every ccas log line is keyed by.

    Attribute values are caller-supplied and must already be redacted -- never pass an
    utterance, a slot value or a raw caller reference (CLAUDE.md Rule 2).
    """
    with tracer().start_as_current_span(name) as span:
        span.set_attribute("ccas.correlation_id", ctx.correlation_id)
        span.set_attribute("ccas.tenant_id", ctx.tenant_id)
        for key, value in attributes.items():
            span.set_attribute(f"ccas.{key}", value)
        yield span
