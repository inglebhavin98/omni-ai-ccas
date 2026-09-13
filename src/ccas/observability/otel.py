"""OpenTelemetry bootstrap.

Kept separate from ``tracing`` so that importing a tracer never has the side effect of
configuring a global provider -- tests and CLIs need the former without the latter.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

__all__ = ["configure_tracing", "is_configured"]

_configured = False


def is_configured() -> bool:
    return _configured


def configure_tracing(
    service_name: str = "omni-ai-ccas",
    exporter: SpanExporter | None = None,
    *,
    force: bool = False,
) -> TracerProvider:
    """Install a global tracer provider. Idempotent unless ``force`` is set."""
    global _configured

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    if not _configured or force:
        trace.set_tracer_provider(provider)
        _configured = True
    return provider
