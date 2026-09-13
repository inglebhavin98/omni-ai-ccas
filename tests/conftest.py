from __future__ import annotations

import pytest

from ccas.schemas import CallerContext, Channel, RedactionReport, SessionState, TraceContext
from tests.factories import make_report, sha256_hex


@pytest.fixture
def clean_report() -> RedactionReport:
    return make_report()


@pytest.fixture
def trace() -> TraceContext:
    return TraceContext(
        trace_id="0" * 32, span_id="1" * 16, correlation_id="corr-1", tenant_id="acme"
    )


@pytest.fixture
def caller() -> CallerContext:
    return CallerContext(caller_ref=sha256_hex("+15550100")[:32])


@pytest.fixture
def session(trace: TraceContext, caller: CallerContext) -> SessionState:
    return SessionState(
        session_id="sess-1",
        trace=trace,
        domain="retail",
        channel=Channel.VOICE,
        caller=caller,
    )
