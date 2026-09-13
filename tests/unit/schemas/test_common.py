from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas.common import Frozen, TraceContext, VerificationLevel, utcnow


class _Model(Frozen):
    name: str


def test_frozen_models_reject_mutation() -> None:
    model = _Model(name="a")
    with pytest.raises(ValidationError):
        model.name = "b"


def test_frozen_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _Model(name="a", surprise="b")  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["retail", "billing.refund.status", "get_order_status", "a1"])
def test_slug_accepts_valid_identifiers(value: str) -> None:
    class _Slugged(Frozen):
        slug: str

    assert _Slugged(slug=value).slug == value


@pytest.mark.parametrize(
    "bad", ["Retail", "with space", ".leading", "trailing.", "double..dot", ""]
)
def test_trace_tenant_id_rejects_non_slugs(bad: str) -> None:
    with pytest.raises(ValidationError):
        TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id="c", tenant_id=bad)


@pytest.mark.parametrize(("trace_id", "span_id"), [("0" * 31, "1" * 16), ("0" * 32, "1" * 15)])
def test_trace_ids_must_be_w3c_shaped(trace_id: str, span_id: str) -> None:
    with pytest.raises(ValidationError):
        TraceContext(trace_id=trace_id, span_id=span_id, correlation_id="c")


def test_verification_levels_are_ordered() -> None:
    assert VerificationLevel.STRONG.satisfies(VerificationLevel.SOFT)
    assert VerificationLevel.STRONG.satisfies(VerificationLevel.STRONG)
    assert not VerificationLevel.SOFT.satisfies(VerificationLevel.STEP_UP)
    assert VerificationLevel.NONE.satisfies(VerificationLevel.NONE)


def test_utcnow_is_timezone_aware() -> None:
    assert utcnow().tzinfo is not None
