from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ccas.config.domain_loader import load_pack
from ccas.schemas.common import TraceContext, VerificationLevel
from ccas.schemas.tools import ToolPayload, ToolStatus
from ccas.tools.mock_backend import MockToolBackend, ToolFixture, ToolFixtureSet
from ccas.tools.registry import build_registry

REPO = Path(__file__).resolve().parents[3]
DOMAINS = REPO / "domains"
GLOBAL = REPO / "configs" / "tools.yaml"
TRACE = TraceContext(trace_id="0" * 32, span_id="1" * 16, correlation_id="c")


def payload(tool: str, arguments: dict[str, object]) -> ToolPayload:
    return ToolPayload(
        tool_call_id="tc-1",
        tool_name=tool,
        session_id="s1",
        arguments=arguments,  # type: ignore[arg-type]
        timeout_ms=1000,
        verification_level=VerificationLevel.STRONG,
        trace=TRACE,
    )


@pytest.mark.parametrize("domain", ["retail", "healthcare"])
def test_both_packs_ship_loadable_fixtures(domain: str) -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / domain)
    assert backend.fixtures.tools


@pytest.mark.parametrize("domain", ["retail", "healthcare"])
def test_every_fixture_names_a_registered_tool(domain: str) -> None:
    """A fixture for a tool nobody declared is dead weight that looks alive."""
    pack = load_pack(DOMAINS, domain)
    registry = build_registry(pack, GLOBAL)
    backend = MockToolBackend.from_pack_dir(DOMAINS / domain)
    for tool_name in backend.fixtures.tools:
        assert registry.has(tool_name), f"{domain}: fixture for unregistered {tool_name!r}"


async def test_a_matching_fixture_is_selected() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    spec = build_registry(load_pack(DOMAINS, "retail"), GLOBAL).resolve("get_order_status")
    response = await backend.invoke(
        spec, payload("get_order_status", {"order_reference": "ORD-404404"})
    )
    assert response.status is ToolStatus.NOT_FOUND


async def test_the_default_fixture_is_the_one_without_a_matcher() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    spec = build_registry(load_pack(DOMAINS, "retail"), GLOBAL).resolve("get_order_status")
    response = await backend.invoke(
        spec, payload("get_order_status", {"order_reference": "ORD-000777"})
    )
    assert response.ok
    assert response.data["status"] == "in_transit"


async def test_an_unfixtured_tool_still_answers_deterministically() -> None:
    """A newly declared tool must be exercisable before anyone writes fixtures."""
    backend = MockToolBackend()
    spec = build_registry(load_pack(DOMAINS, "retail"), GLOBAL).resolve("send_link")
    args = {"purpose": "reset", "channel": "sms"}
    first = await backend.invoke(spec, payload("send_link", args))
    second = await backend.invoke(spec, payload("send_link", args))
    assert first.ok
    assert first.data["synthesised"] is True
    assert first.data["result_ref"] == second.data["result_ref"]


async def test_calls_are_recorded_for_assertions() -> None:
    backend = MockToolBackend.from_pack_dir(DOMAINS / "retail")
    spec = build_registry(load_pack(DOMAINS, "retail"), GLOBAL).resolve("get_order_status")
    await backend.invoke(spec, payload("get_order_status", {"order_reference": "ORD-000777"}))
    assert backend.calls == [("get_order_status", {"order_reference": "ORD-000777"})]


def test_a_failing_fixture_must_carry_an_error_code() -> None:
    with pytest.raises(ValidationError, match="needs an error_code"):
        ToolFixture(status=ToolStatus.NOT_FOUND)


def test_fixtures_match_on_an_argument_subset() -> None:
    fixture = ToolFixture(when={"a": 1}, data={})
    assert fixture.matches({"a": 1, "b": 2})
    assert not fixture.matches({"a": 2})


def test_an_empty_matcher_matches_anything() -> None:
    assert ToolFixture(data={}).matches({"anything": "at all"})


def test_an_empty_fixture_set_is_valid() -> None:
    assert ToolFixtureSet().for_tool("whatever") == ()
