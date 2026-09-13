"""Workbench API tests.

Driven through ``TestClient`` with a stub provider, so the suite exercises the same
handlers a browser hits without a network or a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ccas.api.main import build_app
from ccas.api.sessions import SessionManager
from ccas.config.settings import Settings
from ccas.llm.bindings import load_bindings
from tests.graph_stub import StubGraphProvider

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        domains_dir=REPO / "domains",
        config_dir=REPO / "configs",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    """A configured provider: the graph routes and responds."""
    manager = SessionManager(
        settings=settings,
        provider=StubGraphProvider(),
        bindings=load_bindings(REPO / "configs" / "models.yaml"),
    )
    return TestClient(build_app(manager))


@pytest.fixture
def bare_client(settings: Settings) -> TestClient:
    """No provider configured -- the honest degraded mode."""
    manager = SessionManager(
        settings=settings, bindings=load_bindings(REPO / "configs" / "models.yaml")
    )
    return TestClient(build_app(manager))


# ----------------------------------------------------------------- readiness


def test_health_is_trivial(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_readiness_names_what_is_missing(bare_client: TestClient) -> None:
    """ "not ready" without a reason is the least useful health check there is."""
    body = bare_client.get("/ready").json()
    assert body["provider_configured"] is False
    assert body["provider_error"]
    # Every pack on disk, not a hard-coded list -- adding one should not fail this.
    on_disk = {d.name for d in (REPO / "domains").iterdir() if (d / "pack.yaml").exists()}
    assert set(body["domains"]) == on_disk


def test_domains_expose_pack_shape(client: TestClient) -> None:
    body = client.get("/v1/domains").json()
    assert body["retail"]["compliance"] == ["pci_dss", "gdpr"]
    assert "get_order_status" in body["retail"]["tools"]
    assert body["healthcare"]["has_taxonomy"] is False


# ------------------------------------------------------------------ sessions


def test_a_new_session_opens_with_the_greeting(client: TestClient) -> None:
    body = client.post("/v1/sessions", json={"domain": "retail"}).json()
    assert body["session_id"].startswith("wb-")
    assert body["transcript"][0]["speaker"] == "bot"
    assert "recorded" in body["transcript"][0]["text"].lower()


def test_an_unknown_domain_is_a_404(client: TestClient) -> None:
    assert client.post("/v1/sessions", json={"domain": "banking"}).status_code == 404


def test_an_unknown_session_is_a_404(client: TestClient) -> None:
    assert client.get("/v1/sessions/wb-nope").status_code == 404


def test_a_turn_advances_the_transcript(client: TestClient) -> None:
    session = client.post("/v1/sessions", json={"domain": "retail"}).json()
    body = client.post(
        f"/v1/sessions/{session['session_id']}/turns", json={"text": "hello there"}
    ).json()
    assert len(body["transcript"]) > len(session["transcript"])
    assert any(t["speaker"] == "caller" for t in body["transcript"])


def test_the_transcript_is_redacted(client: TestClient) -> None:
    session = client.post("/v1/sessions", json={"domain": "retail"}).json()
    body = client.post(
        f"/v1/sessions/{session['session_id']}/turns",
        json={"text": "my card is 4111 1111 1111 1111"},
    ).json()
    rendered = json.dumps(body)
    assert "4111 1111 1111 1111" not in rendered
    assert "[PAYMENT_CARD_1]" in rendered


def test_every_turn_reports_all_policy_verdicts(client: TestClient) -> None:
    """Seeing *why* a turn went the way it did is the point of a workbench."""
    session = client.post("/v1/sessions", json={"domain": "retail"}).json()
    assert [v["policy"] for v in session["policy_verdicts"]] == [
        "risk",
        "retry",
        "sentiment",
        "confidence",
    ]


def test_a_session_can_be_dropped(client: TestClient) -> None:
    session = client.post("/v1/sessions", json={"domain": "retail"}).json()
    assert client.delete(f"/v1/sessions/{session['session_id']}").status_code == 200
    assert client.get(f"/v1/sessions/{session['session_id']}").status_code == 404


def test_without_a_provider_the_session_still_opens(bare_client: TestClient) -> None:
    """Redaction, tools and policies work with no LLM; routing honestly does not."""
    session = bare_client.post("/v1/sessions", json={"domain": "retail"}).json()
    assert session["transcript"]
    body = bare_client.post(
        f"/v1/sessions/{session['session_id']}/turns", json={"text": "where is my delivery"}
    ).json()
    assert body["escalation"] is not None


# -------------------------------------------------------------------- probes


def test_redaction_preview_needs_no_provider(bare_client: TestClient) -> None:
    body = bare_client.get(
        "/v1/redaction/preview",
        params={"text": "call me on 415-555-0142", "domain": "retail"},
    ).json()
    assert body["egress_permitted"] is True
    assert "[PHONE_1]" in body["redacted"]
    assert body["entity_counts"] == {"phone": 1}


def test_redaction_preview_applies_pack_patterns(bare_client: TestClient) -> None:
    body = bare_client.get(
        "/v1/redaction/preview", params={"text": "ref ORD-884210", "domain": "retail"}
    ).json()
    assert "ORD-884210" not in body["redacted"]


def test_the_tool_catalogue_marks_scope(bare_client: TestClient) -> None:
    body = bare_client.get("/v1/tools", params={"domain": "retail"}).json()
    assert body["transfer_to_human"]["scope"] == "shared"
    assert body["get_order_status"]["scope"] == "pack"


def test_the_prober_runs_the_same_gate_chain(bare_client: TestClient) -> None:
    body = bare_client.post(
        "/v1/tools/get_order_status/invoke",
        params={"domain": "retail"},
        json={"arguments": {"order_reference": "ORD-884210"}, "verification": "strong"},
    ).json()
    assert body["status"] == "ok"
    assert body["data"]["recipient_name"] == "[RECIPIENT_NAME_1]"


def test_the_prober_enforces_verification(bare_client: TestClient) -> None:
    body = bare_client.post(
        "/v1/tools/update_delivery_address/invoke",
        params={"domain": "retail"},
        json={
            "arguments": {
                "order_reference": "ORD-884210",
                "address_line_1": "a",
                "postal_code": "p",
                "country": "c",
            },
            "verification": "soft",
        },
    ).json()
    assert body["status"] == "denied"


def test_the_prober_validates_arguments(bare_client: TestClient) -> None:
    body = bare_client.post(
        "/v1/tools/get_order_status/invoke",
        params={"domain": "retail"},
        json={"arguments": {}, "verification": "strong"},
    ).json()
    assert body["status"] == "invalid_args"


def test_an_unregistered_tool_is_a_404(bare_client: TestClient) -> None:
    assert (
        bare_client.post("/v1/tools/delete_everything/invoke", json={"arguments": {}}).status_code
        == 404
    )


def test_the_prober_works_for_the_second_pack(bare_client: TestClient) -> None:
    """Rule 1 again: the same endpoint, a different pack, no code change."""
    body = bare_client.post(
        "/v1/tools/check_copay/invoke",
        params={"domain": "healthcare"},
        json={"arguments": {"service_category": "outpatient"}, "verification": "strong"},
    ).json()
    assert body["status"] == "ok"


# ----------------------------------------------------------------------- ui


def test_the_console_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ai-ccas workbench" in response.text
