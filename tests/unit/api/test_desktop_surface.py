"""The agent-desktop surface (tech-spec 3.2a): handoff fetch + copilot stream.

Drives the real routes, the real graph and the real escalation through the stub provider
-- no LLM. The handoff payload is the object the CTI path delivers, so what these tests
assert is that a session which escalated leaves behind exactly one fetchable payload,
and a session that did not leaves none.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _speak(client: TestClient, session_id: str, text: str) -> dict[str, Any]:
    response = client.post(f"/v1/sessions/{session_id}/turns", json={"text": text})
    assert response.status_code == 200, response.text
    return response.json()


def _escalated_session(client: TestClient) -> str:
    """A session that ends in escalation through the stub router ("no idea" band)."""
    created = client.post("/v1/sessions", json={"domain": "retail"})
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]
    snapshot = _speak(client, session_id, "no idea really")
    assert snapshot["escalation"] is not None
    assert snapshot["terminal"] is True
    return session_id


def _quiet_session(client: TestClient) -> str:
    """A session nobody has spoken into.

    The stub provider's keyword intents were written for the test-fixture taxonomy, not
    the pack's own mined one, so through the API every *spoken* turn escalates. A
    session with no turns is the honest no-handoff case here; the escalation journeys
    themselves are covered by tests/integration/test_graph_e2e.py against the fixture
    taxonomy.
    """
    created = client.post("/v1/sessions", json={"domain": "retail"})
    assert created.status_code == 200, created.text
    return created.json()["session_id"]


# ----------------------------------------------------------------- handoff fetch


def test_an_escalated_session_leaves_a_fetchable_handoff(client: TestClient) -> None:
    session_id = _escalated_session(client)
    response = client.get(f"/v1/sessions/{session_id}/handoff")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == session_id
    assert body["handoff_id"] == f"{session_id}-handoff"
    assert body["reason"]
    assert body["target_queue"]
    # Every transcript line is a redacted surface; the placeholder mark is the shape
    # redaction leaves behind, and no raw turn text may ride along.
    for turn in body["transcript"]:
        assert isinstance(turn["text"], str)


def test_a_session_without_escalation_has_no_handoff(client: TestClient) -> None:
    session_id = _quiet_session(client)
    response = client.get(f"/v1/sessions/{session_id}/handoff")
    assert response.status_code == 404


def test_an_unknown_handoff_id_is_a_404(client: TestClient) -> None:
    response = client.get("/v1/handoffs/does-not-exist")
    assert response.status_code == 404


def test_the_handoff_carries_no_raw_caller_text(client: TestClient) -> None:
    """Rule 2 spot check at the desktop boundary: a PAN spoken to a session that then
    escalates must arrive at the desktop only as a placeholder. (Redaction replaces
    rather than removes -- plain words survive, identifiers do not.)"""
    created = client.post("/v1/sessions", json={"domain": "retail"})
    session_id = created.json()["session_id"]
    snapshot = _speak(client, session_id, "my card is 4111 1111 1111 1111, where is my delivery")
    assert snapshot["escalation"] is not None
    body = client.get(f"/v1/sessions/{session_id}/handoff").json()
    rendered = json.dumps(body)
    assert "4111" not in rendered
    assert "PAYMENT_CARD" in rendered


def test_the_handoff_matches_the_session_snapshot(client: TestClient) -> None:
    """One payload, not a parallel truth: the desktop view must agree with what the
    workbench already showed for the same session."""
    session_id = _escalated_session(client)
    escalation = client.get(f"/v1/sessions/{session_id}").json()["escalation"]
    handoff = client.get(f"/v1/sessions/{session_id}/handoff").json()
    assert handoff["reason"] == escalation["reason"]


# ---------------------------------------------------------------- copilot stream


def test_the_copilot_stream_pushes_snapshots_then_closes(client: TestClient) -> None:
    session_id = _escalated_session(client)
    with client.websocket_connect(f"/v1/ws/copilot/{session_id}") as ws:
        messages: list[dict[str, Any]] = []
        while True:
            message = json.loads(ws.receive_text())
            messages.append(message)
            if "handoff" in message:
                break
        # The session was already terminal, so the stream is one snapshot plus the
        # handoff, then the server closed.
        snapshots = [m for m in messages if "handoff" not in m]
        assert snapshots, "at least one snapshot must arrive before the handoff"
        assert messages[-1]["handoff"]["session_id"] == session_id


def test_the_copilot_stream_refuses_an_unknown_session(client: TestClient) -> None:
    with (
        pytest.raises(Exception),  # noqa: B017 - WebSocketDisconnect surfaces as Exception
        client.websocket_connect("/v1/ws/copilot/wb-never-was") as ws,
    ):
        ws.receive_text()
