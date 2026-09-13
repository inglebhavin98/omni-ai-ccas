"""The console's contract with the API it calls.

`src/ccas/api/static/index.html` is 313 lines of hand-written JS and its only previous
test asserted the file was served and contained its own title. That passes while every
button is broken.

A browser test would be better and needs a dependency outside the locked stack (Rule 5).
These assert the two things that break in practice without one: the console calling a
route that has been renamed, and the JS driving an element that is not in the DOM. Both
are silent in a browser -- a dead button looks like a slow one.

What this deliberately does NOT cover: rendering, event wiring, or whether a turn appears
on screen. See docs/future-scoped-work.md 9.21.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
CONSOLE = REPO / "src" / "ccas" / "api" / "static" / "index.html"

#: `api("/v1/...")` and `` api(`/v1/...${x}`) `` -- the console's only way out.
#: Each delimiter is matched to its own closing delimiter: a template literal can contain
#: a quote (``${$("toolName").value}``), so a shared character class truncates it.
_CALL = re.compile(
    r"""api\(\s*(?:`(/v1/[^`]*)`|"(/v1/[^"]*)"|'(/v1/[^']*)')""",
)
_ELEMENT_ID = re.compile(r"""\$\(\s*["']([A-Za-z0-9_]+)["']\s*\)""")
_DEFINED_ID = re.compile(r"""id=["']([A-Za-z0-9_]+)["']""")


@pytest.fixture(scope="module")
def markup() -> str:
    return CONSOLE.read_text(encoding="utf-8")


def _template_to_route(path: str) -> str:
    """`/v1/sessions/${session.session_id}/turns` -> `/v1/sessions/{}/turns`.

    Scans for the brace that closes each `${`, rather than the first one: the console
    writes `${$("toolName").value}`, whose inner `}` belongs to the nested call.
    """
    out, i = [], 0
    while i < len(path):
        if path.startswith("${", i):
            depth, j = 1, i + 2
            while j < len(path) and depth:
                depth += (path[j] == "{") - (path[j] == "}")
                j += 1
            out.append("{}")
            i = j
            continue
        out.append(path[i])
        i += 1
    return "".join(out)


def _route_shape(path: str) -> str:
    """FastAPI's `/v1/sessions/{session_id}` -> `/v1/sessions/{}`, so the two compare."""
    return re.sub(r"\{[^}]*\}", "{}", path)


def _called_routes(markup: str) -> set[str]:
    """Every /v1 path the console requests, as a route shape."""
    routes = set()
    for groups in _CALL.findall(markup):
        raw = next(g for g in groups if g)
        routes.add(_template_to_route(raw).split("?", 1)[0])
    return routes


def test_the_console_calls_at_least_one_endpoint(markup: str) -> None:
    """A regex that silently matches nothing would make every test below vacuous."""
    assert len(_called_routes(markup)) >= 5


def test_every_endpoint_the_console_calls_exists(markup: str, client: TestClient) -> None:
    """The failure this exists for: a route renamed in Python, still called in JS."""
    served = {_route_shape(r.path) for r in client.app.routes if hasattr(r, "path")}
    called = _called_routes(markup)
    missing = sorted(c for c in called if c not in served)
    assert not missing, f"console calls routes the API does not serve: {missing}"


def test_every_element_the_console_drives_is_in_the_dom(markup: str) -> None:
    """`$("typo")` returns null and the handler dies silently on the next line."""
    referenced = set(_ELEMENT_ID.findall(markup))
    defined = set(_DEFINED_ID.findall(markup))
    assert referenced, "no element ids found -- the extraction is broken, not the console"
    assert not referenced - defined, (
        f"JS drives ids with no element: {sorted(referenced - defined)}"
    )


def test_the_console_posts_a_turn_the_api_accepts(client: TestClient) -> None:
    """Exercise the console's own sequence: open a session, then send a turn."""
    created = client.post("/v1/sessions", json={"domain": "retail"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    turn = client.post(f"/v1/sessions/{session_id}/turns", json={"text": "where is my delivery"})
    assert turn.status_code == 200
    assert "transcript" in turn.json()


def test_the_console_reads_the_shape_the_domains_route_returns(client: TestClient) -> None:
    """The console populates its domain picker from these keys."""
    body = client.get("/v1/domains").json()
    assert body, "no domains -- the picker would be empty"
    for name, info in body.items():
        assert isinstance(name, str)
        assert "has_taxonomy" in info and "tools" in info


def test_the_console_is_served_and_self_contained(client: TestClient, markup: str) -> None:
    """No CDN, no build step: it must run from the file the API returns.

    An external script tag would make the console depend on network access the rest of
    the platform does not need, and on a third party seeing who loads it.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "omni-ai-ccas workbench" in response.text
    external = re.findall(r"""<(?:script|link)[^>]+(?:src|href)=["'](https?://[^"']+)""", markup)
    assert not external, f"console loads external resources: {external}"
