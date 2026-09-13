"""Shared API fixtures.

Lifted out of test_workbench.py so more than one module can drive the app. The stub
provider means these exercise the real routes, the real graph and the real redaction
pipeline without an LLM call -- which is what lets the API be tested at all while the
free-tier quota is spent.
"""

from __future__ import annotations

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
