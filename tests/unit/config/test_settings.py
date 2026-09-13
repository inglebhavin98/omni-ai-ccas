from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ccas.config.settings import Settings


def build(**kw: object) -> Settings:
    # _env_file=None keeps a developer's local .env out of the assertions.
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_defaults_point_at_the_repo_layout() -> None:
    settings = build()
    assert settings.default_domain == "retail"
    assert settings.domains_dir == Path("./domains")
    assert settings.models_config == Path("configs/models.yaml")
    assert settings.latency_budget_config == Path("configs/latency_budget.yaml")


def test_env_prefix_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CCAS_DEFAULT_DOMAIN", "healthcare")
    monkeypatch.setenv("CCAS_LOG_LEVEL", "DEBUG")
    settings = build()
    assert settings.default_domain == "healthcare"
    assert settings.log_level == "DEBUG"


def test_anthropic_key_uses_its_conventional_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = build()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"


def test_secrets_do_not_appear_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    assert "sk-super-secret" not in repr(build())


def test_redaction_bypass_is_rejected_outside_local() -> None:
    """CLAUDE.md Rule 2: the bypass is a fixture affordance, not a deployment flag."""
    with pytest.raises(ValidationError, match="permitted only in local"):
        build(env="production", allow_redaction_bypass=True)


def test_redaction_bypass_is_allowed_locally() -> None:
    assert build(env="local", allow_redaction_bypass=True).allow_redaction_bypass


def test_redaction_defaults_to_enforcing() -> None:
    assert build().allow_redaction_bypass is False


def test_domain_dir_resolves_under_domains_root() -> None:
    assert build().domain_dir("retail") == Path("domains/retail")


def test_settings_are_frozen() -> None:
    settings = build()
    with pytest.raises(ValidationError):
        settings.default_domain = "healthcare"
