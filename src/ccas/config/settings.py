"""Process settings.

Every environment read in the codebase funnels through here (CLAUDE.md Rule 8), so a
deployment can be audited by reading one class rather than grepping for ``os.environ``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccas.schemas.common import Slug

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CCAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: str = "local"
    log_level: str = "INFO"

    default_domain: Slug = "retail"
    domains_dir: Path = Path("./domains")
    config_dir: Path = Path("./configs")

    # --- llm (hybrid: both bindings stay live, Rule 6) ---------------------
    llm_default_provider: str = "anthropic"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_referer: str = "https://github.com/ai-ccas"
    openrouter_title: str = "ai-ccas"
    """OpenRouter asks callers to identify themselves; these become HTTP-Referer and
    X-Title, which is how a free-tier account is attributed rather than rate-limited."""
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: SecretStr = SecretStr("not-needed")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # --- voice (Phase 5) ---------------------------------------------------
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")
    livekit_api_key: SecretStr | None = Field(default=None, alias="LIVEKIT_API_KEY")
    livekit_api_secret: SecretStr | None = Field(default=None, alias="LIVEKIT_API_SECRET")
    deepgram_api_key: SecretStr | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    cartesia_api_key: SecretStr | None = Field(default=None, alias="CARTESIA_API_KEY")
    cartesia_voice_id: str = ""

    # --- redaction (Rule 2) ------------------------------------------------
    redaction_policy: Path = Path("./configs/redaction_policy.yaml")
    presidio_onnx_path: Path | None = None
    allow_redaction_bypass: bool = False

    # --- infra -------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def _guard_redaction_bypass(self) -> Settings:
        """The bypass exists for fixtures only. Outside ``local`` it is a Rule 2 breach."""
        if self.allow_redaction_bypass and self.env != "local":
            raise ValueError(
                f"CCAS_ALLOW_REDACTION_BYPASS is set in env={self.env!r}; "
                "redaction bypass is permitted only in local test runs"
            )
        return self

    @property
    def models_config(self) -> Path:
        return self.config_dir / "models.yaml"

    @property
    def latency_budget_config(self) -> Path:
        return self.config_dir / "latency_budget.yaml"

    @property
    def app_config(self) -> Path:
        return self.config_dir / "app.yaml"

    def domain_dir(self, domain: str) -> Path:
        return self.domains_dir / domain


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
