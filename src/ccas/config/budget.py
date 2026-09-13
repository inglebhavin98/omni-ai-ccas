"""Latency budget loaded from ``configs/latency_budget.yaml`` (CLAUDE.md Rule 3).

Keeping the budget in a file rather than in code lets ``tests/latency/`` assert against
the same numbers an operator tunes, so the two can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ccas.schemas.common import Frozen
from ccas.schemas.session import LatencyLedger

__all__ = ["LatencyBudget", "StageBudgets", "load_budget"]


class StageBudgets(Frozen):
    """Per-stage ceilings. One shape serves every channel.

    The audio stages default to zero because a text channel does not have them -- there is
    no end-of-speech to detect, nothing to transcribe, no first audio chunk to emit. Zero
    is the honest value: the stage costs nothing because it does not run. The stages that
    *are* the platform's own work -- redact, router, tool, llm -- are required everywhere,
    because every channel pays them.
    """

    #: Voice-only. Absent on a text channel (ADR-0019).
    vad_ms: int = Field(default=0, ge=0)
    stt_ms: int = Field(default=0, ge=0)
    tts_ttfb_ms: int = Field(default=0, ge=0)

    redact_ms: int = Field(ge=0)
    router_ms: int = Field(ge=0)
    tool_ms: int = Field(ge=0)
    llm_ttft_ms: int = Field(ge=0)

    @property
    def total_ms(self) -> int:
        return sum(self.model_dump().values())

    def as_dict(self) -> dict[str, int]:
        return dict(self.model_dump())


class LatencyBudget(Frozen):
    budget_ms: int = Field(ge=1)
    stages: StageBudgets
    percentile: int = Field(default=95, ge=50, le=100)

    @model_validator(mode="after")
    def _warn_on_impossible_budget(self) -> Self:
        # Stage sums may slightly exceed the ceiling on purpose (negative slack keeps
        # regressions visible), but a wild overrun means the file is simply wrong.
        if self.stages.total_ms > self.budget_ms * 1.25:
            raise ValueError(
                f"stage budgets total {self.stages.total_ms}ms against a "
                f"{self.budget_ms}ms ceiling -- that is not a tight budget, it is a typo"
            )
        return self

    @property
    def slack_ms(self) -> int:
        return self.budget_ms - self.stages.total_ms

    def exceeds(self, stage: str, observed_ms: int) -> bool:
        budgeted = self.stages.as_dict().get(stage)
        if budgeted is None:
            raise KeyError(f"unknown latency stage {stage!r}")
        return observed_ms > budgeted

    def new_ledger(self) -> LatencyLedger:
        return LatencyLedger(budget_ms=self.budget_ms)


def load_budget(path: Path) -> LatencyBudget:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LatencyBudget.model_validate(data)
