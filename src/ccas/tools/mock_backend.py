"""Deterministic backend for development and tests.

Responses come from ``domains/<pack>/tool_fixtures.yaml`` rather than from code, for the
same reason tools do: a mock that hard-coded one vertical's tool names would put that
vocabulary in the core (CLAUDE.md Rule 1), and the Phase 6 acceptance criterion is that
the healthcare pack runs the whole suite with no ``src/`` change.

A tool with no fixture still answers -- deterministically, by echoing its arguments --
so a newly declared tool is exercisable before anyone writes fixtures for it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, JsonValue, Slug
from ccas.schemas.pii import sha256_hex
from ccas.schemas.tools import ToolPayload, ToolSpec, ToolStatus
from ccas.tools.backend import BackendResponse, ToolBackend

__all__ = ["FIXTURES_FILENAME", "MockToolBackend", "ToolFixture", "ToolFixtureSet"]

FIXTURES_FILENAME = "tool_fixtures.yaml"


class ToolFixture(Frozen):
    """One canned response. ``when`` matches on a subset of the arguments."""

    when: dict[str, JsonValue] = Field(default_factory=dict)
    data: dict[str, JsonValue] = Field(default_factory=dict)
    status: ToolStatus = ToolStatus.OK
    error_code: Slug | None = None
    error_message: str | None = None
    retryable: bool = False
    delay_ms: int = Field(default=0, ge=0, le=60_000)
    """Simulated latency. Lets a fixture drive the executor's timeout path."""

    @model_validator(mode="after")
    def _check_error_shape(self) -> Self:
        if self.status is not ToolStatus.OK and self.error_code is None:
            raise ValueError(f"fixture with status {self.status.value} needs an error_code")
        return self

    def matches(self, arguments: dict[str, JsonValue]) -> bool:
        return all(arguments.get(key) == value for key, value in self.when.items())


class ToolFixtureSet(Frozen):
    tools: dict[Slug, tuple[ToolFixture, ...]] = Field(default_factory=dict)

    def for_tool(self, name: str) -> tuple[ToolFixture, ...]:
        return self.tools.get(name, ())


class MockToolBackend(ToolBackend):
    name = "mock"

    def __init__(self, fixtures: ToolFixtureSet | None = None) -> None:
        self.fixtures = fixtures or ToolFixtureSet()
        self.calls: list[tuple[str, dict[str, JsonValue]]] = []

    @classmethod
    def from_pack_dir(cls, pack_dir: Path) -> MockToolBackend:
        path = pack_dir / FIXTURES_FILENAME
        if not path.is_file():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(ToolFixtureSet.model_validate(raw))

    async def invoke(self, spec: ToolSpec, payload: ToolPayload) -> BackendResponse:
        self.calls.append((spec.name, dict(payload.arguments)))

        fixture = next(
            (f for f in self.fixtures.for_tool(spec.name) if f.matches(payload.arguments)),
            None,
        )
        if fixture is None:
            return self._synthesised(spec, payload)

        if fixture.delay_ms:
            await asyncio.sleep(fixture.delay_ms / 1000)
        if fixture.status is not ToolStatus.OK:
            return BackendResponse(
                status=fixture.status,
                error_code=fixture.error_code,
                error_message=fixture.error_message,
                retryable=fixture.retryable,
            )
        return BackendResponse(data=dict(fixture.data))

    @staticmethod
    def _synthesised(spec: ToolSpec, payload: ToolPayload) -> BackendResponse:
        """A stable answer for a tool nobody has written fixtures for yet."""
        canonical = "|".join(f"{k}={payload.arguments[k]!r}" for k in sorted(payload.arguments))
        return BackendResponse(
            data={
                "tool": spec.name,
                "echoed_arguments": dict(payload.arguments),
                "result_ref": sha256_hex(f"{spec.name}|{canonical}")[:12],
                "synthesised": True,
            }
        )
