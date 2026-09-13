from __future__ import annotations

from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.policies.base import PolicyContext
from ccas.schemas.common import RiskTier
from ccas.schemas.domain import DomainPack
from ccas.schemas.taxonomy import EscalationPolicy

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def retail_pack() -> DomainPack:
    return load_pack(REPO / "domains", "retail")


@pytest.fixture
def ctx(retail_pack: DomainPack) -> PolicyContext:
    return PolicyContext(
        confidence=retail_pack.confidence,
        escalation=EscalationPolicy(),
        risk_tier=RiskTier.LOW,
        max_turns=retail_pack.max_turns,
        default_queue=retail_pack.default_queue.name,
    )
