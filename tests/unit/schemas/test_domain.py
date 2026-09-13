from __future__ import annotations

import pytest
from pydantic import ValidationError

from ccas.schemas import (
    ComplianceProfile,
    ComplianceRegime,
    ConfidenceThresholds,
    DomainPack,
    PatternSpec,
    PiiEntityType,
    QueueSpec,
    ToolSpec,
)

STRICT_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def tool(name: str = "get_record_status", domain: str = "retail") -> ToolSpec:
    return ToolSpec(name=name, description="d", input_schema=STRICT_SCHEMA, domain=domain)


def pack(**kw: object) -> DomainPack:
    base: dict[str, object] = {
        "domain": "retail",
        "version": "1.0.0",
        "display_name": "Retail",
        "taxonomy_ref": "taxonomy.json",
        "queues": (QueueSpec(name="tier-1", display_name="Tier 1"),),
        "greeting": "Thanks for calling. How can I help?",
    }
    base.update(kw)
    return DomainPack(**base)  # type: ignore[arg-type]


def test_tools_must_belong_to_the_pack() -> None:
    with pytest.raises(ValidationError, match="but the pack is"):
        pack(tools=(tool(domain="healthcare"),))


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate tool name"):
        pack(tools=(tool(), tool()))


def test_duplicate_queue_names_are_rejected() -> None:
    queue = QueueSpec(name="tier-1", display_name="Tier 1")
    with pytest.raises(ValidationError, match="duplicate queue name"):
        pack(queues=(queue, queue))


def test_duplicate_pattern_names_are_rejected() -> None:
    pattern = PatternSpec(name="ref", pattern=r"\d{8}", entity_type=PiiEntityType.ACCOUNT_REF)
    with pytest.raises(ValidationError, match="duplicate pattern name"):
        pack(redaction_patterns=(pattern, pattern))


def test_a_pack_needs_at_least_one_queue() -> None:
    with pytest.raises(ValidationError):
        pack(queues=())


def test_lookup_helpers_index_by_name() -> None:
    p = pack(tools=(tool(), tool("create_record")))
    assert set(p.tools_by_name) == {"get_record_status", "create_record"}
    assert p.default_queue.name == "tier-1"
    assert p.queues_by_name["tier-1"].display_name == "Tier 1"


def test_recording_consent_requires_a_prompt() -> None:
    with pytest.raises(ValidationError, match="needs a consent_prompt"):
        ComplianceProfile(recording_consent_required=True)


def test_compliance_regimes_are_declared_per_pack() -> None:
    profile = ComplianceProfile(
        regimes=(ComplianceRegime.HIPAA,),
        transcript_retention_days=2555,
        recording_consent_required=True,
        consent_prompt="This call may be recorded.",
    )
    assert ComplianceRegime.HIPAA in pack(compliance=profile).compliance.regimes


def test_clarify_floor_cannot_exceed_the_route_threshold() -> None:
    with pytest.raises(ValidationError, match="clarify_floor must not exceed"):
        ConfidenceThresholds(route=0.5, clarify_floor=0.9)


def test_version_must_be_semver() -> None:
    with pytest.raises(ValidationError):
        pack(version="1.0")


def test_a_new_vertical_needs_no_code_change() -> None:
    """The point of the pack: an unseen domain slug is just data."""
    p = pack(domain="municipal-utilities", tools=(tool(domain="municipal-utilities"),))
    assert p.domain == "municipal-utilities"
