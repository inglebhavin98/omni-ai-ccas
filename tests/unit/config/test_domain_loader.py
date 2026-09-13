from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ccas.config.domain_loader import (
    DomainPackNotFoundError,
    available_domains,
    load_domain,
    load_pack,
    load_taxonomy,
)

REPO = Path(__file__).resolve().parents[3]
DOMAINS = REPO / "domains"


def test_every_pack_on_disk_is_discoverable() -> None:
    on_disk = tuple(sorted(d.name for d in DOMAINS.iterdir() if (d / "pack.yaml").exists()))
    assert available_domains(DOMAINS) == on_disk
    # Rule 1's bar: a feature is complete only when it works for more than one pack.
    assert {"retail", "healthcare"} <= set(on_disk)


def test_available_domains_is_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert available_domains(tmp_path / "nope") == ()


@pytest.mark.parametrize("domain", ["retail", "healthcare", "insurance"])
def test_shipped_packs_validate(domain: str) -> None:
    pack = load_pack(DOMAINS, domain)
    assert pack.domain == domain
    assert pack.queues


def test_retail_declares_its_tools_with_strict_schemas() -> None:
    pack = load_pack(DOMAINS, "retail")
    assert set(pack.tools_by_name) == {
        "get_order_status",
        "start_return",
        "update_delivery_address",
    }
    for tool in pack.tools:
        assert tool.input_schema["additionalProperties"] is False


def test_side_effecting_tools_demand_strong_verification() -> None:
    pack = load_pack(DOMAINS, "retail")
    for tool in pack.tools:
        if tool.side_effecting:
            assert tool.requires_idempotency_key
            assert tool.requires_verification.rank >= 2


def test_packs_differ_in_posture_without_differing_in_code() -> None:
    """The whole point of Rule 1: two verticals, one core."""
    retail = load_pack(DOMAINS, "retail")
    healthcare = load_pack(DOMAINS, "healthcare")
    assert healthcare.confidence.route > retail.confidence.route
    assert healthcare.compliance.transcript_retention_days > (
        retail.compliance.transcript_retention_days
    )


def test_unknown_domain_raises_instead_of_defaulting() -> None:
    with pytest.raises(DomainPackNotFoundError, match="available packs"):
        load_pack(DOMAINS, "does-not-exist")


def test_pack_directory_and_declared_domain_must_agree(tmp_path: Path) -> None:
    root = tmp_path / "retail"
    root.mkdir()
    data = yaml.safe_load((DOMAINS / "retail" / "pack.yaml").read_text(encoding="utf-8"))
    data["domain"] = "banking"
    data["tools"] = []
    (root / "pack.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="lives in a directory named"):
        load_pack(tmp_path, "retail")


def test_missing_taxonomy_is_reported_lazily() -> None:
    """A pack without a taxonomy still loads; asking for one explains how to get it.

    Uses a pack that genuinely has none. `retail` acquired an adopted taxonomy in
    ADR-0020, and a test asserting its absence would now be asserting the opposite of
    what the project wants.
    """
    without = [
        d.name
        for d in sorted(DOMAINS.iterdir())
        if (d / "pack.yaml").is_file() and not (d / "taxonomy.json").is_file()
    ]
    assert without, "every pack has a taxonomy -- this test has nothing left to check"
    loaded = load_domain(DOMAINS, without[0])
    assert not loaded.has_taxonomy
    with pytest.raises(DomainPackNotFoundError, match="mine_taxonomy"):
        _ = loaded.taxonomy


def test_taxonomy_domain_must_match_the_pack(tmp_path: Path) -> None:
    pack = load_pack(DOMAINS, "retail")
    root = tmp_path / "retail"
    root.mkdir()
    (root / "taxonomy.json").write_text(
        json.dumps(
            {
                "taxonomy_id": "t",
                "domain": "banking",
                "version": "1.0.0",
                "nodes": [
                    {
                        "intent_id": "billing",
                        "level": 1,
                        "label": "Billing",
                        "description": "d",
                        "volume": {"utterance_count": 1, "call_count": 1, "share_of_total": 0.5},
                        "automation": {
                            "feasibility": 0.5,
                            "complexity": 0.5,
                            "confidence": 0.5,
                            "rationale": "r",
                            "volume_share": 0.5,
                        },
                    }
                ],
                "embedding_model": "m",
                "clusterer": "hdbscan",
                "labeler_model": "m",
                "coverage": 1.0,
                "noise_ratio": 0.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is for domain 'banking'"):
        load_taxonomy(tmp_path, pack)


def test_intent_requiring_an_undeclared_tool_fails_at_load(tmp_path: Path) -> None:
    """Better a startup failure than a mid-call one."""
    pack = load_pack(DOMAINS, "retail")
    root = tmp_path / "retail"
    root.mkdir()
    (root / "taxonomy.json").write_text(
        json.dumps(
            {
                "taxonomy_id": "t",
                "domain": "retail",
                "version": "1.0.0",
                "nodes": [
                    {
                        "intent_id": "billing",
                        "level": 1,
                        "label": "Billing",
                        "description": "d",
                        "required_tools": ["cancel_everything"],
                        "volume": {"utterance_count": 1, "call_count": 1, "share_of_total": 0.5},
                        "automation": {
                            "feasibility": 0.5,
                            "complexity": 0.5,
                            "confidence": 0.5,
                            "rationale": "r",
                            "volume_share": 0.5,
                        },
                    }
                ],
                "embedding_model": "m",
                "clusterer": "hdbscan",
                "labeler_model": "m",
                "coverage": 1.0,
                "noise_ratio": 0.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not declare"):
        load_taxonomy(tmp_path, pack)
