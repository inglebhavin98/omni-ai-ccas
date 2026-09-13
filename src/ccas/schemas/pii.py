"""Redaction contracts (Module 2).

The type system is the enforcement mechanism for CLAUDE.md Rule 2: ``RedactedText``
is the only string type permitted to cross an external-provider boundary, and it
cannot exist without a ``RedactionReport`` explaining how it got that way.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ccas.schemas.common import Frozen, Slug

__all__ = [
    "PiiEntityType",
    "RedactedText",
    "RedactionReport",
    "RedactionSpan",
    "RedactionStatus",
    "sha256_hex",
]

EngineName = Literal["regex", "presidio_onnx", "llm", "manual"]


class PiiEntityType(StrEnum):
    """Domain-agnostic entity taxonomy.

    Deliberately generic: a domain pack that needs its own subscriber or booking
    reference uses ``ACCOUNT_REF``, or ``CUSTOM`` with a ``custom_label``. Adding a domain
    member here would violate Rule 1.
    """

    PERSON = "person"
    ORG = "org"
    LOCATION = "location"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    IP = "ip"
    DATE_OF_BIRTH = "date_of_birth"
    AGE = "age"
    NATIONAL_ID = "national_id"
    PAYMENT_CARD = "payment_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"
    ACCOUNT_REF = "account_ref"
    CREDENTIAL = "credential"
    HEALTH_INFO = "health_info"
    BIOMETRIC = "biometric"
    VEHICLE_ID = "vehicle_id"
    CUSTOM = "custom"


class RedactionStatus(StrEnum):
    CLEAN = "clean"
    """Engine ran and the post-substitution leak check passed. Egress permitted."""

    DIRTY = "dirty"
    """Entities were found but substitution failed. Egress forbidden."""

    UNVERIFIED = "unverified"
    """Engine unavailable or errored. Egress forbidden -- never fail open (Rule 2)."""

    BYPASSED = "bypassed"
    """Explicit test-only override. Egress forbidden in every environment."""


class RedactionSpan(Frozen):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    entity_type: PiiEntityType
    custom_label: Slug | None = None
    score: float = Field(ge=0.0, le=1.0)
    engine: EngineName
    replacement: str = Field(min_length=1)
    """Stable placeholder, e.g. ``[PERSON_1]``. The same entity keeps the same token
    for the life of a session so the model can still corefer across turns."""

    @model_validator(mode="after")
    def _check_bounds_and_label(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must exceed start ({self.start})")
        is_custom = self.entity_type is PiiEntityType.CUSTOM
        if is_custom and self.custom_label is None:
            raise ValueError("custom_label is required when entity_type is CUSTOM")
        if not is_custom and self.custom_label is not None:
            raise ValueError("custom_label is only valid when entity_type is CUSTOM")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: RedactionSpan) -> bool:
        return self.start < other.end and other.start < self.end


class RedactionReport(Frozen):
    """Provenance for one redaction pass.

    Carries no matched text by design -- only entity types, counts and pattern *names*.
    A report is safe to log; the thing it describes is not.
    """

    status: RedactionStatus
    spans: tuple[RedactionSpan, ...] = ()
    engines_run: tuple[EngineName, ...] = ()
    policy_version: str = Field(min_length=1)
    elapsed_us: int = Field(ge=0)
    leak_check_passed: bool = False
    residual_patterns: tuple[str, ...] = ()
    """Names of patterns that still matched after substitution. Never the match itself."""

    @model_validator(mode="after")
    def _check_clean_is_earned(self) -> Self:
        if self.status is RedactionStatus.CLEAN:
            if not self.leak_check_passed:
                raise ValueError("CLEAN requires leak_check_passed=True")
            if self.residual_patterns:
                raise ValueError(
                    f"CLEAN forbids residual_patterns, got {list(self.residual_patterns)}"
                )
            if not self.engines_run:
                raise ValueError("CLEAN requires at least one engine to have run")
        if self.status is not RedactionStatus.CLEAN and self.leak_check_passed:
            raise ValueError(f"leak_check_passed=True is inconsistent with {self.status}")
        return self

    @property
    def egress_permitted(self) -> bool:
        """The single predicate every boundary check must consult."""
        return self.status is RedactionStatus.CLEAN

    @property
    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in self.spans:
            key = span.custom_label or span.entity_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @classmethod
    def clean(
        cls,
        *,
        spans: tuple[RedactionSpan, ...] = (),
        engines_run: tuple[EngineName, ...],
        policy_version: str,
        elapsed_us: int,
    ) -> RedactionReport:
        """Build a CLEAN report. Callers must have already run the leak check."""
        return cls(
            status=RedactionStatus.CLEAN,
            spans=spans,
            engines_run=engines_run,
            policy_version=policy_version,
            elapsed_us=elapsed_us,
            leak_check_passed=True,
        )

    @classmethod
    def unverified(cls, *, policy_version: str, reason: str) -> RedactionReport:
        """Engine unavailable. Produces a report that can never permit egress."""
        return cls(
            status=RedactionStatus.UNVERIFIED,
            engines_run=(),
            policy_version=policy_version,
            elapsed_us=0,
            leak_check_passed=False,
            residual_patterns=(reason,),
        )


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RedactedText(Frozen):
    """The only string type permitted to leave the process.

    ``source_sha256`` exists so two utterances can be compared or deduplicated without
    retaining the original anywhere. The original is never stored (Rule 2).
    """

    text: str
    report: RedactionReport
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def egress_permitted(self) -> bool:
        return self.report.egress_permitted

    def require_egress(self) -> str:
        """Return the text, or refuse. Use at every external boundary."""
        if not self.egress_permitted:
            raise PermissionError(f"egress blocked: redaction status is {self.report.status.value}")
        return self.text
