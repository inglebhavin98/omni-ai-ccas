"""Adapter output -> redaction -> ``CallLog``.

This is where Rule 2 is actually applied to a corpus. A record whose redaction does not
come back CLEAN is *quarantined*: it is counted, its id and the reason are recorded, and
it is dropped. It is never emitted in a weaker form, because a partially redacted record
that looks like a normal one is the single worst artifact this pipeline could produce.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ccas.ingestion.base import RawDtmf, RawRecord
from ccas.redaction.pipeline import RedactionMode, RedactionPipeline
from ccas.schemas.call_log import CallLog, DtmfEvent, Utterance
from ccas.schemas.pii import RedactedText, RedactionReport, RedactionStatus

__all__ = ["DTMF_MASK_THRESHOLD", "NormalizeOutcome", "NormalizeStats", "Normalizer"]

#: Runs at or above this length are masked before storage. Four digits is already a PIN.
DTMF_MASK_THRESHOLD = 4


@dataclass(slots=True)
class NormalizeOutcome:
    record_id: str
    call_log: CallLog | None = None
    reason: str | None = None
    """Why the record was quarantined. A pattern or status name -- never content."""

    @property
    def quarantined(self) -> bool:
        return self.call_log is None


@dataclass(slots=True)
class NormalizeStats:
    seen: int = 0
    emitted: int = 0
    quarantined: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    entity_counts: dict[str, int] = field(default_factory=dict)

    def record(self, outcome: NormalizeOutcome) -> None:
        self.seen += 1
        if outcome.quarantined:
            self.quarantined += 1
            key = outcome.reason or "unknown"
            self.reasons[key] = self.reasons.get(key, 0) + 1
            return
        self.emitted += 1
        assert outcome.call_log is not None
        for utterance in outcome.call_log.utterances:
            for label, count in utterance.content.report.entity_counts.items():
                self.entity_counts[label] = self.entity_counts.get(label, 0) + count

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined / self.seen if self.seen else 0.0

    def as_dict(self) -> dict[str, object]:
        """Safe to log: counts and reason names only."""
        return {
            "seen": self.seen,
            "emitted": self.emitted,
            "quarantined": self.quarantined,
            "quarantine_rate": round(self.quarantine_rate, 4),
            "reasons": dict(self.reasons),
            "entity_counts": dict(self.entity_counts),
        }


def _mask_dtmf(event: RawDtmf) -> DtmfEvent:
    """Mask anything long enough to be a secret.

    ``*`` is the mask character because it is already legal in the stored pattern. A
    genuine star keypress becomes indistinguishable from a mask, which is the correct
    trade: the live slot filler consumes DTMF in realtime, and what lands in storage
    only needs to record *that* digits were entered.
    """
    if len(event.digits) < DTMF_MASK_THRESHOLD:
        return DtmfEvent(digits=event.digits, at_ms=event.at_ms, redacted=False)
    return DtmfEvent(digits="*" * len(event.digits), at_ms=event.at_ms, redacted=True)


class Normalizer:
    """Turns adapter records into contract-valid, egress-clean ``CallLog`` objects."""

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: RedactionPipeline) -> None:
        if pipeline.mode is not RedactionMode.BATCH:
            raise ValueError(
                "ingestion must use RedactionMode.BATCH -- realtime mode skips NER, and "
                "stored records are exactly what must not contain a caller's name "
                "(docs/adr/0007-two-mode-redaction.md)"
            )
        self._pipeline = pipeline

    def normalize(self, record: RawRecord) -> NormalizeOutcome:
        call_id = CallLog.make_call_id(record.source, record.record_id)

        # One allocator per record: coreference holds inside a conversation, and a token
        # can never mean two different people across conversations.
        allocator = self._pipeline.new_allocator()
        redacted = self._pipeline.redact_many(record.texts, allocator)

        blocked = next((r for r in redacted if not r.report.egress_permitted), None)
        if blocked is not None:
            return NormalizeOutcome(record_id=record.record_id, reason=_reason_for(blocked))

        utterances = tuple(
            Utterance(
                index=index,
                speaker=turn.speaker,
                content=content,
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                asr_confidence=turn.confidence,
                language=turn.language,
            )
            for index, (turn, content) in enumerate(zip(record.turns, redacted, strict=True))
        )

        call_log = CallLog(
            call_id=call_id,
            source=record.source,
            source_record_id=record.record_id,
            channel=record.channel,
            domain_hint=record.domain_hint,
            locale=record.locale,
            utterances=utterances,
            dtmf_events=tuple(_mask_dtmf(event) for event in record.dtmf),
            duration_ms=record.duration_ms,
            labels=record.labels,
            metadata=record.metadata,
            redaction=_combined_report(redacted),
        )
        return NormalizeOutcome(record_id=record.record_id, call_log=call_log)

    def normalize_many(
        self, records: Iterable[RawRecord], stats: NormalizeStats | None = None
    ) -> Iterator[NormalizeOutcome]:
        for record in records:
            outcome = self.normalize(record)
            if stats is not None:
                stats.record(outcome)
            yield outcome


def _reason_for(blocked: RedactedText) -> str:
    report = blocked.report
    if report.status is RedactionStatus.UNVERIFIED:
        return f"unverified:{'|'.join(report.residual_patterns) or 'unknown'}"
    return f"dirty:{'|'.join(report.residual_patterns) or 'unknown'}"


def _combined_report(redacted: tuple[RedactedText, ...]) -> RedactionReport:
    """Record-level rollup. Every per-utterance report is kept on its own utterance."""
    spans = tuple(span for r in redacted for span in r.report.spans)
    engines: tuple[str, ...] = ()
    for result in redacted:
        for engine in result.report.engines_run:
            if engine not in engines:
                engines = (*engines, engine)
    return RedactionReport.clean(
        spans=spans,
        engines_run=engines,  # type: ignore[arg-type]
        policy_version=redacted[0].report.policy_version,
        elapsed_us=sum(r.report.elapsed_us for r in redacted),
    )
