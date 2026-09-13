"""The redaction pipeline.

Detect with regex, then Presidio for what regex cannot settle; resolve overlaps;
substitute stable placeholders; re-scan the output; only then decide the status.

The status is *earned*, never assumed. ``CLEAN`` requires every step to have run and the
post-substitution scan to come back empty. Anything else blocks egress (CLAUDE.md Rule 2).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from ccas.redaction.engine import DetectedSpan, merge_spans
from ccas.redaction.leak_detector import LeakDetector
from ccas.redaction.placeholder import PlaceholderAllocator
from ccas.redaction.policy import PatternSet, RedactionPolicy, load_pattern_set, load_policy
from ccas.redaction.presidio_onnx import PresidioEngine
from ccas.redaction.regex_engine import GENERIC_PATTERNS_PATH, RegexEngine
from ccas.redaction.vault import PlaceholderVault
from ccas.schemas.domain import DomainPack
from ccas.schemas.pii import (
    EngineName,
    PiiEntityType,
    RedactedText,
    RedactionReport,
    RedactionSpan,
    RedactionStatus,
    sha256_hex,
)

__all__ = ["RedactionMode", "RedactionPipeline", "build_pipeline"]

DEFAULT_POLICY_PATH = Path("configs/redaction_policy.yaml")


class RedactionMode(StrEnum):
    """Which engines run, and therefore what the pipeline costs.

    The split exists because it was measured, not assumed: regex alone is ~14us p99,
    regex plus a spaCy NER pass is ~7.5ms p99 -- 2.5x over the realtime slice, using the
    *small* model. See docs/adr/0007-two-mode-redaction.md.
    """

    REALTIME = "realtime"
    """Regex only. Fits the 3ms call-path slice with room to spare. Catches every
    structured identifier and self-identified names; full NER does not run."""

    BATCH = "batch"
    """Regex plus Presidio NER. Everything that is stored, mined, or handed to a human
    passes through this mode, where latency is not on a caller's clock."""


class RedactionPipeline:
    """Stateless across calls; pass an allocator to share placeholders within a session."""

    __slots__ = ("_detector", "_mode", "_policy", "_presidio", "_regex")

    def __init__(
        self,
        policy: RedactionPolicy,
        regex: RegexEngine,
        presidio: PresidioEngine | None = None,
        mode: RedactionMode = RedactionMode.BATCH,
    ) -> None:
        self._policy = policy
        self._regex = regex
        self._presidio = presidio
        self._mode = mode
        leak_policy = policy.leak_detection
        self._detector = LeakDetector(regex, enabled=leak_policy.enabled)

    @property
    def policy(self) -> RedactionPolicy:
        return self._policy

    @property
    def mode(self) -> RedactionMode:
        return self._mode

    def for_mode(self, mode: RedactionMode) -> RedactionPipeline:
        """Same engines, different cost profile. Cheap -- share one set of engines."""
        return RedactionPipeline(self._policy, self._regex, self._presidio, mode)

    def new_allocator(self, vault: PlaceholderVault | None = None) -> PlaceholderAllocator:
        return PlaceholderAllocator(self._policy.placeholder_format, vault)

    # ---------------------------------------------------------------- readiness

    def unavailable_engines(self) -> tuple[str, ...]:
        """Enabled, fail-closed engines that cannot run. Non-empty means UNVERIFIED."""
        blocking: list[str] = []
        for spec in self._policy.engines:
            if not spec.enabled or not spec.fail_closed:
                continue
            if not self._runs_in_mode(spec.name):
                continue
            engine = self._engine_named(spec.name)
            if engine is None or not engine.available:
                blocking.append(spec.name)
        return tuple(blocking)

    def _runs_in_mode(self, engine_name: str) -> bool:
        if engine_name == "presidio_onnx":
            return self._mode is RedactionMode.BATCH
        return True

    @property
    def ready(self) -> bool:
        return not self.unavailable_engines()

    def engine_load_error(self, name: str) -> str | None:
        """Why an engine is unavailable, for an actionable CLI message."""
        engine = self._engine_named(name)
        return getattr(engine, "load_error", None)

    def _engine_named(self, name: str) -> RegexEngine | PresidioEngine | None:
        if name == "regex":
            return self._regex
        if name == "presidio_onnx":
            return self._presidio
        return None

    # ---------------------------------------------------------------- redaction

    def redact(self, text: str, allocator: PlaceholderAllocator | None = None) -> RedactedText:
        started = time.perf_counter_ns()
        source_hash = sha256_hex(text)
        allocator = allocator or self.new_allocator()

        blocking = self.unavailable_engines()
        if blocking:
            # Fail closed. A missing engine is not a licence to emit the original.
            return RedactedText(
                text="",
                report=RedactionReport.unverified(
                    policy_version=self._policy.version,
                    reason=f"engine_unavailable:{','.join(blocking)}",
                ),
                source_sha256=source_hash,
            )

        detected, engines_run = self._detect(text)
        spans = merge_spans(tuple(d for d in detected if d.score >= self._policy.min_score))
        redacted_text, applied = self._substitute(text, spans, allocator)

        leak = self._detector.check(redacted_text)
        elapsed_us = (time.perf_counter_ns() - started) // 1000

        if not leak.clean:
            report = RedactionReport(
                status=RedactionStatus.DIRTY,
                spans=applied,
                engines_run=engines_run,
                policy_version=self._policy.version,
                elapsed_us=elapsed_us,
                leak_check_passed=False,
                residual_patterns=leak.residual_patterns,
            )
            # The text is withheld: a DIRTY payload must not carry the thing that leaked.
            return RedactedText(text="", report=report, source_sha256=source_hash)

        return RedactedText(
            text=redacted_text,
            report=RedactionReport.clean(
                spans=applied,
                engines_run=engines_run,
                policy_version=self._policy.version,
                elapsed_us=elapsed_us,
            ),
            source_sha256=source_hash,
        )

    def redact_value(
        self,
        value: str,
        entity_type: PiiEntityType,
        allocator: PlaceholderAllocator | None = None,
        custom_label: str | None = None,
    ) -> RedactedText:
        """Replace an entire value with a placeholder, regardless of its shape.

        Pattern matching cannot help when the *context* is what makes a value sensitive.
        A keypad-entered reference is six bare digits; a name is just a word. When the
        caller was asked for a specific thing and gave it, the answer is that thing --
        so it is replaced wholesale rather than scanned.
        """
        started = time.perf_counter_ns()
        allocator = allocator or self.new_allocator()
        token = allocator.allocate(entity_type, value, custom_label)
        span = RedactionSpan(
            start=0,
            end=len(value),
            entity_type=entity_type,
            custom_label=custom_label,
            score=1.0,
            engine="manual",
            replacement=token,
        )
        return RedactedText(
            text=token,
            report=RedactionReport.clean(
                spans=(span,),
                engines_run=("manual",),
                policy_version=self._policy.version,
                elapsed_us=(time.perf_counter_ns() - started) // 1000,
            ),
            source_sha256=sha256_hex(value),
        )

    def redact_many(
        self, texts: Iterable[str], allocator: PlaceholderAllocator | None = None
    ) -> tuple[RedactedText, ...]:
        """Redact a conversation. One allocator, so an entity keeps one token throughout."""
        shared = allocator or self.new_allocator()
        return tuple(self.redact(text, shared) for text in texts)

    # ---------------------------------------------------------------- internals

    def _detect(self, text: str) -> tuple[tuple[DetectedSpan, ...], tuple[EngineName, ...]]:
        detected: list[DetectedSpan] = []
        engines_run: list[EngineName] = []

        regex_policy = self._policy.engine("regex")
        if regex_policy is None or regex_policy.enabled:
            detected.extend(self._regex.detect(text))
            engines_run.append("regex")

        presidio_policy = self._policy.engine("presidio_onnx")
        if (
            self._runs_in_mode("presidio_onnx")
            and presidio_policy is not None
            and presidio_policy.enabled
            and self._presidio is not None
            and self._presidio.available
        ):
            detected.extend(self._presidio.detect(text))
            engines_run.append("presidio_onnx")

        return tuple(detected), tuple(engines_run)

    @staticmethod
    def _substitute(
        text: str, spans: tuple[DetectedSpan, ...], allocator: PlaceholderAllocator
    ) -> tuple[str, tuple[RedactionSpan, ...]]:
        """Replace right-to-left so earlier offsets stay valid.

        Recorded offsets refer to the *original* text, which is what an auditor needs to
        reconcile a report against a source record.
        """
        applied: list[RedactionSpan] = []
        buffer = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            surface = text[span.start : span.end]
            replacement = allocator.allocate(span.entity_type, surface, span.custom_label)
            buffer = buffer[: span.start] + replacement + buffer[span.end :]
            applied.append(
                RedactionSpan(
                    start=span.start,
                    end=span.end,
                    entity_type=span.entity_type,
                    custom_label=span.custom_label,
                    score=span.score,
                    engine=span.engine,
                    replacement=replacement,
                )
            )
        return buffer, tuple(reversed(applied))


def build_pipeline(
    policy_path: Path = DEFAULT_POLICY_PATH,
    patterns_path: Path = GENERIC_PATTERNS_PATH,
    pack: DomainPack | None = None,
    presidio: PresidioEngine | None = None,
    mode: RedactionMode = RedactionMode.BATCH,
) -> RedactionPipeline:
    """Assemble a pipeline, layering a pack's patterns over the global set."""
    policy = load_policy(policy_path)
    pattern_set: PatternSet = load_pattern_set(patterns_path)
    if pack is not None and pack.redaction_patterns:
        pattern_set = pattern_set.merged_with(pack.redaction_patterns)

    regex = RegexEngine(pattern_set)

    presidio_policy = policy.engine("presidio_onnx")
    # REALTIME never runs NER, so it must never pay to load the model -- constructing it
    # costs ~1s of spaCy startup that the call path has no use for.
    if (
        presidio is None
        and mode is RedactionMode.BATCH
        and presidio_policy is not None
        and presidio_policy.enabled
    ):
        presidio = PresidioEngine(
            entities=presidio_policy.entities,
            min_score=presidio_policy.min_score,
            spacy_model=presidio_policy.spacy_model,
        )
    return RedactionPipeline(policy, regex, presidio, mode)
