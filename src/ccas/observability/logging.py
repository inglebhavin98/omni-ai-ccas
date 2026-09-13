"""Structured JSON logging to ``logs/execution.log``.

Logs are the easiest place for Rule 2 to be violated by accident -- a debug line with
the caller's utterance in it looks harmless in review and is a breach in production.
``_forbid_raw_content`` makes that a loud failure instead of a silent one.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog

from ccas.schemas.pii import RedactedText

__all__ = [
    "DEFAULT_LOG_PATH",
    "LOGGER_NAMESPACE",
    "ForbiddenLogFieldError",
    "configure_logging",
    "get_logger",
]

DEFAULT_LOG_PATH = Path("logs/execution.log")

#: Handlers attach here, not to the root logger. ``logs/execution.log`` is a JSON-lines
#: file that tooling parses; a third-party library writing a plain-text warning into it
#: (spaCy and Presidio both do) would corrupt every consumer.
LOGGER_NAMESPACE = "ccas"

#: Keys that can only ever hold raw caller content. Passing one is a programming error.
_FORBIDDEN_KEYS = frozenset(
    {
        "utterance",
        "transcript",
        "raw_text",
        "raw",
        "text",
        "audio",
        "digits",
        "dtmf",
        "caller_id",
        "ani",
        "prompt",
        "answer",
        "slot_value",
    }
)

_configured = False


class ForbiddenLogFieldError(RuntimeError):
    """A log call carried a field that may contain unredacted caller content."""


def _forbid_raw_content(
    _logger: object, _method: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Reject raw-content keys, and unwrap ``RedactedText`` only when it is CLEAN."""
    offending = _FORBIDDEN_KEYS & event.keys()
    if offending:
        raise ForbiddenLogFieldError(
            f"log event {event.get('event')!r} carries raw-content field(s) "
            f"{sorted(offending)}; log entity types and counts, never content "
            "(CLAUDE.md Rule 2)"
        )
    for key, value in list(event.items()):
        if isinstance(value, RedactedText):
            event[key] = value.text if value.egress_permitted else "<redaction-failed>"
    return event


def _stringify_paths(
    _logger: object, _method: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in list(event.items()):
        if isinstance(value, Path):
            event[key] = str(value)
    return event


def configure_logging(
    log_path: Path = DEFAULT_LOG_PATH,
    level: str = "INFO",
    *,
    console: bool = False,
    force: bool = False,
) -> None:
    """Install JSON-lines logging on the ``ccas`` namespace. Idempotent unless forced.

    ``console`` adds a human-readable stderr stream for interactive use; the JSON file
    is always written, so a demo run leaves the same audit trail a service would.
    Third-party loggers are deliberately untouched -- they keep their default stderr
    behaviour and never reach the JSON file.
    """
    global _configured
    if _configured and not force:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAMESPACE)
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    logger.propagate = False

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _forbid_raw_content,
            _stringify_paths,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = LOGGER_NAMESPACE) -> structlog.stdlib.BoundLogger:
    """Bound logger under the ``ccas`` namespace.

    Names are namespaced so that every ccas logger -- including the CLI and the batch
    scripts -- lands in the JSON file, and nothing else does. Always attach
    ``correlation_id`` (CLAUDE.md Rule 8).
    """
    qualified = (
        name
        if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}.")
        else f"{LOGGER_NAMESPACE}.{name}"
    )
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(qualified)
    return logger
