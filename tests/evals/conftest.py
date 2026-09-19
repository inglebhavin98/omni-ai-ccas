"""Make the eval gates leave a record (docs/future-scoped-work.md 6.7).

`test_provider_parity.py` emits `parity.router` through `LOG.info` with the agreement
rate, the divergence count and per-variant p95. Nothing calls ``configure_logging`` in a
pytest run, so that event went to captured stdout and was discarded on pass.

The cost was concrete: the Rule 6 gate ran green for the first time on 2026-09-15 and the
only thing anyone could say about it afterwards was the exit code. The numbers it had just
measured -- at 16 LLM calls against a 50/day cap -- were gone.

Scoped to ``tests/evals`` on purpose. These are the runs whose output is expensive to
reproduce; the rest of the suite has no reason to write to the shared log.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ccas.observability.logging import configure_logging

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="package", autouse=True)
def _eval_logging() -> Iterator[None]:
    """Send eval events to ``logs/execution.log``, where every other gate writes.

    ``force=True`` because a test earlier in the session may already have configured
    logging at a temp path; without it this would silently keep writing there, which is
    the same class of failure this fixture exists to fix.
    """
    configure_logging(REPO / "logs" / "execution.log", console=False, force=True)
    yield
