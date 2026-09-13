"""Deterministic synthetic corpus.

Exists so that ingestion, redaction and mining can be exercised end to end in CI with no
download, and so that fixtures carrying PII shapes are demonstrably not real people's
data (CLAUDE.md Rule 2).

Every value is generated from a seed; the same seed always yields the same corpus.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

from ccas.ingestion.base import RawDtmf, RawRecord, RawTurn, SourceAdapter
from ccas.schemas.call_log import DatasetSource, GoldLabels
from ccas.schemas.common import Channel, Speaker

__all__ = ["SyntheticAdapter"]

_FIRST = ("Dana", "Priya", "Tomas", "Mei", "Olu", "Ines", "Karim", "Noor")
_LAST = ("Whitfield", "Raman", "Nkemelu", "Okafor", "Lindqvist", "Barros", "Haddad")

#: Intent-shaped openers. Deliberately generic -- a synthetic corpus that encoded one
#: vertical's vocabulary would make the core's domain-agnosticism untestable (Rule 1).
_OPENERS = (
    ("status_check", "I want to check the status of my request"),
    ("status_check", "can you tell me where my request has got to"),
    ("update_details", "I need to change the details on my account"),
    ("update_details", "my contact information is out of date"),
    ("billing_question", "there is a charge I do not recognise"),
    ("billing_question", "I have a question about my latest statement"),
    ("cancel_request", "I would like to cancel what I set up last week"),
    ("speak_to_human", "I just want to talk to a person please"),
)

_AGENT_LINES = (
    "Of course, I can help with that.",
    "Let me take a look for you.",
    "Thanks for waiting, I have that here.",
    "Is there anything else I can do?",
)


def _card(rng: random.Random) -> str:
    """A Luhn-valid but obviously synthetic card number."""
    body = [4] + [rng.randint(0, 9) for _ in range(14)]
    total = 0
    for index, digit in enumerate(reversed(body)):
        value = digit
        if index % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    digits = "".join(map(str, body)) + str((10 - total % 10) % 10)
    return " ".join(digits[i : i + 4] for i in range(0, 16, 4))


class SyntheticAdapter(SourceAdapter):
    source = DatasetSource.SYNTHETIC

    def __init__(self, count: int = 20, seed: int = 1234, pii_ratio: float = 0.6) -> None:
        self.count = count
        self.seed = seed
        self.pii_ratio = pii_ratio

    @property
    def expected_layout(self) -> str:
        return "no files -- records are generated from a seed; `path` is ignored"

    def read(self, path: Path | None = None, limit: int | None = None) -> Iterator[RawRecord]:
        rng = random.Random(self.seed)  # noqa: S311 - fixtures, not cryptography
        # For a generated corpus `limit` is authoritative: asking for 40 records and
        # silently getting 20 because of a constructor default is a bad surprise.
        total = limit if limit is not None else self.count
        for index in range(total):
            yield self._record(rng, index)

    def _record(self, rng: random.Random, index: int) -> RawRecord:
        intent, opener = rng.choice(_OPENERS)
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        turns = [
            RawTurn(speaker=Speaker.BOT, text="Thanks for calling. How can I help?"),
            RawTurn(speaker=Speaker.CALLER, text=f"Hi, this is {name}. {opener}."),
        ]
        dtmf: list[RawDtmf] = []

        if rng.random() < self.pii_ratio:
            turns.append(RawTurn(speaker=Speaker.HUMAN_AGENT, text=rng.choice(_AGENT_LINES)))
            secret = rng.choice(
                (
                    f"my card is {_card(rng)}",
                    f"you can email me at {name.split()[0].lower()}@example.com",
                    f"my number is 415-555-{rng.randint(1000, 9999)}",
                    f"my reference is ORD-{rng.randint(100000, 999999)}",
                )
            )
            turns.append(RawTurn(speaker=Speaker.CALLER, text=f"Sure, {secret}."))
            dtmf.append(RawDtmf(digits=f"{rng.randint(1000, 9999)}", at_ms=4200))

        turns.append(RawTurn(speaker=Speaker.HUMAN_AGENT, text=_AGENT_LINES[-1]))
        turns.append(RawTurn(speaker=Speaker.CALLER, text="No, that's everything. Thanks."))

        return RawRecord(
            source=self.source,
            record_id=f"synthetic-{self.seed}-{index:05d}",
            channel=Channel.VOICE,
            turns=tuple(turns),
            dtmf=tuple(dtmf),
            domain_hint=None,
            duration_ms=len(turns) * 4200,
            labels=GoldLabels(intent=intent),
            metadata={"seed": self.seed},
        )
