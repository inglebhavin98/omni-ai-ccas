"""Adopt a labelled corpus's published label set as a taxonomy (ADR-0020).

Not mining. Rule 9 forbids clustering a labelled corpus, because clustering rediscovers
the labels it was given and proves nothing. Taking the labels the corpus *publishes* is a
different operation with different evidence behind it, and `TaxonomyProvenance.ADOPTED`
says so on the artefact itself.

What is real here and what is not, stated plainly because an adopted taxonomy looks
exactly as authoritative as a mined one:

* **Real:** the hierarchy, the intent names, the utterance counts, and the slots -- Bitext
  annotates `{{Order Number}}` wherever a parameter belongs, so slots are measured rather
  than guessed.
* **Not real:** that these intents describe *your* callers. They describe the corpus
  author's idea of a support taxonomy. Volume shares are shares of a published dataset,
  not of traffic, and `AutomationScore.confidence` is set low to say so.

Half the corpus defines the taxonomy and the other half is held out, so an evaluator
cannot grade a router on the rows that defined the labels it is being graded against.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from ccas.ingestion.datasets import DatasetRole, require_role
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.common import Slug, utcnow
from ccas.schemas.pii import PiiEntityType
from ccas.schemas.taxonomy import (
    AutomationScore,
    IntentNode,
    IntentTaxonomy,
    SlotSpec,
    SlotType,
    TaxonomyProvenance,
    VolumeStats,
)

__all__ = [
    "DERIVATION_SPLIT",
    "adopt_taxonomy",
    "holdout_rows",
    "slots_from_placeholders",
    "split_of",
]

#: The half that defines the taxonomy. Its complement is the evaluation set.
DERIVATION_SPLIT = "a"

#: Bitext marks parameters inline: "cancel {{Reference}}".
_PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")


#: An adopted taxonomy has not been checked against real traffic, and the score should not
#: pretend otherwise. Low confidence is the honest value, not a hedge.
_ADOPTED_CONFIDENCE = 0.3


def split_of(row_id: str) -> str:
    """Deterministic half. Stable across runs and machines, so a holdout stays held out."""
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return DERIVATION_SPLIT if digest[0] % 2 == 0 else "b"


def holdout_rows(row_ids: Iterable[str]) -> list[str]:
    """The rows an evaluator may use: everything the taxonomy was *not* derived from."""
    return [row_id for row_id in row_ids if split_of(row_id) != DERIVATION_SPLIT]


def _slug(value: str) -> Slug:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "unknown"


def _rows(path: Path, limit: int | None) -> Iterator[tuple[str, dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(itertools.islice(handle, limit)):
            yield f"{path.stem}:{index}", json.loads(line)


def _slot_for(name: str, pii: Mapping[str, PiiEntityType]) -> SlotSpec:
    """A published placeholder becomes a slot. The prompt is generic by necessity.

    The corpus says *where* a parameter belongs, never how to ask for it, so the wording
    here is a placeholder for a pack author to improve -- not an assertion about how this
    vertical talks to its customers.
    """
    readable = name.strip().lower()
    slug = _slug(name)
    return SlotSpec(
        name=slug,
        slot_type=SlotType.STRING,
        required=True,
        elicitation_prompt=f"Could you give me the {readable}?",
        reprompt=f"Sorry, I still need the {readable}.",
        pii_entity=pii.get(slug),
    )


def slots_from_placeholders(
    names: Iterable[str], pii: Mapping[str, PiiEntityType] | None = None
) -> tuple[SlotSpec, ...]:
    """One slot per distinct parameter, deduplicated by slug.

    Published placeholders vary in case and spacing -- `{{Account Type}}` and
    `{{Account type}}` name one parameter -- and `IntentNode` rejects duplicate slot
    names, so collapsing them here is what keeps a real corpus adoptable.
    """
    by_slug: dict[str, SlotSpec] = {}
    for name in sorted(names):
        slot = _slot_for(name, pii or {})
        by_slug.setdefault(slot.name, slot)
    return tuple(by_slug.values())


def adopt_taxonomy(
    path: Path,
    domain: str,
    source: DatasetSource,
    limit: int | None = None,
    version: str = "0.1.0",
    min_rows: int = 5,
    slot_pii: Mapping[str, PiiEntityType] | None = None,
) -> IntentTaxonomy:
    """Build an ``IntentTaxonomy`` from a corpus's published category/intent labels."""
    # Gate 1: only a corpus admitted as ground truth has labels worth adopting.
    require_role(source, DatasetRole.NLU_GROUND_TRUTH)

    counts: dict[tuple[str, str], int] = {}
    placeholders: dict[tuple[str, str], set[str]] = {}
    for row_id, row in _rows(path, limit):
        if split_of(row_id) != DERIVATION_SPLIT:
            continue  # held out for evaluation
        category, intent = row.get("category"), row.get("intent")
        if not category or not intent:
            continue
        key = (_slug(str(category)), _slug(str(intent)))
        counts[key] = counts.get(key, 0) + 1
        # Caller side only. The agent's half of a row carries response-template variables
        # -- website URL, opening time, "first step" -- 391 of them against the caller's 9.
        # Scanning both gave intents 47 "slots" the bot would have interrogated callers for.
        found = _PLACEHOLDER.findall(str(row.get("instruction", "")))
        placeholders.setdefault(key, set()).update(found)

    counts = {key: n for key, n in counts.items() if n >= min_rows}
    if not counts:
        raise ValueError(f"no labelled rows in {path} after the derivation split")

    total = sum(counts.values())
    nodes: list[IntentNode] = []
    for category in sorted({c for c, _ in counts}):
        category_total = sum(n for (c, _), n in counts.items() if c == category)
        nodes.append(
            IntentNode(
                intent_id=category,
                level=1,
                label=category.replace("_", " ").title(),
                description=f"Contacts about {category.replace('_', ' ')}.",
                volume=_volume(category_total, total),
                automation=_automation(category_total / total),
            )
        )

    for (category, intent), n in sorted(counts.items()):
        slots = slots_from_placeholders(placeholders.get((category, intent), ()), slot_pii)
        nodes.append(
            IntentNode(
                intent_id=f"{category}.{intent}",
                level=2,
                parent_id=category,
                label=intent.replace("_", " ").title(),
                description=(
                    f"Caller wants to {intent.replace('_', ' ')} ({category.replace('_', ' ')})."
                ),
                volume=_volume(n, total),
                automation=_automation(n / total, slot_count=len(slots)),
                slots=slots,
            )
        )

    return IntentTaxonomy(
        taxonomy_id=f"{domain}-adopted-{source.value}",
        domain=domain,
        version=version,
        nodes=tuple(nodes),
        provenance=TaxonomyProvenance.ADOPTED,
        adopted_from=source.value,
        derivation_split=DERIVATION_SPLIT,
        coverage=1.0,
        noise_ratio=0.0,
        source_call_ids=(),
        generated_at=utcnow(),
    )


def _volume(count: int, total: int) -> VolumeStats:
    return VolumeStats(
        utterance_count=count,
        call_count=count,
        share_of_total=round(count / total, 6),
    )


def _automation(share: float, slot_count: int = 0) -> AutomationScore:
    """Feasibility from what the corpus actually shows: volume and parameter count.

    More required parameters means more turns and more ways to fail, so complexity rises
    with slots. Confidence stays low whatever the numbers say -- nothing here has been
    checked against live traffic.
    """
    complexity = min(1.0, 0.25 + 0.15 * slot_count)
    return AutomationScore(
        feasibility=round(max(0.0, 1.0 - complexity), 4),
        complexity=round(complexity, 4),
        confidence=_ADOPTED_CONFIDENCE,
        rationale=(
            "Adopted from a published label set: the intent is real, its fit to this "
            "deployment's traffic is unverified."
        ),
        volume_share=round(share, 6),
    )
