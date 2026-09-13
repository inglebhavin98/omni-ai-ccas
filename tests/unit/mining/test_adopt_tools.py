"""Binding adopted intents to a pack's tools (9.19).

Adoption takes what the corpus says: intent names, counts, and the parameters it annotates.
It cannot know which tool serves an intent, because the corpus has no idea the pack exists.
That mapping is pack authorship, and it lives in the pack.

Two mismatches have to be bridged, and neither may be bridged by inventing data:

* **Names.** The corpus calls a parameter `order_number`; the pack's tool takes
  `order_reference`. `slot_aliases` renames the corpus's name to the pack's, because
  `_check_slots_cover_tools` matches on exact slot names.
* **Coverage.** A tool whose required arguments the corpus does not annotate cannot be
  bound at all. Adding the missing slots would make an adopted taxonomy assert the corpus
  annotated something it did not, so the binding is refused instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.mining.adopt import adopt_taxonomy
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.taxonomy import IntentTaxonomy

REPO = Path(__file__).resolve().parents[3]
BITEXT = REPO / "data" / "raw" / "bitext" / "customer_support.jsonl"

pytestmark = pytest.mark.skipif(not BITEXT.is_file(), reason="bitext corpus not hydrated")


@pytest.fixture(scope="module")
def taxonomy() -> IntentTaxonomy:
    pack = load_pack(REPO / "domains", "retail")
    return adopt_taxonomy(
        BITEXT,
        domain="retail",
        source=DatasetSource.BITEXT,
        slot_pii=pack.slot_pii,
        slot_aliases=pack.slot_aliases,
        intent_tools=pack.intent_tools,
    )


def test_the_corpus_slot_is_renamed_to_the_packs_name(taxonomy: IntentTaxonomy) -> None:
    """`_check_slots_cover_tools` matches exact names, so the alias must be applied."""
    node = taxonomy.resolve("order.track_order")
    names = {s.name for s in node.slots}
    assert "order_reference" in names
    assert "order_number" not in names


def test_a_bound_intent_declares_its_tool(taxonomy: IntentTaxonomy) -> None:
    assert "get_order_status" in taxonomy.resolve("order.track_order").required_tools


def test_an_unbound_intent_declares_none(taxonomy: IntentTaxonomy) -> None:
    """Most intents have no tool that fits, and saying so is the honest outcome."""
    assert taxonomy.resolve("feedback.complaint").required_tools == ()


def test_the_renamed_slot_keeps_its_pii_entity(taxonomy: IntentTaxonomy) -> None:
    """Rule 2: renaming must not drop the marker that forces redaction on capture."""
    slot = next(
        s for s in taxonomy.resolve("order.track_order").slots if s.name == "order_reference"
    )
    assert slot.pii_entity is not None


def test_a_binding_the_slots_cannot_cover_is_refused() -> None:
    """Inventing the missing slots would make an adopted taxonomy assert what the corpus
    never annotated."""
    pack = load_pack(REPO / "domains", "retail")
    with pytest.raises(ValueError, match=r"cannot cover|uncovered"):
        adopt_taxonomy(
            BITEXT,
            domain="retail",
            source=DatasetSource.BITEXT,
            limit=4000,
            slot_pii=pack.slot_pii,
            slot_aliases=pack.slot_aliases,
            # update_delivery_address needs address_line_1, postal_code and country.
            # This intent is in the first 4,000 rows and the corpus annotates none of them.
            intent_tools={"shipping.change_shipping_address": ("update_delivery_address",)},
            pack_tools=pack.tools_by_name,
        )


def test_the_shipped_taxonomy_loads_with_its_bindings() -> None:
    """The real gate: load_taxonomy runs both coverage checks and must not raise."""
    from ccas.config.domain_loader import load_domain

    loaded = load_domain(REPO / "domains", "retail")
    bound = [n for n in loaded.taxonomy.nodes if n.required_tools]
    assert bound, "no intent is bound to a tool -- the graph can answer but never act"


def test_a_limited_adoption_truncates_the_taxonomy_rather_than_sampling_it() -> None:
    """`--limit` is for speed and it silently changes what the taxonomy contains.

    Bitext is grouped by category, so the first 4,000 rows hold 5 intents of 27 -- not a
    sample. A limited run therefore produces a taxonomy missing four fifths of the label
    set, which looks complete and is not. Recorded as a test because the flag reads like
    sampling and is not.
    """
    pack = load_pack(REPO / "domains", "retail")
    limited = adopt_taxonomy(
        BITEXT,
        domain="retail",
        source=DatasetSource.BITEXT,
        limit=4000,
        slot_pii=pack.slot_pii,
        slot_aliases=pack.slot_aliases,
    )
    full = adopt_taxonomy(
        BITEXT, domain="retail", source=DatasetSource.BITEXT, slot_pii=pack.slot_pii
    )
    assert len(limited.leaves()) < len(full.leaves()) / 2
    assert len(full.leaves()) == 27
