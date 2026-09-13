"""Adopting a published label set is not mining (ADR-0020).

Rule 9 forbids mining Bitext: clustering a labelled corpus rediscovers its own labels and
proves nothing. Taking the labels it *publishes* and using them as a taxonomy is a
different operation -- no clustering happens -- and it is how a chat vertical gets a
taxonomy without waiting on the AIxBlock coverage question.

The trap it must not fall into is the one CLAUDE.md names: "do not grade a taxonomy
against the corpus it was mined from". An adopted taxonomy is derived from half the corpus
and evaluated on the other half, and the split is recorded so an evaluator cannot
accidentally test on what it trained on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccas.config.domain_loader import load_pack
from ccas.ingestion.datasets import DatasetRoleError
from ccas.mining.adopt import DERIVATION_SPLIT, adopt_taxonomy, holdout_rows, split_of
from ccas.schemas.call_log import DatasetSource
from ccas.schemas.taxonomy import IntentTaxonomy, TaxonomyProvenance

REPO = Path(__file__).resolve().parents[3]
BITEXT = REPO / "data" / "raw" / "bitext" / "customer_support.jsonl"

pytestmark = pytest.mark.skipif(not BITEXT.is_file(), reason="bitext corpus not hydrated")


@pytest.fixture(scope="module")
def taxonomy() -> IntentTaxonomy:
    """Adopted exactly as the script does, including the pack's slot/PII mapping.

    The mapping lives in the pack rather than the adopter because the core may not know
    what a vertical calls things (Rule 1), so a test that hardcoded it would be testing a
    path no caller takes.
    """
    pack = load_pack(REPO / "domains", "retail")
    return adopt_taxonomy(
        BITEXT,
        domain="retail",
        source=DatasetSource.BITEXT,
        limit=4000,
        slot_pii=pack.slot_pii,
    )


def test_the_result_declares_itself_adopted_not_mined(taxonomy: IntentTaxonomy) -> None:
    """A reader must be able to tell an adopted taxonomy from a mined one."""
    assert taxonomy.provenance is TaxonomyProvenance.ADOPTED
    assert taxonomy.adopted_from == "bitext"
    assert taxonomy.embedding_model == ""
    assert taxonomy.labeler_model == ""


def test_the_hierarchy_comes_from_the_published_labels(taxonomy: IntentTaxonomy) -> None:
    """Bitext publishes category -> intent, which is an L1/L2 tree already."""
    levels = {node.level for node in taxonomy.nodes}
    assert levels == {1, 2}
    for node in taxonomy.nodes:
        if node.level == 2:
            assert node.parent_id is not None
            assert node.intent_id.startswith(f"{node.parent_id}.")


def test_leaves_are_the_intents_not_the_categories(taxonomy: IntentTaxonomy) -> None:
    leaves = taxonomy.leaves()
    assert leaves
    assert all(leaf.level == 2 for leaf in leaves)


def test_volumes_are_counted_not_invented(taxonomy: IntentTaxonomy) -> None:
    """Adopted does not mean unmeasured -- the counts are real rows."""
    total = sum(node.volume.utterance_count for node in taxonomy.leaves())
    assert total > 0
    for node in taxonomy.leaves():
        assert node.volume.utterance_count > 0
        assert 0.0 < node.volume.share_of_total <= 1.0
    assert sum(n.volume.share_of_total for n in taxonomy.leaves()) == pytest.approx(1.0, abs=0.01)


def test_slots_come_from_the_published_placeholders(taxonomy: IntentTaxonomy) -> None:
    """`{{Order Number}}` marks where a parameter belongs -- that is a slot, measured."""
    with_slots = [n for n in taxonomy.leaves() if n.slots]
    assert with_slots, "bitext annotates placeholders; some intent must have a slot"
    for node in with_slots:
        for slot in node.slots:
            assert slot.elicitation_prompt


def test_the_derivation_split_is_recorded(taxonomy: IntentTaxonomy) -> None:
    """Without this an evaluator cannot know what it must hold out."""
    assert taxonomy.derivation_split == DERIVATION_SPLIT


def test_holdout_and_derivation_do_not_overlap() -> None:
    """The guard against grading a taxonomy on the rows that defined it."""
    derive = {r for r in range(2000) if split_of(f"row:{r}") == DERIVATION_SPLIT}
    holdout = set(holdout_rows(f"row:{r}" for r in range(2000)))
    assert derive and holdout
    assert not ({f"row:{r}" for r in derive} & holdout)


def test_the_split_is_deterministic() -> None:
    assert split_of("bitext:17") == split_of("bitext:17")


def test_a_corpus_without_the_ground_truth_role_is_refused() -> None:
    """AIxBlock has no labels to adopt; asking for them should fail loudly."""
    with pytest.raises(DatasetRoleError):
        adopt_taxonomy(BITEXT, domain="retail", source=DatasetSource.AIXBLOCK, limit=10)


def test_placeholders_that_slugify_alike_become_one_slot() -> None:
    """`{{Account Type}}` and `{{Account type}}` are one parameter, not two.

    Caught by the full corpus after a 4,000-row sample passed: `IntentNode` rejects
    duplicate slot names, so a case-variant placeholder crashed the adoption outright.
    """
    from ccas.mining.adopt import slots_from_placeholders

    slots = slots_from_placeholders({"Account Type", "Account type", "ACCOUNT TYPE"})
    assert [s.name for s in slots] == ["account_type"]


def test_slots_are_ordered_deterministically() -> None:
    from ccas.mining.adopt import slots_from_placeholders

    first = slots_from_placeholders({"Order Number", "Account Type", "Invoice Number"})
    second = slots_from_placeholders({"Invoice Number", "Order Number", "Account Type"})
    assert [s.name for s in first] == [s.name for s in second]


def test_only_caller_side_placeholders_become_slots(taxonomy: IntentTaxonomy) -> None:
    """A slot is what you ask the caller for, not what the agent fills in.

    Bitext marks placeholders in both halves of a row. The caller's side has 9 distinct
    parameters -- order number, invoice number, refund amount. The agent's side has 391,
    and they are response-template variables: website URL, opening time, "first step".
    Scanning both produced intents with 47 "slots" the bot would have interrogated the
    caller for.
    """
    names = {slot.name for node in taxonomy.nodes for slot in node.slots}
    assert names, "some intent must have a slot"
    assert len(names) <= 15, f"agent-side template variables leaked into slots: {sorted(names)}"
    for leaked in ("website_url", "first_step", "customer_support_hours", "opening_time"):
        assert leaked not in names


def test_identifying_slots_are_marked_for_redaction(taxonomy: IntentTaxonomy) -> None:
    """Rule 2: a slot holding an identifier must force redaction of whatever fills it."""
    by_name = {slot.name: slot for node in taxonomy.nodes for slot in node.slots}
    for name in ("order_number", "invoice_number", "person_name"):
        if name in by_name:
            assert by_name[name].pii_entity is not None, f"{name} must be redacted"
