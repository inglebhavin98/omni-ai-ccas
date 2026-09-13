# ADR-0020 — Adopting a published label set is not mining

- **Status:** accepted
- **Date:** 2026-09-13
- **Follows:** [0018](0018-freeze-voice-pivot-to-chat.md)
- **Constrained by:** [0004](0004-dataset-role-separation.md), amended by
  [0015](0015-natcs-is-supervision-not-a-trajectory.md)

## Context

The pivot to chat needed a taxonomy the retail pack could actually serve. Mining one was
not available on the timescale, for reasons that are recorded rather than guessed:

- The only minable corpus is AIxBlock, which is **insurance voice sales**. Mining it
  produces `domains/insurance/`, not retail.
- Its coverage tops out at **19.7%** and the cause is unexplained after three refuted
  hypotheses (`docs/future-scoped-work.md` 9.17).
- The LLM labelling step is blocked on a spent free-tier quota.

Meanwhile Bitext is sitting unused, and it is the closest thing this project has to the
target: 26,872 rows of **chat**, in **retail/e-commerce**, with 11 categories over 27
intents — an L1/L2 tree already — and placeholder annotations marking where parameters
belong.

Rule 9 forbids mining it, and that rule is right: clustering a labelled corpus rediscovers
the labels it was handed and proves nothing.

But **mining it and adopting its published labels are different operations**. No
clustering happens. The question is whether the second is legitimate, and under what
conditions it does not quietly become the first.

## Decision

`ccas.mining.adopt` builds an `IntentTaxonomy` from a corpus's published category/intent
labels. Four things keep it honest.

**1. The artefact says what it is.** `TaxonomyProvenance` is `MINED` or `ADOPTED`, and a
validator enforces that each carries its own evidence and none of the other's: a mined
taxonomy must name an embedder, a clusterer and a labeller; an adopted one must name the
corpus and must *not* assert mining provenance it does not have. An adopted taxonomy looks
exactly as authoritative as a mined one on a screen, so the distinction has to live in the
data.

**2. Half the corpus is held out.** CLAUDE.md forbids grading a taxonomy against the corpus
it was mined from, and the same trap is open here: derive labels from all 26,872 rows, then
measure router accuracy on those rows, and the number means nothing. `split_of()` is a
deterministic hash — 13,432 rows define the taxonomy, 13,440 stay held out, and
`derivation_split` is recorded on the artefact so an evaluator cannot get this wrong by
accident.

**3. Only the gate that fits is applied.** Adoption requires `NLU_GROUND_TRUTH`, not
`INTENT_MINING`. Asking to adopt AIxBlock fails loudly: it has no labels to adopt.

**4. The score says it is unverified.** `AutomationScore.confidence` is fixed at 0.3, with
the rationale *"the intent is real, its fit to this deployment's traffic is unverified"*.
Volume shares are shares of a published dataset, not of traffic.

### What is real and what is not

| real | not real |
|---|---|
| the hierarchy, intent names, utterance counts | that these intents describe *your* callers |
| the slots — Bitext annotates where parameters belong | the elicitation wording, which is generic filler for a pack author to improve |

**The slot extraction was wrong on the first attempt and the error is instructive.** Bitext
marks placeholders in both halves of a row. Scanning both gave `switch_account` **47
slots**, including `first_step`, `website_url` and `customer_support_hours` — the bot would
have interrogated callers for its own response template. The caller's side has **9**
distinct parameters: order number, invoice number, refund amount, delivery city. Only the
`instruction` field is scanned now. A slot is what you ask the caller for.

Which slots hold identifiers is declared in the **pack** (`slot_pii`), not in the adopter.
The core may not know what a vertical calls things (Rule 1), and hardcoding `order_number`
in `src/ccas/` failed the domain-literal gate immediately — correctly.

## Alternatives considered

- **Wait for the AIxBlock mine.** Wrong domain (insurance, not retail), wrong channel
  (voice), 19.7% coverage, and blocked on quota. Three independent reasons.
- **Hand-author a retail taxonomy.** Faster, and it would encode my guesses about retail
  contact reasons with no evidence at all. Bitext's labels are at least somebody's
  observed support taxonomy.
- **Treat adoption as mining with `clusterer: none`.** Would have avoided a schema change
  and made every downstream consumer unable to tell a measured taxonomy from a borrowed
  one.
- **Adopt without a holdout.** Simpler, and it forecloses the only evaluation that makes
  the chat channel's accuracy a number rather than an opinion.

## Consequences

- `domains/retail/taxonomy.json` ships: 38 nodes, 27 intents, 21 slots, 9 of them distinct,
  3 carrying a PII entity. **The chat channel routes against a real taxonomy for the first
  time.**
- `IntentTaxonomy` gains `provenance`, `adopted_from` and `derivation_split`; `clusterer`
  gains `"none"`; `embedding_model` and `labeler_model` are no longer required. Already at
  `schema_version` 1.1 (ADR-0017), which has not shipped, so no migration.
- **Module 3's gate now grades by provenance** — an adopted taxonomy is checked for corpus
  and split rather than for an embedder it never used.
- 13,440 held-out rows are waiting for the eval harness. Until it runs, the chat router's
  accuracy is **unmeasured**, and that is the next thing worth building.
- `required_tools` is empty on every node. Bitext's labels do not map onto the retail
  pack's three tools, so the graph will answer and escalate but not yet act. Mapping them
  is pack authorship, not adoption.
