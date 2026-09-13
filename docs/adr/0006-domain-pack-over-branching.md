# ADR 0006 — Verticals are data, not code paths

Status: accepted · 2026-09-12

## Context

The platform must serve retail, healthcare, banking, telecom, utilities and SaaS. The
default way this goes wrong is incremental: one `if domain == "healthcare"` for a
consent prompt, then one for a confidence threshold, then one for a queue name. Each is
defensible in isolation and the core is vertical-specific within a quarter.

"Domain-agnostic" stated as an intention does not survive contact with a deadline.

## Decision

Everything a vertical needs is declared in `domains/<pack>/pack.yaml` as a `DomainPack`:
taxonomy reference, tools, queues, prompts, confidence thresholds, redaction patterns,
turn ceiling and compliance profile. Intents are `Slug` strings, never enums.

Enforcement is a test, not a convention. `tests/security/test_no_domain_literals.py`
scans `src/ccas/` for whole-word vertical vocabulary across healthcare, retail, banking
and telecom, and fails the build on a hit.

Two packs ship from Phase 1 (`retail`, `healthcare`) rather than one, because a single
pack cannot demonstrate that the core has no hidden assumptions.

## Consequences

- Adding a vertical is a YAML file plus a mined taxonomy. No `src/` change.
- The gate is strict enough to catch prose: on its first run it flagged three docstrings
  in the contract layer. Those were rewritten rather than allowlisted, which is the
  precedent this ADR intends to set.
- The ban list deliberately excludes engineering words that merely sound domain-ish —
  `policy` (`RetryPolicy`), `account` (`ACCOUNT_REF`), `card`, `iban`. Banning those
  would produce noise and the gate would be muted within a month.
- The Phase 6 acceptance criterion is that the full Phase 4/5 suite passes against the
  healthcare pack with zero `src/` changes. If it does not, the leak is fixed before the
  phase closes.
- Cost: indirection. Reading the retail flow means reading the core *and* the pack.
