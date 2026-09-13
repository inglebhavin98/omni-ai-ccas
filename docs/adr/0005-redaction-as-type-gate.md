# ADR 0005 — Redaction is a type gate, not a pipeline step

Status: accepted · 2026-09-12

## Context

"Redact before sending to the model" is the kind of rule that holds for about six weeks.
It is enforced by reviewer attention, and reviewer attention is exactly what a debug log
line, a new adapter, or a 2am hotfix does not get.

The original architecture ran ingestion and redaction as sibling stages. That leaves a
window in which an unredacted `CallLog` is a perfectly valid object, and any code holding
one can serialise it, log it, or send it onward.

## Decision

Redaction is expressed in the type system, not in pipeline ordering.

- `RedactedText` is the only string type permitted to cross an external boundary. It
  cannot exist without a `RedactionReport` explaining how it got that way.
- `CallLog`, `HandoffContext` and `LLMRequest` **refuse construction** unless every
  text-bearing field carries an egress-permitted report. There is no unredacted instance
  to leak.
- `RedactionStatus.CLEAN` must be *earned*: a validator rejects `CLEAN` without a passed
  leak check, without an engine having run, or with residual pattern hits.
- An unavailable engine yields `UNVERIFIED`, which blocks egress. The system fails closed.
- A `RedactionReport` carries entity types, counts and pattern *names* — never matched
  text — so a report is safe to log even though its subject is not.
- The log layer enforces the same rule independently: `_forbid_raw_content` raises on any
  event carrying a raw-content key.

Module 2 therefore becomes a library consumed by Module 1 offline and Module 5 in
realtime, rather than a stage between them.

## Consequences

- Realtime redaction is on the call path and must be paid for: 3 ms of the 800 ms budget,
  regex-first with Presidio consulted only on a regex miss.
- The invariant is testable without a running system — `tests/security/test_egress_gate.py`
  asserts that unredacted egress payloads are unconstructible.
- Fixtures need a deliberate affordance (`tests/factories.py`), and the bypass setting is
  rejected outside `env=local`.
- Cost: every text field in the contract layer is a wrapper type rather than a `str`,
  which is more verbose at every construction site. Accepted — verbosity at the call site
  is cheaper than a breach.
