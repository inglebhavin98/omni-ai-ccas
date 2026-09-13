# ADR 0009 — Tool layering, total-redaction of tool output, and policy order

Status: accepted · 2026-09-13

## Context

Phase 4 needed three decisions the roadmap had left open: where tool definitions live,
how much of a tool's output to redact, and in what order the policies run.

A proposal arrived to read tools "strictly from `configs/tools.yaml`". Taken literally
that breaks Rule 1 — `track_order` and `check_claim_status` are domain vocabulary, and a
global file holding both teaches the core two verticals.

## Decision

### 1. Tools layer; they do not live in one place

`configs/tools.yaml` holds tools that mean the same thing in every vertical —
`transfer_to_human`, `schedule_callback`, `send_link`, `verify_identity`. Each pack
declares its own on top. A pack **may add; it may not shadow** a global name.

`GlobalToolSet` rejects any tool whose `domain` is not `core`, so a vertical's tool
cannot be smuggled into the shared set. This is the same layering redaction patterns
already use, which is the point: one mechanism, applied twice.

Mock responses live in `domains/<pack>/tool_fixtures.yaml` rather than in the backend, so
the mock stays free of vertical vocabulary and the Phase 6 acceptance criterion — the
healthcare pack running the suite with no `src/` change — still holds.

### 2. Every string in a tool result is redacted, not just the declared fields

`ToolSpec.pii_output_fields` trusts whoever wrote the spec to have anticipated every
field a backend might return. That is a bad assumption about a system integration.

So both run: declared fields are **replaced wholesale** (a name has no lexical shape and
would survive a pattern scan), and every remaining string goes through the redaction
pipeline. A result that does not come back clean is returned as an error with its data
withheld — the same rule ingestion follows.

The guarantee is not expressed by nesting `RedactedText` inside `ToolResult.data`; that
would double-store hashes and make the CTI payload unreadable. It is enforced by the
executor redacting in place, with `ToolResult.safe_for_model` as the gate a model or a
`HandoffContext` consults.

### 3. Retries are bounded by total wall time, not per-attempt timeout

Measured: a two-attempt retry on a 900 ms tool spent **1,924 ms** — each attempt inside
its own timeout while the turn blew its 150 ms slice. The executor now takes a
`total_budget_ms` and shortens each attempt to whatever is left. Same case, **151 ms**.

### 4. Policy order is `risk → retry → sentiment → confidence`

Each policy is pure: state and thresholds in, a verdict out. First decisive verdict wins,
and the winner's name becomes `EscalationDecision.triggered_by`, so a handoff states
which rule fired rather than implying one.

The order is the decision. `risk` first because a regulated intent must not be attempted
whatever else is true. `retry` next because an exhausted call cannot be repaired by
asking again. **`sentiment` before `confidence`** because the reverse produces the worst
turn the system can take: asking an already-furious caller to rephrase.

Sentiment also fires on *slope* — three consecutive rises ending above half the threshold
— because an absolute threshold alone fires a turn too late.

## Consequences

- Adding a vertical still adds no code: tools, fixtures and thresholds are all pack data.
- Tool output is readable by a human agent, because redaction is now precise enough to
  leave a delivery ETA alone. Getting there required dropping `date_numeric` below the
  default score threshold — it had been turning every ISO date into `[DATE_OF_BIRTH_1]`.
- The executor is the only place a guarantee lives, so a new backend cannot weaken one.
  Backends do transport; they do not validate, authorise, time out, retry or redact.
- Policies are trivially testable and their ordering is itself under test. The cost is
  that a policy cannot consult anything it was not given — deliberate, since a policy
  able to reach the whole pack would grow a dependency on a vertical's shape.
- `src/ccas/tools/` and `src/ccas/policies/` sit at the top level rather than under
  `orchestration/`. Both are consumed outside the graph — the copilot will invoke tools,
  and the eval harness will replay policies — so nesting them under the graph would have
  been wrong within a phase.
