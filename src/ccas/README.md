# `src/ccas` — the platform

Six modules over one frozen contract layer. The dependency rule is the thing to hold on
to: **`schemas/` imports nothing from this project, and everything else depends on it.**
Nothing depends on anything above it, and nothing in here may import `src/cli`.

```
                       schemas/          frozen contracts, no intra-project imports
                          ^
        ┌─────────────────┼─────────────────┐
    config/          observability/        llm/         cross-cutting
        ^                 ^                 ^
  ┌─────┴─────┬───────────┴───────┬─────────┴──────┐
ingestion/  redaction/         mining/          tools/ policies/
   M1          M2                M3              nodes/ graph/  M4
                                                     ^
                                                  voice/        M5
                                                     ^
                                            copilot/ evals/     M6
```

## Modules

| Package | M | Lines | What it owns |
|---|---|---|---|
| [`schemas/`](schemas/README.md) | — | 1,987 | Every contract. Frozen Pydantic v2. The only mutable model is `SessionState` |
| [`config/`](config/README.md) | — | 319 | `Settings`, domain-pack loader, latency budget |
| [`observability/`](observability/README.md) | — | 263 | Structured logs that refuse raw content, OTel spans, correlation ids |
| [`llm/`](llm/README.md) | — | 1,130 | `LLMProvider` protocol and its bindings; only `RedactedText` goes in |
| [`ingestion/`](ingestion/README.md) | M1 | 596 | Corpus adapters → redactor → `CallLog`, and the dataset-role gate |
| [`redaction/`](redaction/README.md) | M2 | 1,131 | Regex + local Presidio + leak detector, and the placeholder vault |
| [`mining/`](mining/README.md) | M3 | 1,679 | embed → HDBSCAN → LLM label → `IntentTaxonomy` |
| [`tools/`](tools/README.md) | M4 | 704 | Tool registry, argument validation, executor with output redaction |
| [`policies/`](policies/README.md) | M4 | 397 | Pure decision functions — confidence, risk, retry, sentiment |
| [`nodes/`](nodes/README.md) | M4 | 701 | One graph node per turn behaviour |
| [`graph/`](graph/README.md) | M4 | 717 | `StateGraph` assembly, routing, checkpointing |
| [`voice/`](voice/README.md) | M5 | 1,173 | LiveKit transport, VAD, barge-in, DTMF, turn loop |
| [`evals/`](evals/README.md) | M6 | 214 | Provider parity; Ragas/DeepEval land with the rest of M6 |
| `copilot/` | M6 | stub | CTI handoff and CRM summary. Not built — the package exists so the boundary does |
| `orchestration/` | — | stub | Superseded by `tools/` `policies/` `nodes/` `graph/`. Kept as a namespace only |
| [`api/`](api/README.md) | — | 646 | Workbench API + console. Loopback only, never a production endpoint |

## The rules that bind this directory

These come from [`CLAUDE.md`](../../CLAUDE.md) and are enforced by tests, not convention.

- **No domain vocabulary anywhere in here** (Rule 1). No `claim`, `order`, `patient`,
  `premium`. Verticals are data in `domains/<pack>/`, never a branch.
  `tests/security/test_no_domain_literals.py` fails the build on a literal.
- **Nothing unredacted crosses a boundary** (Rule 2). `CallLog` and `HandoffContext`
  refuse construction unless the redaction status is `CLEAN`; only `RedactedText` reaches
  a provider. If the redactor is unavailable the status is `UNVERIFIED` and egress is
  forbidden — fail the turn, never fail open.
- **800 ms p95 round trip** (Rule 3). Async everywhere on the call path, explicit
  timeouts on every external call, no blocking I/O in an async function.
- **Deterministic orchestration** (Rule 4). Explicit nodes and edges. No agent loop, no
  auto-chaining, and a reachable escalation path from every node.
- **Module ≤ 400 lines, function ≤ 50** (Rule 8). Split rather than nest.

## Conventions

`from __future__ import annotations` in every file. Full type hints; `mypy --strict` is
clean across 129 modules. Structured logging only — `structlog`, never `print`, and every
line carries a `correlation_id`. Pure functions in `mining/`, `redaction/` and
`policies/`; I/O lives at the edges.

Tests mirror this tree under `tests/unit/`, with a per-module gate test at
`tests/test_module_<N>_<name>.py`.
