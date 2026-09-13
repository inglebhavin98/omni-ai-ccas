# `tests/` — the gates

Test first. One micro-PR is one source file and its test, red → green → refactor. Never
weaken or skip a test to make a build pass.

## Layout

| Path | Holds |
|---|---|
| `test_module_<N>_<name>.py` | Per-module gate — is this module usable as a whole? |
| `unit/` | Mirrors `src/ccas/` package for package |
| `unit/scripts/` | Covers `scripts/` helpers |
| `security/` | Rule 1 and Rule 2 invariants |
| `latency/` | Rule 3 budget assertions |
| `evals/` | Provider parity (Rule 6) |
| `integration/` | Needs a live dependency — marked, not run by default |
| `fixtures/` | Synthetic or public-dataset only. **Never real customer data** |

## The two that fail loudest

- `security/test_no_domain_literals.py` scans `src/ccas` and `src/cli` for vertical
  vocabulary. The fix for a failure is a field on `DomainPack`, never a branch.
- The redaction property tests fuzz generated identifiers with Hypothesis. Coverage floor
  is 100% on `schemas/` and `redaction/`, 85% elsewhere.

## Running them

```bash
make check          # ruff format + ruff + mypy --strict + pytest -- the commit gate
make test-fast      # unit only
make security       # leakage + domain-agnosticism
make latency        # budget gate
make evals          # provider parity
```

If `make check` dies at the lint step with an `en-core-web-sm` fetch error, that is a
network failure in dependency resolution, not a code failure — run the venv binaries
directly (`docs/future-scoped-work.md` 9.14).

## What a new thing needs

| Adding | Also add |
|---|---|
| A PII pattern | A `fixtures/pii_corpus/` entry **and** a Hypothesis property |
| A graph node | A reachability test **and** a failure-path test |
| Latency-touching code | A `tests/latency/` assertion |
| A module | Its gate test, a demo stage, and structured logs (Rule 11) |

No network in unit tests. External calls use recorded cassettes or fakes.
