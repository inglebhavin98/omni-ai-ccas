# `evals/` — M6: measuring the platform

Offline scoring and provider parity. Ragas and DeepEval land with the rest of M6; what
ships today is parity.

## Files

| File | Holds |
|---|---|
| `parity.py` | Cross-variant comparison, verdicts, throttle handling |

## Unmeasured is not agreement

The rule that makes parity numbers mean something
([ADR-0014](../../../docs/adr/0014-parity-verdicts-and-throttling.md)):

- a case nobody served — a 429, an exhausted quota — is **unmeasured**, and leaves the
  denominator
- a case that **failed** is divergent
- a run with too few measurements **skips**, with the count in the reason. It never passes
- **zero measured cases is 0% agreement, not 100%**

That last one is the trap. An empty result set trivially satisfies "no divergences
found", which is how a parity gate silently stops testing anything.

Any change to a prompt, tool schema, or graph node must pass
`tests/evals/test_provider_parity.py`. If two variants diverge beyond threshold, fix it
or record the divergence in an ADR — do not quietly pin to one.

Grading uses Bitext and NatCS gold labels. Never grade a taxonomy against the corpus it
was mined from.

```bash
make evals
```
