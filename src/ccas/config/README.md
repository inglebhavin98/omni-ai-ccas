# `config/` — settings, domain packs, budgets

Configuration is data. Nothing in the platform reads `os.environ` directly; it reads
`Settings`, which is `pydantic-settings` and validates at startup rather than at first
use.

## Files

| File | Holds |
|---|---|
| `settings.py` | `Settings` — paths, provider keys, feature flags. One source of truth |
| `domain_loader.py` | `load_pack()` → `DomainPack`; raises `DomainPackNotFoundError` |
| `budget.py` | Per-stage latency budgets from `configs/latency_budget.yaml` |

## Domain packs are the whole point of Rule 1

A vertical is a directory, not a branch. `domains/<pack>/pack.yaml` declares intents,
slots, tools, queues, thresholds, prompts and redaction patterns; the core loads one and
behaves differently without containing a single `if domain == ...`
([ADR-0006](../../../docs/adr/0006-domain-pack-over-branching.md)).

Three packs exist: `retail`, `healthcare`, `insurance`. The third was created from
observed data rather than a design sketch — see
[ADR-0013](../../../docs/adr/0013-aixblock-corpus-is-insurance-and-undiarized.md).

A feature is complete only when it works for more than one pack. `load_taxonomy` also
rejects an intent whose tools require arguments its slots do not cover, because that call
would otherwise die on a schema violation mid-conversation.

```bash
uv run python -m cli.demo pack healthcare      # inspect a loaded pack
uv run pytest tests/unit/config -q
```
