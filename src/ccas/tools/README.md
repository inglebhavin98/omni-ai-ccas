# `tools/` — M4: the tool layer

The LLM chooses **which** declared tool to call. It never constructs a URL, a query, or
an unregistered call. Every tool comes from the pack registry.

## Files

| File | Holds |
|---|---|
| `registry.py` | Tool lookup. Shared tools in `configs/tools.yaml` (`domain: core`); a pack adds its own and may not shadow one |
| `backend.py` | Backend protocol — the seam to a real system of record |
| `mock_backend.py` | Fixture-driven backend from `domains/<pack>/tool_fixtures.yaml` |
| `executor.py` | Argument validation, placeholder restore, dispatch, output redaction |

## The executor is a security boundary

Two things happen there and nowhere else
([ADR-0009](../../../docs/adr/0009-tool-layer-and-policy-order.md)):

1. **Placeholders are restored immediately before dispatch.** A registered backend is an
   internal system of record that already holds the caller's data — reaching it is the
   point of the call — so it gets real values. Nothing else does.
2. **Output is redacted before a model can read it.** Declaring `pii_output_fields` is
   not sufficient; the executor redacts regardless, because a backend can return a field
   nobody declared.

Arguments are validated against the pack's JSON Schema before dispatch. Naming is
`verb_noun` snake_case — `get_order_status`.

```bash
uv run pytest tests/unit/tools -q
```
