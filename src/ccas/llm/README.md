# `llm/` — provider abstraction

One protocol, several bindings, and a hard input constraint: **only `RedactedText` may be
passed to a provider.** That is Rule 2 at the last boundary before text leaves the
process.

## Files

| File | Holds |
|---|---|
| `base.py` | `LLMProvider` protocol, `LLMProviderError`, `ProviderRateLimitedError` |
| `bindings.py` | `load_bindings()` over `configs/models.yaml`; resolves a node to a model |
| `factory.py` | `build_provider()` — binding → live client, with an actionable error |
| `openrouter_provider.py` | OpenAI-compatible gateway. The configured default |
| `anthropic_provider.py` | Anthropic binding |
| `vllm_provider.py` | Self-hosted OpenAI-compatible binding |
| `capabilities.py` | Per-model `structured_mode` — most free models ignore `response_format` |
| `json_repair.py` | Recovering structured output from models that will not honour a schema |
| `prompt.py` | Prompt assembly with a stable, cacheable prefix |

## Two rules specific to this package

**No node depends on a single model** (Rule 6). Every node in `configs/models.yaml`
declares at least two variants naming distinct models, and `load_bindings` refuses
anything less. A case nobody served — a 429, an exhausted quota — is *unmeasured*, not
divergent, and leaves the denominator
([ADR-0014](../../../docs/adr/0014-parity-verdicts-and-throttling.md)).

**`structured_mode` is per model, not per provider**
([ADR-0012](../../../docs/adr/0012-openrouter-free-models.md)). Getting it wrong fails
silently and far away from the cause.

Hard-coding a model id outside `configs/models.yaml` is a defect. Free models serve the
offline and chat paths; the voice call path needs a low-latency provider.

```bash
make evals                     # provider parity
uv run pytest tests/unit/llm -q
```
