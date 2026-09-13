# ADR 0003 — Ship both LLM bindings, keep them at parity

Status: accepted · 2026-09-12

## Context

The brief asked for an open-source stack while also naming commercial streaming vendors
(Deepgram, Cartesia). The honest reading is that "open source" means **no lock-in**, not
**no hosted model**.

Choosing a single provider now decides a question we have no data to answer: nobody has
measured this workload's TTFT, tool-calling reliability or cost on either option. And a
provider abstraction with exactly one implementation rots — the interface quietly grows
the shape of whichever SDK is behind it.

## Decision

`LLMProvider` has two concrete bindings, both shipped and both supported.

| | Anthropic | vLLM |
|---|---|---|
| Router, judge | `claude-haiku-4-5` | `llama-3.1-8b-instruct` / `llama-3.3-70b-instruct` |
| Task agents, labeler, summarizer | `claude-opus-5` | `llama-3.3-70b-instruct` |
| Controls | adaptive thinking, `effort`, prefix caching | `temperature`, guided decoding |

`configs/models.yaml` declares **every node for both providers**, and `load_bindings`
refuses a single-provider node. `tests/evals/test_provider_parity.py` gates divergence at
an agreement threshold.

## Consequences

- Request shaping differs per provider and is unit-tested per provider (`build_kwargs`,
  `build_payload`) rather than flattened to a lowest common denominator.
- Anthropic capability gating lives in `llm/capabilities.py`: the Messages API *rejects*
  unsupported parameters rather than ignoring them, so `thinking` and `effort` are sent
  only to models that accept them. An unrecognised model id gets the conservative shape
  — a plainer request beats a 400 at startup.
- Server-side refusal fallbacks are enabled on the Opus-5-class bindings.
- A provider outage becomes a config change rather than a migration.
- `ProviderUnavailableError` is raised rather than auto-failing-over. A silent failover
  would make every parity number meaningless.
- Cost: two implementations, two test suites, and a standing parity eval to maintain.
