# ADR 0012 — OpenRouter and free models: what they can and cannot serve

Status: accepted · 2026-09-13
Amends ADR-0003 (which paired Anthropic with vLLM); that pairing is no longer in use,
but its principle — no node may depend on a single model — is preserved and strengthened.

## Context

The platform moved to OpenRouter with free models. That changes three things at once:
who the provider is, what "parity" means when every model sits behind one gateway, and
whether the voice latency budget is reachable at all.

Four models were nominated:

| model | free | `response_format` | reasoning | context |
|---|---|---|---|---|
| `nex-agi/nex-n2.5-pro:free` | yes | **yes** | yes | 262k |
| `nvidia/nemotron-3.5-lightning:free` | yes | no | yes | 1M |
| `inclusionai/ling-3.0-flash-vl:free` | yes | no | yes | 262k |
| `inclusionai/ling-3.0-flash-fin:free` | yes | no | yes | 262k |

Three of four **cannot be constrained by the server**. The router, the taxonomy labeler
and the responder all depend on schema-valid JSON.

## Decision

### 1. Structured output is a property of the model, not the provider

`ModelBinding.structured_mode` is declared per binding:

- `response_format` — the server constrains decoding; the body is JSON by construction.
- `prompt` — JSON is requested in the system prompt and extracted from whatever comes
  back, via `llm/json_repair.py`.

The extractor is deliberately narrow: it unwraps fences and finds a brace-balanced
object in prose. It does **not** repair malformed JSON — a model that emitted broken
JSON misunderstood the task, and guessing at its intent would put invented values into a
caller's transcript.

### 2. Parity is between models, not vendors

`load_bindings` now requires **at least two variants per node, naming distinct models**.
A variant is usually a provider; when everything is behind one gateway it is a second
model. Rule 6's intent — that no node can silently depend on one model — is unchanged,
and is now enforced more tightly: two variants naming the same model are rejected,
because comparing a model against itself measures nothing.

### 3. Free models cannot serve the voice call path

Measured over four real journeys against the live API:

| | measured | budget | over by |
|---|---|---|---|
| router | median **2,025 ms** (1,837–2,712) | 90 ms | **22×** |
| full turn | median **2,033 ms**, max 9,325 ms | 800 ms | 2.5× |

This is not tuning distance. A caller hearing two seconds of silence after every
utterance is not a product, and no amount of prompt or parameter work closes a 22× gap.

So the split is explicit:

- **Offline and chat paths are fully served by free models** — mining, labelling,
  judging, summarising, and the workbench. Latency is irrelevant there and quality was
  good: grounded answers, correct escalations, honest refusals on ambiguous input.
- **The voice call path needs a low-latency provider.** `latency_budget_ms` stays at its
  real values so the gap is visible rather than quietly redefined. Raising the budget to
  match what free models deliver would make Rule 3 meaningless.

`timeout_ms` and `latency_budget_ms` now say different things on purpose: the first is a
transport ceiling (30 s on the router — a slow model is not an outage), the second is
what voice actually needs. The distance between them is the finding.

## Consequences

- All 10 node×variant bindings are live and returning valid JSON against the real API.
- Free tiers 429 and time out routinely, so the binding retries both with backoff and
  honours `Retry-After`. Both are **converted**, never allowed to escape: a raw httpx
  error propagating out of a graph node takes the whole call down, which is how the
  first real run died.
- Truncation is diagnosed rather than reported as missing JSON. `nemotron-lightning`
  reasons inline whatever `reasoning: false` says and spent all 512 router tokens before
  emitting anything; "truncated at 512 tokens" says what to change, "no JSON" does not.
  That model was moved to the offline labeler, where its 1M context is an asset and its
  26-second responses cost nothing.
- The Anthropic and vLLM bindings remain in the codebase and in `ProviderName`. They are
  not configured, and the factory now defers to the SDK's own credential chain rather
  than demanding an env var — a machine authenticated by `ant auth login` was previously
  rejected outright.
- Model availability on a free tier is not a stable contract. The capability table above
  was verified on 2026-09-13 against `GET /api/v1/models`; re-check it before trusting a
  binding that suddenly fails.
