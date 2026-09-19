# omni-ai-ccas

A domain-agnostic, AI-native Customer Care Experience Platform built to fully replace
legacy IVR/CCaaS stacks (Genesys, Cisco). The same core serves retail, healthcare,
banking, telecom, utilities and SaaS by loading a different **domain pack** — no
vertical-specific branches in `src/`.

> Engineering rules (domain-agnosticism, zero-leakage PII, the 800 ms latency budget,
> deterministic orchestration, the locked stack) live in [`CLAUDE.md`](./CLAUDE.md).
> They are binding.

## Modules

| Module | Package | Responsibility |
|---|---|---|
| M1 | `ingestion/` | AIxBlock · NatCS · Bitext · CSV adapters → redactor → `CallLog` |
| M2 | `redaction/` | regex + Presidio (ONNX, local) + leak detector |
| M3 | `mining/` | embed → HDBSCAN → LLM label → `IntentTaxonomy` (L1/L2/L3) |
| M4 | `tools/` `policies/` `nodes/` `graph/` | tool registry + executor, pure policies, LangGraph `StateGraph` over `SessionState` |
| M5 | `voice/` | LiveKit WebRTC · Silero VAD · STT · TTS · barge-in · DTMF |
| M6 | `copilot/`, `evals/` | CTI handoff, CRM summary, Ragas/DeepEval, async judge |

Cross-cutting: `schemas/` (frozen contracts), `llm/` (provider abstraction — OpenRouter
configured, Anthropic and vLLM bindings available), `config/` (settings + pack loader),
`observability/` (OTel + audit), `api/` (loopback workbench).

Every package carries its own README. Start at [`src/ccas/README.md`](src/ccas/README.md).

## Documentation

| Document | Answers |
|---|---|
| [`docs/how-it-works.md`](docs/how-it-works.md) | **Start here.** The whole project in plain language — no expertise assumed |
| [`docs/design-doc.md`](docs/design-doc.md) | Architecture, data flows, subsystem integration |
| [`docs/tech-spec.md`](docs/tech-spec.md) | Data models, API contracts, latency budgets |
| [`docs/skills.md`](docs/skills.md) | Project rules, tooling, agent capabilities |
| [`docs/adr/`](docs/adr/README.md) | Why each major choice was made |
| [`docs/future-scoped-work.md`](docs/future-scoped-work.md) | Deliberately deferred work |

## Quick start

```bash
uv sync --all-extras --group dev
cp .env.example .env
make check

# manual review console -- see every stage, live or pending
uv run python -m cli.demo pipeline "where is my delivery"
uv run python -m cli.demo gate         # watch the zero-leakage gate refuse
uv run python -m cli.demo datasets     # corpus role matrix

# interactive console at http://127.0.0.1:8000 (loopback only)
make workbench
```

## Datasets

The three corpora have fixed, enforced roles — see [`data/README.md`](data/README.md)
and [ADR-0004](docs/adr/0004-dataset-role-separation.md).

| Corpus | Role |
|---|---|
| `data/raw/aixblock` | transcript ingestion · redaction corpus · **intent mining** |
| `data/raw/bitext` | **NLU ground truth** · slot extraction · judge evaluation |
| `data/raw/natcs` | **NLU ground truth in natural phrasing** · judge evaluation ([ADR-0015](docs/adr/0015-natcs-is-supervision-not-a-trajectory.md)) |

## Status

| Phase | Modules | State |
|---|---|---|
| 1 | contracts · config · LLM bindings · observability | **shipped** |
| 2 | M1 ingestion · M2 zero-leakage redaction | **shipped** |
| 3 | M3 intent mining & taxonomy | **shipped** (needs a real corpus to produce a real taxonomy) |
| 4 | M4 LangGraph agentic mesh | **shipped** |
| 5 | M5 LiveKit voice engine | **shipped, then frozen** ([ADR-0018](docs/adr/0018-freeze-voice-pivot-to-chat.md)) — complete but paused; vendor adapters never ran against a live service. Kept green under `make voice` |
| 6 | M6 copilot & evaluation suite | **M6a shipped** — handoff, egress gate, CRM adapter, gate test, demo stage. **M6b partial** — `evals/parity.py` and `evals/router_accuracy.py` have produced real numbers ([ADR-0021](docs/adr/0021-a-timeout-is-read-against-the-run.md)); the async judge and Ragas/DeepEval are not built |

Redaction runs in two modes (ADR-0007): **regex only on the call path** (111 µs p99,
inside the 3 ms slice) and **regex + local NER in batch** (7.0 ms) for anything stored,
mined, or shown to a human. Zero leakage is enforced by contract validators and proved
by a Hypothesis fuzz over generated identifiers.

## A note on the history

The git history is **reconstructed, not bisectable.** The commits group the work into the
order it is best *read* -- contracts, then the layers that depend on them -- but the files
were written interleaved, so intermediate commits do not pass their own tests. Only `HEAD`
is green. `git bisect` will mislead you.

What the messages are good for is the *why*: several commits reverse an earlier decision,
and the ADR they cite explains what the evidence was. `fix(datasets): NatCS is supervision,
not a trajectory` corrects a dataset role that was assigned backwards, and
`feat(mining): mean-centre before clustering` carries an ADR that corrects its own headline
after the number it quoted failed to survive a larger sample.

## Licence

MIT — see [`LICENSE`](LICENSE).

The upstream corpora are **not** covered by it and carry their own terms; none of their
data is committed. `domains/retail/taxonomy.json` is derived from the Bitext dataset.
See [`NOTICE.md`](NOTICE.md).
