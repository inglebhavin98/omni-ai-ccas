# Architecture Decision Records

One file per major trade-off or technology decision, numbered sequentially and never
renumbered once merged. A decision that reverses an earlier one supersedes it by
reference rather than editing it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-livekit-webrtc.md) | LiveKit WebRTC only for Phase 1 voice ingress | accepted |
| [0002](0002-langgraph-state.md) | Deterministic `StateGraph`, not an agent loop | accepted |
| [0003](0003-hybrid-llm-provider.md) | Ship both LLM bindings, keep them at parity | amended by 0012 |
| [0004](0004-dataset-role-separation.md) | Corpora have fixed, enforced roles | amended by 0015 |
| [0005](0005-redaction-as-type-gate.md) | Redaction is a type gate, not a pipeline step | accepted |
| [0006](0006-domain-pack-over-branching.md) | Verticals are data, not code paths | accepted |
| [0007](0007-two-mode-redaction.md) | Regex on the call path, NER in batch | amended by 0019 |
| [0008](0008-mining-signal-selection.md) | What counts as signal when mining a taxonomy | amended by 0013 |
| [0009](0009-tool-layer-and-policy-order.md) | Tool layering, total redaction of tool output, policy order | accepted |
| [0010](0010-placeholder-vault.md) | The session placeholder vault | accepted |
| [0011](0011-voice-turn-loop.md) | The voice turn loop, and what the budget gate proves | accepted |
| [0012](0012-openrouter-free-models.md) | OpenRouter and free models: what they can and cannot serve | accepted |
| [0013](0013-aixblock-corpus-is-insurance-and-undiarized.md) | The AIxBlock corpus is insurance sales calls, and it is not diarized | accepted |
| [0014](0014-parity-verdicts-and-throttling.md) | A throttled parity case is unmeasured, not divergent | amended by 0021 |
| [0015](0015-natcs-is-supervision-not-a-trajectory.md) | NatCS is supervision, not a trajectory | accepted |
| [0016](0016-aixblock-turn-recovery.md) | Recovering turns from an undiarized corpus, and the gap that defines an utterance | accepted |
| [0017](0017-mean-centre-before-clustering.md) | Mean-centre embeddings before clustering | accepted |
| [0018](0018-freeze-voice-pivot-to-chat.md) | Freeze voice, make chat the proving ground | accepted |
| [0019](0019-channel-profiles.md) | Channel differences are data, not branches | accepted |
| [0020](0020-adopt-a-published-label-set.md) | Adopting a published label set is not mining | accepted |
| [0021](0021-a-timeout-is-read-against-the-run.md) | A timeout is unmeasured only if the model answered elsewhere | accepted |

## When to write one

Write an ADR when a choice would otherwise be reconstructed from a diff months later:
a technology selection, a locked-stack change (CLAUDE.md Rule 5), a latency budget
change (Rule 3), or any trade-off where the rejected option was genuinely reasonable.

Do not write one for a refactor, a bug fix, or a choice with no live alternative.

## Format

```
# ADR NNNN — <decision in the imperative>

Status: proposed | accepted | superseded by ADR-NNNN · YYYY-MM-DD

## Context     what forced a decision; the constraints that were actually binding
## Decision    what we chose, stated so it can be checked against the code
## Consequences  what this costs, what it forecloses, what now has to stay true
```
