# ADR-0018 — Freeze voice, make chat the proving ground

- **Status:** accepted
- **Date:** 2026-09-13
- **Affects:** [0001](0001-livekit-webrtc.md) (LiveKit ingress), [0007](0007-two-mode-redaction.md)
  (the two-mode split), [0011](0011-voice-turn-loop.md) (the turn loop),
  [0012](0012-openrouter-free-models.md) (which models can serve which path)

## Context

Voice is the harder half of this platform and it is built: LiveKit transport, Silero VAD,
Deepgram STT, Cartesia TTS, barge-in, DTMF, the turn loop and a measured 800 ms budget.
Three of those adapters have never run against a real vendor because we have no
credentials, and the 800 ms ceiling is the single tightest constraint in the system.

Chat is the easier half and most of it already exists without having been called a
channel: `src/ccas/api/sessions.py` drives the graph with `Channel.CHAT` and the identical
turn shape voice uses, and `src/static/index.html` is a working client against it.

Three things make the swap worth doing now rather than later.

**The core is already channel-agnostic, verifiably.** Nothing in `src/ccas/` outside
`voice/` imports `voice/`; the dependency runs one way. No node, policy or graph edge
branches on `channel`. Both paths call
`graph.ainvoke({"turns": [turn], "turn_index": n})` — voice differs only in also seeding
`latency`.

**Free models become usable.** ADR-0012 measured the free OpenRouter models at ~2,025 ms
median against a 90 ms voice router budget — 22× over, which is why that ADR restricts them
to the offline and chat paths. On chat, two seconds is an ordinary reply. The pivot turns
the whole pipeline green on models we already have, and drops three uncredentialed vendors
off the critical path.

**Chat can afford better redaction.** `graph/context.py` hardcodes
`RedactionMode.REALTIME`, which skips the spaCy NER pass to fit a 3 ms slice (ADR-0007).
Chat has no such slice. BATCH costs 111 µs → 7.0 ms, which is nothing when a human is
typing, and it closes ADR-0007's stated residual risk: *a name spoken outside a
self-identification cue reaches the provider.*

## Decision

**Voice is frozen, not removed.** All of `src/ccas/voice/`, its tests, its budget and its
ADRs remain and must keep passing. What stops is voice acting as a live constraint on core
work: its suites leave the default run and move behind `make voice`, and the 800 ms budget
is no longer a gate on model selection.

**Chat becomes the proving ground for the core.** The bar is a chat CCAS whose router
accuracy is measured rather than asserted.

**Freezing introduces one risk and it is handled explicitly.** A frozen suite outside the
default run drifts silently: someone edits the core, voice breaks, nobody notices.
`tests/test_module_5_voice_frozen.py` is the tripwire and *does* run by default. It asserts
the core still does not import `voice`, that the voice code and tests still exist (a freeze
that quietly became a deletion would pass every other check), and that the freeze is
documented where someone would look for it.

## Alternatives considered

- **Carry on with voice.** Blocked on credentials for three adapters, and every model
  decision stays hostage to a 90 ms router budget no free model can meet.
- **Delete voice and rebuild later.** Throws away a measured 800 ms budget, a working
  barge-in implementation and four ADRs' worth of reasoning, to save the cost of a marker.
- **Keep voice in the default run.** Green today, and it would keep failing the build for
  a channel nobody is developing — the reliable way to teach a team to ignore a red suite.
- **Branch on `channel` in the core.** The obvious shortcut and the one Rule 1 exists to
  prevent. Channel differences are configuration, and belong in a profile loaded like a
  domain pack.

## Consequences

- `make test` no longer covers voice; `make voice` does, and `make check` runs both.
- The chat path needs its own latency budget. `configs/latency_budget.yaml` is voice-shaped
  (VAD, STT, TTS slices) and `tests/latency/` asserts against it. A chat budget is a
  separate contract, not a relaxation of this one — Rule 3 still forbids raising a budget
  to make a test pass.
- `SlotSpec.dtmf_capturable` stays in the taxonomy contract. It is inert on chat and
  meaningful on voice; removing it would break the frozen module for no gain.
- **Resuming voice means running `make voice` green before trusting anything.** The freeze
  guarantees the code is present, not that it still works against a core that moved.
