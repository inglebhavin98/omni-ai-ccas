# ADR-0019 — Channel differences are data, not branches

- **Status:** accepted
- **Date:** 2026-09-13
- **Follows:** [0018](0018-freeze-voice-pivot-to-chat.md)
- **Amends:** [0007](0007-two-mode-redaction.md) — which mode runs is now per channel

## Context

Adding chat as a first-class channel raised the obvious question at the first line of
code: `build_context` hardcoded `RedactionMode.REALTIME`, so chat inherited a redactor
tuned to a constraint chat does not have.

REALTIME exists because voice cannot afford the spaCy NER pass inside a 3 ms slice
(ADR-0007, measured: regex alone 111 µs p99, adding NER 7.0 ms). Chat has no 3 ms slice. A
person is typing. 7 ms is invisible, and it closes the residual risk ADR-0007 names in its
own consequences: *on the live path, a name spoken outside a self-identification cue
reaches the provider.*

The shortcut is one line — `if channel is Channel.VOICE: ...` — and it is the exact shape
Rule 1 exists to prevent for domains. The argument transfers without modification. The
moment a node branches on channel, every new channel is a code change and the core stops
being channel-agnostic, which is the property that made this pivot cheap in the first
place: nothing in `src/ccas/` outside `voice/` imports `voice/`, and no node, policy or
edge reads `channel`.

## Decision

Channel differences load from `configs/channels.yaml` the way a vertical loads from
`domains/<pack>/`. `ChannelProfile` carries only what genuinely differs and has a live
consumer:

| field | why it differs |
|---|---|
| `redaction_mode` | voice cannot afford NER at 3 ms; every text channel can |
| `budget_ms` + `budget_ref` | a text turn has no VAD, STT or TTS stage |
| `supports_keypad_entry` | DTMF means something on a phone call and nowhere else |

`build_context(..., channel=...)` looks the profile up and passes the mode to the pipeline.
**Chat is the default**, because chat is the proving ground now (ADR-0018) and voice is
frozen; voice is opt-in and keeps the trade its slice forces on it.

Three rules keep this from decaying back into branching:

1. **Every member of `Channel` must have a profile.** `load_channel_profiles` refuses a
   file that omits one — a channel the platform can be handed but has no profile for is a
   crash waiting for its first caller.
2. **No default on lookup.** `require()` raises `UnknownChannelError` rather than falling
   back, because the convenient fallback is the weak redactor.
3. **A test scans `src/ccas/` for `if`/`elif` mentioning `Channel.` or `channel ==`** and
   fails on a hit. The mechanism is only worth having if the shortcut stays closed.

`StageBudgets` gains defaults of zero for `vad_ms`, `stt_ms` and `tts_ttfb_ms`. Zero is the
honest value on a text channel: the stage costs nothing because it does not run. The four
stages that are the platform's own work — redact, router, tool, llm — stay required
everywhere, because every channel pays them.

`configs/latency_budget_chat.yaml` is a **separate contract, not a relaxation**. Rule 3
still forbids raising a budget to make a test pass; it does not forbid a different channel
having a different ceiling. 4,000 ms, of which 2,500 ms is the router — because ADR-0012
measured the free models at ~2,025 ms median, unusable against voice's 90 ms and ordinary
here.

## Alternatives considered

- **Branch on channel in `build_context` only.** One `if`, contained, and it would have
  been copied into the second place that needed it within a week. The scan test exists
  because I expect that pressure.
- **Put channel settings in the domain pack.** Wrong axis. Retail-on-chat and
  retail-on-voice share a vertical and differ in channel; duplicating every pack per
  channel multiplies the thing packs exist to avoid.
- **Infer the mode from the latency budget.** Cute, and it couples two things that only
  happen to correlate. A channel could have a tight budget and still afford NER.
- **One budget file with per-channel sections.** Then `tests/latency/` has to know which
  section it is asserting, and the voice budget stops being a single reviewable artefact.

## Consequences

- Chat now runs `BATCH` redaction on the live path: regex + spaCy NER, ~7 ms. The default
  test run got ~40% slower and that is the cost of the stronger guarantee.
- **ADR-0007's residual risk is closed for text channels and remains open for voice.** The
  two-mode split stands; what changed is that only voice is on the weak side of it.
- A new channel is a YAML block. If it ever needs more than these three fields, that is a
  signal the core has grown a channel assumption worth finding.
- `SlotSpec.dtmf_capturable` still exists in the taxonomy contract and is now inert on
  text channels by profile rather than by accident.
