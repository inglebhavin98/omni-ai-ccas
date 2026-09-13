# ADR 0011 — The voice turn loop, and what the budget gate actually proves

Status: accepted · 2026-09-13

## Context

Phase 5 had to answer three questions the earlier phases had deferred: how a call is
structured, how barge-in works, and what "the 800 ms budget passes" can honestly mean
when two of the seven stages belong to vendors nobody here can call.

## Decision

### 1. One graph invocation per caller turn

A call is many invocations sharing one checkpointed state. An invocation ends when the
platform has spoken and is waiting for a reply. That is what makes the turn ceiling, the
clarification bound and the latency ledger mean anything — each is per turn, and a turn
is now a thing with edges.

### 2. Playback is a task, not an await

The obvious implementation awaits synthesis inline. It also makes barge-in
*structurally impossible*: the frame loop is blocked on speaking, so it cannot observe
the caller starting to talk. The first end-to-end run showed exactly this — `clears: 0`
on a call where the caller talked over the bot throughout.

Playback therefore runs as an `asyncio.Task` and the loop keeps consuming inbound audio.
Barge-in then does three things in a fixed order: clear queued playback, cancel
generation, cancel the in-flight turn. Cancelling generation first leaves whatever is
already queued still playing, which is the thing the caller is complaining about.

A 250 ms guard suppresses the "yeah"/"mm" overlaps that callers produce constantly
without meaning to interrupt.

### 3. The ledger is seeded, passed through, and finished

The voice side measures VAD, STT and redaction; the graph adds routing, tools and the
LLM; the voice side writes back TTS time-to-first-byte. The ledger travels *through* the
graph as part of state rather than being reassembled afterwards, so a stage that stops
being measured shows up as a zero rather than silently disappearing.

### 4. The budget gate charges vendor stages at their budget

`tests/latency/test_e2e_rtt.py` runs the real loop against scripted audio, STT, TTS and
LLM. Those stand-ins are effectively free, so a naive gate would measure ~5 ms turns,
pass forever, and prove nothing.

Instead the three vendor stages are charged at their **budgeted** cost. What the gate
then proves is the useful half: with STT, LLM and TTS at full price, everything the
platform itself owns — redaction, routing, policies, tool dispatch, grounding, the loop —
fits in what remains. Measured: **580 ms p95 against an 800 ms ceiling**, with the
platform's own stages under 1 ms.

What it does **not** prove is that Deepgram, Cartesia or a real model meet their slices.
Per ADR-0001 those numbers are a floor, not a forecast, and `scripts/bench_latency.py`
prints that caveat every run.

## Consequences

- Barge-in is real and tested, including the negative case. The test waits on the
  *condition* (playback past the guard) rather than racing the scheduler — the first
  version passed or failed depending on machine load.
- Deterministic timing needs a shared clock. `ScriptedTransport` and `ScriptedTts` both
  take a `pace_factor`; set differently, one races ahead and barge-in timing becomes
  meaningless. Set to zero, a test runs instantly and cannot exercise interruption.
- DTMF is a first-class input, not an afterthought: a keypad turn skips VAD and STT
  entirely and is asserted to do so. Only slots the taxonomy marks `dtmf_capturable`
  accept it — typing a reference is normal, typing a reason for return is not.
- **A keypad answer is redacted wholesale, not scanned.** The first full voice run put
  `884210` straight into the transcript: six bare digits match no pattern, so the
  scanner left them. Pattern matching cannot help when *context* is what makes a value
  sensitive — the caller was asked for a reference and gave one. The slot's declared
  `pii_entity` decides, falling back to treating any run of four or more digits as an
  account reference. A single digit is a menu choice and stays readable.
- `AudioFrame` is the most sensitive object in the system — pre-STT, therefore
  pre-redaction. It is never a Pydantic field, never logged, and renders blind.
- The worker **refuses to start** when preflight objects, listing every problem at once.
  A worker that answered calls without redaction, a taxonomy, or a provider would be
  worse than one that does not answer.
- The Deepgram, Cartesia and LiveKit adapters have **not met their live services**. Wire
  formats and re-chunking are pinned by tests; the room loop is not written. Tracked in
  `docs/future-scoped-work.md`.
