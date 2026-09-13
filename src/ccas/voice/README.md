# `voice/` — M5: the call path (**frozen**)

WebRTC in, speech out, inside 800 ms p95.

> **Frozen as of 2026-09-13 — [ADR-0018](../../../docs/adr/0018-freeze-voice-pivot-to-chat.md).**
> Chat is the proving ground; voice is complete but paused. Three vendor adapters
> (Deepgram, Cartesia, LiveKit) have never run against a live service for want of
> credentials, and the 800 ms ceiling is the tightest constraint in the system.
>
> The code and its tests stay and **must keep passing** — but under `make voice`, not
> `make test`. The 800 ms budget no longer gates model selection, which is what makes the
> free OpenRouter models usable on chat (~2,025 ms median: impossible for a 90 ms voice
> router, ordinary for a reply in a chat window).
>
> Resuming voice means this package going green again first.

```bash
make voice        # the frozen channel -- excluded from the default suite
```

Everything below describes the design as built, and is what resuming it would return to.

## Files

| File | Holds |
|---|---|
| `transport.py` | LiveKit Agents / WebRTC session |
| `vad.py` | Silero voice-activity detection |
| `audio.py` | Frame handling, resampling, buffering |
| `bargein.py` | Caller interrupts — stop playback immediately |
| `dtmf.py` | Keypad capture, masked at or above 4 digits before storage |
| `session.py` | The turn loop: VAD → STT → graph → TTS |
| `worker.py` | Entry point. `preflight` reports what blocks answering a call |

## The budget is a contract — for this channel

Per-stage budgets live in `configs/latency_budget.yaml`. **Raising a budget is an ADR,
not a commit.** Since ADR-0018 the budget constrains *this package* rather than the
platform: it no longer decides which models the rest of the system may use.

- **Stream everywhere.** STT partials; LLM tokens piped directly into TTS. Never buffer a
  full response before speaking.
- **Every external call carries an explicit timeout.** No unbounded awaits.
- **Blocking I/O or CPU work in an async path is a defect.** Offload to a thread.
- Redaction on this path is regex-only (`RedactionMode.CALL_PATH`) — local NER does not
  fit in the slice ([ADR-0007](../../../docs/adr/0007-two-mode-redaction.md)).

## DTMF is treated as content

Four digits is already a PIN. Runs at or above the threshold are masked before storage,
and a keypad answer to a PII slot is replaced wholesale — the caller was asked for a
reference and gave one, whatever shape it has.

Never log a digit run, a raw utterance, or a pre-hash caller reference.

```bash
uv run python -m ccas.voice.worker preflight
uv run python -m ccas.voice.worker dev
make latency
```
