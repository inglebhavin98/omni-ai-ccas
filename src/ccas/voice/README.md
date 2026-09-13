# `voice/` — M5: the call path

WebRTC in, speech out, inside 800 ms p95. Everything here is on the latency budget, so
every rule in Rule 3 applies literally.

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

## The budget is a contract

Per-stage budgets live in `configs/latency_budget.yaml` and `tests/latency/` fails the
build on breach. **Raising a budget is an ADR, not a commit.**

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
