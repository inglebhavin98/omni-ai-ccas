# ADR 0001 — LiveKit WebRTC only for Phase 1 voice ingress

Status: accepted · 2026-09-12

## Context

The platform must eventually terminate PSTN calls through SIP trunking, with RFC 2833
DTMF as a fallback for noisy lines and accented speech — the legacy-migration analysis
flags DTMF as a gap that sinks IVR replacements.

But SIP ingress needs a trunk provider, carrier credentials, and per-minute spend before
a single measurement can be taken. The thing Phase 5 actually has to prove is that the
**800 ms round trip is achievable at all**, and that question is about VAD, STT, routing,
tool latency and TTS — none of which care how the audio arrived.

Three options were live: WebRTC only, WebRTC plus SIP from day one, and a
transport-agnostic interface with a synthetic-audio driver first.

## Decision

Phase 5 ships **LiveKit WebRTC only**, behind an `AudioTransport` protocol.

A `mock_file` driver ships alongside it, replaying fixture WAVs deterministically, so
Modules 4 and 6 are testable in CI with no media server and no network. SIP ingress and
RFC 2833 DTMF are deferred, and land as an additional driver behind the same protocol.

## Consequences

- The latency budget can be measured end to end in Phase 5 with zero telephony cost.
- `tests/latency/test_e2e_rtt.py` runs on the mock driver, so the budget gate works in CI.
- Barge-in, VAD tuning and the interruption state machine are all exercised in Phase 5;
  none of them are SIP-specific, so that work is not repeated later.
- **We do not learn what real carrier jitter and packet loss do to the budget** until
  SIP lands. That is the accepted risk: WebRTC over a good link is the optimistic case,
  and the measured numbers should be read as a floor, not a forecast.
- DTMF is modelled in the contracts from Phase 1 (`DtmfEvent`, `SlotSpec.dtmf_capturable`)
  so the deferral is a missing driver, not a missing design.
