# `policies/` — M4: pure decisions

Every function here takes state and thresholds and returns a decision. No I/O, no pack
object, no clock. That is what makes them testable exhaustively and safe to call from a
node.

## Files

| File | Holds |
|---|---|
| `base.py` | `Policy` protocol and the decision shape |
| `confidence.py` | Does this intent clear the pack's threshold, or clarify? |
| `risk.py` | `RiskTier` → verification requirement, escalation pressure |
| `retry.py` | Bounded retry with an explicit ceiling |
| `sentiment.py` | Caller-state signal feeding escalation |
| `engine.py` | Ordered application — order is a decision, see ADR-0009 |

**A policy receives state and thresholds, never the pack itself.** Passing the pack would
let a policy reach for a field nobody declared it needed, and the seam between "what the
vertical says" and "what the platform does" would blur.

Thresholds are pack data. A policy that hard-codes a number is a defect.

```bash
uv run pytest tests/unit/policies -q
```
