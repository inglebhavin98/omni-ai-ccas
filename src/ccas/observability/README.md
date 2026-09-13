# `observability/` — logs that cannot leak

Structured JSON logs, OTel spans, and a correlation id on every event. The interesting
part is what this package **refuses** to do.

## Files

| File | Holds |
|---|---|
| `logging.py` | `configure_logging`, `get_logger`, and the `_forbid_raw_content` processor |
| `tracing.py` | `new_correlation_id`, span context helpers |
| `otel.py` | OpenTelemetry wiring |

## The forbidden-field guard

Logs are the easiest place to breach Rule 2 by accident — a debug line with the caller's
utterance in it looks harmless in review and is a leak in production. So
`_forbid_raw_content` raises `ForbiddenLogFieldError` if an event carries any of:

```
utterance  transcript  raw_text  raw  text  audio  digits
dtmf  caller_id  ani  prompt  answer  slot_value
```

It also unwraps `RedactedText` **only** when the report says egress is permitted;
otherwise the value renders as `<redaction-failed>`.

Log entity *types* and *counts*, never content. `{"entity_counts": {"person": 12}}` is
fine; the person's name is not.

Everything lands in `logs/execution.log` as JSON lines.

```bash
uv run pytest tests/unit/observability -q
```
