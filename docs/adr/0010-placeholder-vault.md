# ADR 0010 — The session placeholder vault

Status: accepted · 2026-09-13

## Context

The first end-to-end run of the mesh failed in a way the unit tests could not have found.

A caller said "where is my delivery", the graph asked for the reference, the caller said
"ORD-884210" — and the tool call failed. Redaction had replaced the reference with
`[ACCOUNT_REF_1]` on the way in, so the slot held a placeholder, the tool's schema
(`^[A-Z0-9-]{6,20}$`) rejected it, and the graph escalated. The platform had correctly
protected the value from the model and, in doing so, made itself unable to do its job.

The mistake was in the framing. Rule 2 was written as "redact before egress", and every
consumer was treated as egress. But a registered tool is not an external boundary: it is
an internal system of record that already holds the caller's data, and the whole purpose
of the call is to reach it. A model must never see the reference; the backend must.

Three options: relax the tool schemas to accept placeholders (pushes the problem into
every integration and makes the arguments meaningless); carry the original alongside the
redacted value on `SlotValue` (puts it on a Pydantic model, so it lands in checkpoints,
`HandoffContext`, and logs — precisely what Rule 2 forbids); or hold the mapping outside
the serializable state.

## Decision

`PlaceholderVault` holds `token -> original` for the life of one session, in process
memory only. The tool executor restores placeholders in string arguments immediately
before dispatch to a registered backend, and nowhere else.

What keeps it honest is what it is *not*:

- **Not a field on any Pydantic model.** It cannot reach a `CallLog`, a
  `HandoffContext`, a LangGraph checkpoint, or a log line, because nothing serializes it.
- **`__repr__` and `__str__` are blind.** A traceback or a debugger watch prints
  `PlaceholderVault(entries=3)`.
- **Read in exactly one place.** `ToolExecutor._detokenize`, after authorisation, before
  dispatch. The recorded `ToolPayload` keeps the masked form, so the `ToolRecord` that
  ends up in a handoff never carries the original.
- **Unknown tokens are refused, not silently passed.** A token this vault cannot resolve
  yields `INVALID_ARGS`/`unresolved_placeholder` rather than reaching a backend masked.

Detokenisation runs *before* schema validation, not after: validating the masked form
would reject every reference a caller ever gives.

## Consequences

- The contained journey works: the model sees `[ACCOUNT_REF_1]`, the backend receives
  `ORD-884210`, the transcript and the CTI payload keep the placeholder.
- Rule 2 is now stated more precisely: **no unredacted text reaches a model, a log, a
  stored record, or an external vendor.** A registered tool backend is none of those.
  The rule did not weaken; its scope was wrong before.
- The vault is per-session and in-memory, so a horizontally scaled deployment must keep a
  session on one worker, or the tool loses the value. That is already true of the voice
  worker, which owns the audio stream. It becomes false the moment sessions migrate
  between workers — at which point the vault needs a session-affine store, not a
  serialization path. Recorded in `docs/future-scoped-work.md`.
- A second gap surfaced from the same run and is now closed at load time: nothing checked
  that an intent's declared **slots cover the required arguments of the tools it names**.
  `load_taxonomy` rejects a taxonomy where they do not, and the miner is told each tool's
  argument names so the slots it mines line up. Previously this failed mid-call as a
  schema violation that looked like a backend problem.
