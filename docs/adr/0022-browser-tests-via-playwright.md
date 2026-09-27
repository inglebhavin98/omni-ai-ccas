# ADR-0022: Browser tests for the workbench via Playwright

Date: 2026-09-27 · Status: accepted

## Context

`src/ccas/api/static/index.html` is hand-written JS with no browser-level test. The
contract tests (`tests/unit/api/test_console_contract.py`) pin the routes it calls and
the element ids it drives, which catches silent structural drift — but not rendering,
event wiring, or whether a turn ever appears on screen. A dead button looks like a slow
one. This was recorded as 9.21 and left open for exactly this reason: the reviewer who
matters clicks the UI before reading any ADR.

Closing it needs a real browser engine, and Rule 5 locks the tech stack. Adding a
browser toolchain to the *runtime* stack would be a much bigger decision than this one
needs to be: the console is a development surface (loopback only), so a failure to
render is a developer-facing defect, not a production one.

## Decision

Playwright is admitted as a **dev-only test harness**, outside the Python locked stack:

- It lives in Node (`package.json` at the repo root, `@playwright/test` as the only
  devDependency), never in `pyproject.toml` dependencies or extras. Nothing ships.
- The browser tests run **only** via an explicit `make browser` target; `make check`
  and CI remain Python-only, so a missing Node toolchain cannot break the gates.
- The test drives the real server (`uvicorn`, real routes, real redaction, mock tool
  backend) with the LLM deliberately unconfigured, and asserts the two things the
  contract tests cannot see: a turn appears in the transcript, and a terminal session
  disables the input — rendering and event wiring, not model behaviour.
- Version pinning and browser-download footprint are dev concerns; the harness is
  excluded from packaging and from the coverage floor.

## Consequences

- 9.21 closes: the console's failure modes are now asserted in a browser, not inferred
  from strings in a file.
- A Node runtime is required to run `make browser`. The Makefile target states that
  plainly and fails with guidance when Node or the browsers are missing, rather than
  failing obscurely.
- A future browser-based surface (e.g. the agent desktop UI) inherits this harness for
  free; expanding Playwright's remit beyond dev-only tests would need a new ADR.

## Alternatives considered

- **Contract tests only** — the status quo; cannot see rendering or wiring.
- **Headless checks in pytest (html parsing)** — still not a browser; wiring is exactly
  what it cannot assert.
- **Cypress** — same class of solution with a heavier runtime and no advantage here.
