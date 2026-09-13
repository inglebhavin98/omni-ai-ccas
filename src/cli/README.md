# `src/cli` — the manual review console

An operator affordance for watching each stage of the pipeline under human review.
**The platform must never import this package.** The dependency runs one way: `cli`
depends on `ccas`, never the reverse.

## Files

| File | Holds |
|---|---|
| `demo.py` | Entry point and subcommands |
| `stages.py` | One entry per module — either a real call or a declared pending stage |

## Commands

```bash
uv run python -m cli.demo pipeline "where is my delivery"   # full stage walk
uv run python -m cli.demo pipeline                          # interactive
uv run python -m cli.demo pack healthcare                   # inspect a loaded pack
uv run python -m cli.demo datasets                          # role matrix + live refusal
uv run python -m cli.demo budget                            # budget + model bindings
uv run python -m cli.demo gate                              # watch the Rule 2 gate refuse
```

## A pending stage never fabricates output

Stages render as `[LIVE]` or `[PHASE n]`. A pending stage names the phase that delivers
it and what it will produce — it does not invent a plausible result. A demo that showed a
redaction it had not performed would be worse than no demo at all, because the whole
point of this console is human review.

`tests/unit/cli/test_demo.py` asserts every pending stage names a phase.

As each module lands, flip its stage from `_pending(...)` to a real call.
