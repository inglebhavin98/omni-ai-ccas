# `api/` — the workbench

An interactive console for watching a session's state while developing. **Loopback only,
never a production endpoint.**

## Files

| File | Holds |
|---|---|
| `main.py` | `create_app()` — FastAPI factory |
| `sessions.py` | Session lifecycle for the console |
| `views.py` | Rendered console |
| `schemas.py` | Request/response shapes for the API surface |

```bash
make workbench      # http://127.0.0.1:8000
```

It binds to `127.0.0.1` deliberately. The workbench exposes session state — including
redaction reports and tool traces — which is exactly what you want on a developer machine
and exactly what must not be reachable from anywhere else.

Rule 2 still applies here: what the console renders has been through the redactor like
everything else. The vault is not exposed.

```bash
uv run pytest tests/unit/api -q
```
