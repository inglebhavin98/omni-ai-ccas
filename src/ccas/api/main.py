"""Workbench API.

A development surface for driving the mesh by hand: type an utterance, watch redaction,
routing, policy verdicts, tool dispatch and the latency ledger. It exposes session state,
so it binds to loopback by default and is not a production endpoint.

Everything it returns is already redacted -- it re-reads the objects the graph produced
rather than reaching past them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ccas.api.schemas import (
    CreateSessionRequest,
    InvokeToolRequest,
    RedactionPreview,
    SessionSnapshot,
    SpeakRequest,
)
from ccas.api.sessions import SessionManager, SessionNotFoundError
from ccas.api.views import snapshot_of, tool_record_view
from ccas.config.domain_loader import DomainPackNotFoundError, load_domain
from ccas.observability.logging import configure_logging, get_logger
from ccas.observability.tracing import current_trace_context
from ccas.redaction.pipeline import RedactionMode, build_pipeline
from ccas.tools.executor import ToolExecutor
from ccas.tools.mock_backend import MockToolBackend
from ccas.tools.registry import ToolNotRegisteredError, build_registry

__all__ = ["build_app", "create_app"]

STATIC_DIR = Path(__file__).parent / "static"
LOG = get_logger("api")


def build_app(manager: SessionManager | None = None) -> FastAPI:
    sessions = manager or SessionManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(Path("logs/execution.log"), console=True)
        LOG.info("api.start", **{k: v for k, v in sessions.readiness().items() if k != "domains"})
        yield

    app = FastAPI(
        title="omni-ai-ccas workbench",
        version="0.1.0",
        description="Development console for the agentic mesh. Loopback only.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------- readiness

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Reports what is missing rather than a bare boolean.

        "not ready" without a reason is the least useful health check there is.
        """
        state = sessions.readiness()
        return JSONResponse(state, status_code=200 if state["domains"] else 503)

    @app.get("/v1/domains")
    async def domains() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in sessions.domains:
            loaded = load_domain(sessions.settings.domains_dir, name)
            pack = loaded.pack
            out[name] = {
                "display_name": pack.display_name,
                "version": pack.version,
                "has_taxonomy": loaded.has_taxonomy,
                "intents": len(loaded.taxonomy.leaves()) if loaded.has_taxonomy else 0,
                "tools": sorted(pack.tools_by_name),
                "queues": sorted(pack.queues_by_name),
                "compliance": [r.value for r in pack.compliance.regimes],
                "confidence": {
                    "route": pack.confidence.route,
                    "clarify_floor": pack.confidence.clarify_floor,
                },
            }
        return out

    # -------------------------------------------------------------- sessions

    @app.post("/v1/sessions")
    async def create_session(body: CreateSessionRequest) -> SessionSnapshot:
        try:
            handle = await sessions.create(body.domain, body.verification)
        except DomainPackNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return snapshot_of(handle)

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> SessionSnapshot:
        return snapshot_of(_require(sessions, session_id))

    @app.post("/v1/sessions/{session_id}/turns")
    async def speak(session_id: str, body: SpeakRequest) -> SessionSnapshot:
        _require(sessions, session_id)
        return snapshot_of(await sessions.say(session_id, body.text))

    @app.delete("/v1/sessions/{session_id}")
    async def drop_session(session_id: str) -> dict[str, str]:
        sessions.drop(session_id)
        return {"status": "dropped"}

    # ------------------------------------------------- direct probes (no LLM)

    @app.get("/v1/redaction/preview")
    async def redaction_preview(
        text: str, domain: str | None = None, mode: str = "realtime"
    ) -> RedactionPreview:
        """See exactly what a model would receive. Works with no provider configured."""
        name = domain or sessions.settings.default_domain
        loaded = load_domain(sessions.settings.domains_dir, name)
        pipeline = build_pipeline(
            sessions.settings.config_dir / "redaction_policy.yaml",
            pack=loaded.pack,
            mode=RedactionMode(mode),
        )
        result = pipeline.redact(text)
        return RedactionPreview(
            input_length=len(text),
            redacted=result.text,
            status=result.report.status.value,
            egress_permitted=result.report.egress_permitted,
            elapsed_us=result.report.elapsed_us,
            entity_counts=result.report.entity_counts,
            residual_patterns=result.report.residual_patterns,
            mode=mode,
        )

    @app.get("/v1/tools")
    async def list_tools(domain: str | None = None) -> dict[str, Any]:
        name = domain or sessions.settings.default_domain
        loaded = load_domain(sessions.settings.domains_dir, name)
        registry = build_registry(loaded.pack, sessions.settings.config_dir / "tools.yaml")
        return {
            spec.name: {
                "description": spec.description,
                "scope": "shared" if registry.is_global(spec.name) else "pack",
                "requires_verification": spec.requires_verification.value,
                "risk_tier": spec.risk_tier.value,
                "side_effecting": spec.side_effecting,
                "timeout_ms": spec.timeout_ms,
                "input_schema": spec.input_schema,
            }
            for spec in registry.specs
        }

    @app.post("/v1/tools/{tool_name}/invoke")
    async def invoke_tool(
        tool_name: str, body: InvokeToolRequest, domain: str | None = None
    ) -> dict[str, Any]:
        """Dispatch one tool through the full gate chain, with no graph and no LLM.

        The same executor the graph uses, so a refusal here is the refusal a call
        would get.
        """
        name = domain or sessions.settings.default_domain
        loaded = load_domain(sessions.settings.domains_dir, name)
        registry = build_registry(loaded.pack, sessions.settings.config_dir / "tools.yaml")
        pipeline = build_pipeline(
            sessions.settings.config_dir / "redaction_policy.yaml",
            pack=loaded.pack,
            mode=RedactionMode.REALTIME,
        )
        executor = ToolExecutor(registry, MockToolBackend.from_pack_dir(loaded.root), pipeline)
        try:
            payload = executor.build_payload(
                tool_name,
                session_id="probe",
                trace=current_trace_context(correlation_id="probe"),
                arguments=body.arguments,
                intent_id=body.intent_id,
                verification=body.verification,
            )
        except ToolNotRegisteredError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return tool_record_view(await executor.execute(payload))

    # ----------------------------------------------------------------- static

    if STATIC_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/favicon.ico")
        async def favicon() -> Response:
            """Answer the request every browser makes unprompted.

            Served inline rather than committed as a binary: a 404 here puts an error in
            the browser console on every load, and a console that always has an error in
            it is a console nobody reads.
            """
            return Response(content=_FAVICON, media_type="image/svg+xml")

    return app


#: A loopback dev console; the mark only has to be distinguishable in a tab strip.
_FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="7" fill="#111"/>'
    b'<circle cx="16" cy="16" r="7" fill="none" stroke="#4ade80" stroke-width="3"/>'
    b"</svg>"
)


def _require(sessions: SessionManager, session_id: str) -> Any:
    try:
        return sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def create_app() -> FastAPI:
    """Entry point for ``uvicorn ccas.api.main:create_app --factory``."""
    return build_app()
