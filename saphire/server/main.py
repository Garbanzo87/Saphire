from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from . import db as D
from .config import settings
from .routers import core, observability, runs

log = logging.getLogger("saphire")


def create_app() -> FastAPI:
    app = FastAPI(title="Saphire", version=__version__,
                  description="Continuously learning agent stack: tracing, training signals, offline/online RL, pre-deployment evaluation.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins or ["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    def _startup():
        D.get_engine()
        db = D.session()
        try:
            D.ensure_project(db, settings.default_project)
        finally:
            db.close()

    @app.middleware("http")
    async def _timing_audit_metering(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        elapsed = time.perf_counter() - t0
        resp.headers["x-process-time-ms"] = f"{elapsed * 1000:.1f}"
        from . import metrics as MX

        route = MX.route_template(request.url.path)
        MX.inc("saphire_http_requests_total", method=request.method, route=route, status=str(resp.status_code))
        MX.observe("saphire_http_request_seconds", elapsed, route=route)
        principal = getattr(request.state, "principal", None)
        path = request.url.path
        if principal is not None and path.startswith("/v1"):
            from .audit import record_request

            record_request(principal, request, resp.status_code, t0)
        return resp

    @app.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "inline_jobs": settings.inline_jobs}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        from fastapi.responses import PlainTextResponse

        from . import metrics as MX

        return PlainTextResponse(MX.render(MX.sample_gauges()), media_type="text/plain; version=0.0.4")

    app.include_router(core.router, prefix="/v1", tags=["core"])
    app.include_router(observability.router, prefix="/v1", tags=["observability"])
    app.include_router(runs.router, prefix="/v1", tags=["runs"])
    from fastapi import Depends

    from ..distributed.env_server import router as env_router
    from .deps import require_api_key

    app.include_router(env_router, prefix="/v1", dependencies=[Depends(require_api_key)])
    from .routers import admin

    app.include_router(admin.router, prefix="/v1")
    app.include_router(admin.auth_router, prefix="/v1")
    from .routers import intelligence

    app.include_router(intelligence.router, prefix="/v1", tags=["intelligence"])
    from .routers import scim

    app.include_router(scim.router)
    return app


app = create_app()
