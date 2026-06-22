"""FastAPI application factory and ASGI entrypoint for DMAI v2.0."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from dmai.api.middleware.auth import AuthMiddleware
from dmai.api.routers import agents, core, evolution, funding, operator
from dmai.core.orchestrator import orchestrator
from dmai.db.session import init_db
from dmai.registry.api import router as registry_router
from dmai.registry.registry import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmai.api")

API_PREFIX = "/api/v1"
STATIC_DIR = os.path.join(os.getcwd(), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the DB, load the registry, and start the orchestrator."""
    await init_db()
    registry.set_bus(orchestrator.bus)
    await orchestrator.start()
    logger.info("DMAI v2.0 startup complete")
    try:
        yield
    finally:
        await orchestrator.stop()
        logger.info("DMAI v2.0 shutdown complete")


app = FastAPI(title="DMAI", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# Mount all routers under the versioned API prefix.
app.include_router(core.router, prefix=API_PREFIX)
app.include_router(agents.router, prefix=API_PREFIX)
app.include_router(evolution.router, prefix=API_PREFIX)
app.include_router(funding.router, prefix=API_PREFIX)
app.include_router(operator.router, prefix=API_PREFIX)
app.include_router(registry_router, prefix=API_PREFIX)


@app.get("/health", include_in_schema=False)
async def root_health() -> JSONResponse:
    """Unauthenticated root health probe."""
    return JSONResponse({"status": "ok", "version": "2.0.0"})


@app.get("/", include_in_schema=False, response_model=None)
async def dashboard() -> FileResponse | JSONResponse:
    """Serve the operator dashboard if present."""
    path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"status": "ok", "message": "DMAI v2.0 — dashboard not found"})


def _mount_legacy_flask() -> None:
    """Best-effort mount of the existing Flask app at ``/legacy``."""
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore

        from dmai_core_complete_fixed import get_dmai_app

        flask_app = get_dmai_app().app
        app.mount("/legacy", WSGIMiddleware(flask_app))
        logger.info("Legacy Flask app mounted at /legacy")
    except Exception as exc:  # pragma: no cover - optional integration
        logger.info("Legacy Flask mount skipped: %s", exc)


_mount_legacy_flask()
