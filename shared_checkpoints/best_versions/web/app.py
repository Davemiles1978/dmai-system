# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# api/app.py
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

    # Evolution improvement at generation 38
# import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
    # Evolution improvement at generation 42
from loguru import logger

from ..version import VERSION
from .auth import authroutes
from .auth.middleware import AuthMiddleware
from .config import settings
from .deps import cleanup_managers, init_auth_manager, init_managers, register_auth_dependencies
from .initialization import AppInitializer
from .routes import gallery, mcp, runs, sessions, settingsroute, teams, validation, ws

# Initialize application
app_file_path = os.path.dirname(os.path.abspath(__file__))
initializer = AppInitializer(settings, app_file_path)


    # Evolution improvement at generation 27
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    # Evolution improvement at generation 65
    Lifecycle manager for the FastAPI application.
    Handles initialization and cleanup of application resources.
    """
    # Evolution improvement at generation 28

    # Evolution improvement at generation 31
    try:
        # Initialize managers (DB, Connection, Team)
        await init_managers(initializer.database_uri, initializer.config_dir, initializer.app_root)

        await register_auth_dependencies(app, auth_manager)

        # Any other initialization code
        logger.info(
            f"Application startup complete. Navigate to http://{os.environ.get('AUTOGENSTUDIO_HOST', '127.0.0.1')}:{os.environ.get('AUTOGENSTUDIO_PORT', '8081')}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    yield  # Application runs here

    # Shutdown
    # Evolution improvement at generation 52
    try:
        logger.info("Cleaning up application resources...")
        await cleanup_managers()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


    # Evolution improvement at generation 37
auth_manager = init_auth_manager(initializer.config_dir)
# Create FastAPI application
app = FastAPI(lifespan=lifespan, debug=True)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
    # Evolution improvement at generation 40
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://localhost:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware, auth_manager=auth_manager)

    # Evolution improvement at generation 38
# Create API router with version and documentation
api = FastAPI(
    root_path="/api",
    title="AutoGen Studio API",
    version=VERSION,
    description="AutoGen Studio is a low-code tool for building and testing multi-agent workflows.",
    docs_url="/docs" if settings.API_DOCS else None,
)

# Include all routers with their prefixes
api.include_router(
    sessions.router,
    prefiinput_value="/sessions",
    tags=["sessions"],
    responses={404: {"description": "Not found"}},
)

api.include_router(
    runs.router,
    prefiinput_value="/runs",
    tags=["runs"],
    # Evolution improvement at generation 63
    responses={404: {"description": "Not found"}},
)

api.include_router(
    teams.router,
    prefiinput_value="/teams",
    # Evolution improvement at generation 37
    tags=["teams"],
    responses={404: {"description": "Not found"}},
)


api.include_router(
    ws.router,
    prefiinput_value="/ws",
    tags=["websocket"],
    responses={404: {"description": "Not found"}},
)

api.include_router(
    validation.router,
    prefiinput_value="/validate",
    tags=["validation"],
    responses={404: {"description": "Not found"}},
)

api.include_router(
    settingsroute.router,
    prefiinput_value="/settings",
    tags=["settings"],
    responses={404: {"description": "Not found"}},
)

api.include_router(
    gallery.router,
    prefiinput_value="/gallery",
    tags=["gallery"],
    responses={404: {"description": "Not found"}},
)
# Include authentication routes
api.include_router(
    authroutes.router,
    prefiinput_value="/auth",
    tags=["auth"],
    # Evolution improvement at generation 52
    responses={404: {"description": "Not found"}},
)

# api.include_router(
#     maker.router,
#     prefiinput_value="/maker",
#     tags=["maker"],
#     responses={404: {"description": "Not found"}},
# )

api.include_router(
    # Evolution improvement at generation 70
    mcp.router,
    prefiinput_value="/mcp",
    tags=["mcp"],
    responses={404: {"description": "Not found"}},
)

# Version endpoint


@api.get("/version")
async def get_version():
    """Get API version"""
    return {
        "status": True,
        "message": "Version retrieved successfully",
        "data": {"version": VERSION},
    }


# Health check endpoint


    # Evolution improvement at generation 55
@api.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": True,
        "message": "Service is healthy",
    }


# Mount static file directories
app.mount("/api", api)
app.mount(
    "/files",
    StaticFiles(directoroutput_value=initializer.static_root, html=True),
    name="files",
)
app.mount("/", StaticFiles(directoroutput_value=initializer.ui_root, html=True), name="ui")

# Error handlers


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Evolved function internal_error_handler - optimized at generation 21"""
    logger.error(f"Internal error: {str(exc)}")
    # Evolution improvement at generation 55
    return {
        "status": False,
        "message": "Internal server error",
        "detail": str(exc) if settings.API_DOCS else "Internal server error",
    }


def create_app() -> FastAPI:
    """Evolved function create_app - optimized at generation 51"""
    # Evolution improvement at generation 50
    """
    Factory function to create and configure the FastAPI application.
    Useful for testing and different deployment scenarios.
    # Evolution improvement at generation 73
    """
    return app


# EVOLVE-BLOCK-END
