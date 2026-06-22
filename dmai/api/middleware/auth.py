"""Authentication middleware and dependencies for the DMAI API.

Two credential tiers exist:

* **API key** (``X-API-Key``) — required for most ``/api/v1`` routes.
* **Master key** (``X-Master-Key``) — required for operator-only routes
  (``/operator/*``, registry install/reload).

A small set of public routes (health/status) need no credentials.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from dmai.config import settings

PUBLIC_PATHS = {"/health", "/api/v1/health", "/api/v1/status", "/docs", "/openapi.json", "/redoc"}

OPERATOR_PREFIXES = ("/api/v1/operator", "/operator")
OPERATOR_PATH_FRAGMENTS = ("/registry/install", "/reload")


def _is_public(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    if path == "/" or path.startswith("/static") or path.startswith("/legacy"):
        return True
    return False


def _needs_operator(path: str) -> bool:
    if any(path.startswith(p) for p in OPERATOR_PREFIXES):
        return True
    return any(frag in path for frag in OPERATOR_PATH_FRAGMENTS)


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforces API-key and master-key access control."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if request.method == "OPTIONS" or _is_public(path):
            return await call_next(request)

        if _needs_operator(path):
            if request.headers.get("X-Master-Key") != settings.master_key:
                return Response(content='{"detail":"operator key required"}', status_code=403,
                                media_type="application/json")
            return await call_next(request)

        # All other API routes require a valid API key.
        if path.startswith("/api/") or path.startswith("/registry"):
            if request.headers.get("X-API-Key") != settings.api_secret_key:
                return Response(content='{"detail":"invalid api key"}', status_code=401,
                                media_type="application/json")
        return await call_next(request)


async def require_operator(x_master_key: str = Header(default="")) -> None:
    """FastAPI dependency enforcing the operator master key."""
    if x_master_key != settings.master_key:
        raise HTTPException(status_code=403, detail="operator key required")
