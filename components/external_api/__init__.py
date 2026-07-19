"""PR CCC-1a: DMAI external-integration API.

Public HTTP surface for other systems (Aevora satellites, greyhound
tipster clients, trader bots, external partners) to read and write
against DMAI without borrowing the internal admin credentials.

Blueprint mount path: /api/external/*

Auth: per-integration API keys stored hashed in api_keys.key_hash,
with per-key scope (space-separated tokens) and per-minute rate limit.
See auth.py for the _require_external_key(scope) decorator.

v1 endpoints (this PR ships /api/external/status only; subsequent PRs
add /api/external/insight, /api/external/signal, /api/external/webhook/<source>).
"""
# PR CCC-1b hotfix 3: force PGStorage init on module load so the
# CCC-1a api_keys migrations (key_hash/scope/rate_limit_per_min/
# revoked/label) actually run against prod. Without this, prod's
# api_keys table stays on its pre-CCC-1a shape and every external
# API DB query 500s with 'column key_hash does not exist'.
import os as _os
import logging as _logging
_bp_logger = _logging.getLogger(__name__)
if _os.environ.get("DATABASE_URL", "").strip():
    try:
        from components.pg_storage import get_pg_storage as _get_pg
        _pg = _get_pg()
        if getattr(_pg, "is_available", lambda: False)():
            _bp_logger.info("external_api: PGStorage init OK, migrations applied")
        else:
            _bp_logger.error("external_api: PGStorage NOT available on boot")
    except Exception as _e:
        _bp_logger.error("external_api: PGStorage boot init failed: %s", _e)

from .routes import external_api_bp  # noqa: F401
from .admin import external_admin_bp  # noqa: F401
from .insight import external_insight_bp  # noqa: F401
from .insight_search import external_insight_search_bp  # noqa: F401

__all__ = [
    "external_api_bp",
    "external_admin_bp",
    "external_insight_bp",
    "external_insight_search_bp",
]
