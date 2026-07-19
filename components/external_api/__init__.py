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
# PR CCC-1b hotfix 4: apply the CCC-1a api_keys migrations directly
# at module load time. Depending on pg_storage._init_schema() to run
# them turned out to be unreliable in prod - the pg_storage init path
# is gated on lazy hydration and one failing statement in the outer
# _SCHEMA_SQL block leaves the CCC-1a ALTER TABLEs unreached.
# Each ALTER TABLE ADD COLUMN IF NOT EXISTS is idempotent so this is
# safe on every boot.
import os as _os
import logging as _logging
_bp_logger = _logging.getLogger(__name__)
_CCC1A_MIGRATIONS = [
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_hash TEXT",
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope TEXT DEFAULT ''",
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rate_limit_per_min INTEGER DEFAULT 60",
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoked INTEGER DEFAULT 0",
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS label TEXT",
    "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)",
    """CREATE TABLE IF NOT EXISTS external_api_calls (
        id           BIGSERIAL PRIMARY KEY,
        key_hash     TEXT NOT NULL,
        service      TEXT,
        endpoint     TEXT NOT NULL,
        status_code  INTEGER,
        ts           TIMESTAMPTZ DEFAULT NOW(),
        duration_ms  INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ext_calls_key_ts ON external_api_calls(key_hash, ts DESC)",
]
if _os.environ.get("DATABASE_URL", "").strip():
    try:
        from components.pg_storage import _get_conn as _pg_conn, _return_conn as _pg_return
        _c = _pg_conn()
        try:
            for _stmt in _CCC1A_MIGRATIONS:
                try:
                    with _c.cursor() as _cur:
                        _cur.execute(_stmt)
                    _c.commit()
                except Exception as _mig_err:
                    _c.rollback()
                    _bp_logger.warning(
                        "external_api: CCC-1a migration skipped (%s): %s",
                        _stmt.split()[0:4], _mig_err,
                    )
            _bp_logger.info("external_api: CCC-1a api_keys migrations applied")
        finally:
            _pg_return(_c)
    except Exception as _e:
        _bp_logger.error("external_api: CCC-1a migration bootstrap failed: %s", _e)

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
