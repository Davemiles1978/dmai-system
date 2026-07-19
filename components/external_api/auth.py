"""PR CCC-1a: API-key auth + per-key rate limiting for /api/external/*.

Auth model
----------

Callers send:
    X-DMAI-Api-Key: dmai_ext_<32-hex-chars>

Plaintext keys are never stored. We SHA-256 the presented key and
look it up in api_keys.key_hash (see pg_storage._SCHEMA_SQL). A key is
usable iff:

  * a row with that key_hash exists,
  * validated = 1,
  * revoked = 0.

Scopes
------

Each key has a `scope` column holding a space-separated set of tokens
like:

    "insight:write signal:read webhook:trader"

The decorator _require_external_key(scope) rejects with 403 if the
required token is not present.

Rate limits
-----------

Every accepted request is written to external_api_calls (audit + rate
window). Before the write, we count the caller's calls in the last
60 seconds and reject with 429 if it exceeds api_keys.rate_limit_per_min.

Never raise
-----------

The decorator swallows unexpected exceptions into a 500 with a short
error message. It must never leak internal traces to callers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Header name is intentionally distinct from the internal X-Cron-Secret
# so a stray internal request can't authenticate against external.
API_KEY_HEADER = "X-DMAI-Api-Key"
KEY_PREFIX = "dmai_ext_"


# ---------------------------------------------------------------------------
# Key hashing
# ---------------------------------------------------------------------------
def hash_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest of an API key.

    Constant-time comparable via hmac.compare_digest at the callsite.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def looks_like_dmai_key(plaintext: str) -> bool:
    """Cheap client-side shape check before the DB lookup."""
    return (
        isinstance(plaintext, str)
        and plaintext.startswith(KEY_PREFIX)
        and len(plaintext) >= len(KEY_PREFIX) + 32
    )


# ---------------------------------------------------------------------------
# DB access helpers - support both sqlite (tests, local) and psycopg (prod)
# ---------------------------------------------------------------------------
def _get_conn():
    """Return a live DB connection matching the app's canonical store.

    Order of preference:
        1. If DATABASE_URL is set and psycopg is importable, use pg.
        2. Otherwise sqlite at DMAI_DB_PATH (default 'data/dmai.db').
    """
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        try:
            from components.pg_storage import _get_conn as _pg_conn
            return _pg_conn()
        except Exception as e:
            logger.error("external_api auth: pg failed, sqlite fallback: %s", e)
    db_path = os.environ.get("DMAI_DB_PATH", "data/dmai.db")
    return sqlite3.connect(db_path)


def _placeholder(conn) -> str:
    """Return '?' for sqlite, '%s' for psycopg - keeps SQL portable."""
    mod = type(conn).__module__.lower()
    return "%s" if "psycopg" in mod else "?"


# ---------------------------------------------------------------------------
# Key lookup
# ---------------------------------------------------------------------------
def lookup_key(key_hash: str) -> Optional[dict]:
    """Return the api_keys row matching key_hash, or None.

    Returns a dict shape independent of the underlying driver so
    handlers don't have to care about sqlite vs psycopg tuples.
    """
    try:
        conn = _get_conn()
    except Exception as e:
        logger.error("external_api lookup_key: cannot open DB: %s", e)
        return None
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT key, service, source, validated, scope,
                       rate_limit_per_min, revoked, label, last_used
                FROM api_keys WHERE key_hash = {ph}""",
            (key_hash,),
        )
        row = cur.fetchone()
        cur.close()
    except Exception as e:
        # Most likely: pre-CCC-1a schema. Fail closed with a warning.
        logger.warning("external_api lookup_key query failed: %s", e)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if row is None:
        return None
    return {
        "key": row[0],
        "service": row[1],
        "source": row[2],
        "validated": row[3] or 0,
        "scope": (row[4] or "").strip(),
        "rate_limit_per_min": row[5] or 60,
        "revoked": row[6] or 0,
        "label": row[7],
        "last_used": row[8],
    }


def scope_grants(key_scope: str, required: str) -> bool:
    """True iff `required` is present in the space-separated key_scope."""
    if not required:
        return True
    tokens = set((key_scope or "").split())
    return required in tokens


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def count_calls_last_minute(key_hash: str) -> int:
    """Return the caller's calls in the last 60 seconds. Fails open (0)."""
    try:
        conn = _get_conn()
    except Exception:
        return 0
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        # Portable: 60s ago as a param, works on both pg and sqlite.
        cur.execute(
            f"""SELECT COUNT(*) FROM external_api_calls
                WHERE key_hash = {ph}
                  AND ts >= (CURRENT_TIMESTAMP - INTERVAL '60 seconds')""",
            (key_hash,),
        )
        n = cur.fetchone()[0]
        cur.close()
        return int(n or 0)
    except Exception:
        # SQLite doesn't understand INTERVAL - fall back to computed epoch.
        try:
            cur = conn.cursor()
            cutoff = time.time() - 60
            cur.execute(
                f"""SELECT COUNT(*) FROM external_api_calls
                    WHERE key_hash = {ph}
                      AND strftime('%s', ts) >= {ph}""",
                (key_hash, str(int(cutoff))),
            )
            n = cur.fetchone()[0]
            cur.close()
            return int(n or 0)
        except Exception as e:
            logger.warning("count_calls_last_minute failed: %s", e)
            return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def record_call(
    key_hash: str, service: Optional[str], endpoint: str,
    status_code: int, duration_ms: int,
) -> None:
    """Append an external_api_calls row. Never raises."""
    try:
        conn = _get_conn()
    except Exception:
        return
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO external_api_calls
                (key_hash, service, endpoint, status_code, duration_ms)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph})""",
            (key_hash, service, endpoint, status_code, duration_ms),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning("record_call failed: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def bump_last_used(key_hash: str) -> None:
    """Update api_keys.last_used to now. Never raises."""
    try:
        conn = _get_conn()
    except Exception:
        return
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = {ph}",
            (key_hash,),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning("bump_last_used failed: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------
def _require_external_key(scope: str = "") -> Callable:
    """Flask decorator: require a valid API key with `scope`.

    Usage:
        @external_api_bp.route("/insight", methods=["POST"])
        @_require_external_key("insight:write")
        def post_insight():
            ...

    On success, sets flask.g.dmai_key = {service, scope, rate_limit_per_min,
    key_hash, label} and records the call to external_api_calls.

    Failure modes (all return JSON, never raise):
      401 missing_key         no header
      401 malformed_key       shape check failed
      401 unknown_key         no row for key_hash
      401 unvalidated_key     validated = 0
      403 revoked_key         revoked = 1
      403 insufficient_scope  scope token not in key_scope
      429 rate_limited        > rate_limit_per_min in last 60s
      500 server_error        any unexpected exception
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            # Imported lazily so this module remains testable without Flask.
            from flask import g, jsonify, request

            t0 = time.time()
            endpoint = getattr(request, "path", "unknown")

            def _finish(status: int, body: dict, key_hash: Optional[str] = None,
                        service: Optional[str] = None):
                dur_ms = int((time.time() - t0) * 1000)
                if key_hash:
                    record_call(key_hash, service, endpoint, status, dur_ms)
                resp = jsonify(body)
                resp.status_code = status
                return resp

            try:
                presented = request.headers.get(API_KEY_HEADER, "").strip()
                if not presented:
                    return _finish(401, {"ok": False, "error": "missing_key"})
                if not looks_like_dmai_key(presented):
                    return _finish(401, {"ok": False, "error": "malformed_key"})
                key_hash = hash_key(presented)
                row = lookup_key(key_hash)
                if row is None:
                    return _finish(401, {"ok": False, "error": "unknown_key"},
                                   key_hash=key_hash)
                if int(row["validated"]) != 1:
                    return _finish(401, {"ok": False, "error": "unvalidated_key"},
                                   key_hash=key_hash, service=row.get("service"))
                if int(row["revoked"]) == 1:
                    return _finish(403, {"ok": False, "error": "revoked_key"},
                                   key_hash=key_hash, service=row.get("service"))
                if not scope_grants(row["scope"], scope):
                    return _finish(
                        403,
                        {"ok": False, "error": "insufficient_scope",
                         "required": scope, "granted": row["scope"]},
                        key_hash=key_hash, service=row.get("service"),
                    )
                # Rate limit
                used = count_calls_last_minute(key_hash)
                if used >= int(row["rate_limit_per_min"]):
                    return _finish(
                        429,
                        {"ok": False, "error": "rate_limited",
                         "limit_per_min": row["rate_limit_per_min"],
                         "used_last_min": used},
                        key_hash=key_hash, service=row.get("service"),
                    )
                # Stash on flask.g for handler use
                g.dmai_key = {
                    "key_hash": key_hash,
                    "service": row["service"],
                    "scope": row["scope"],
                    "rate_limit_per_min": row["rate_limit_per_min"],
                    "label": row["label"],
                    "last_used": row["last_used"],
                }
                bump_last_used(key_hash)
                # Call the wrapped handler
                result = fn(*args, **kwargs)
                # Record the call at whatever status the handler produced
                try:
                    status = result[1] if isinstance(result, tuple) else getattr(result, "status_code", 200)
                except Exception:
                    status = 200
                dur_ms = int((time.time() - t0) * 1000)
                record_call(key_hash, row.get("service"), endpoint, int(status), dur_ms)
                return result
            except Exception as e:  # noqa: BLE001
                logger.exception("_require_external_key unexpected: %s", e)
                return _finish(500, {"ok": False, "error": "server_error"})
        return wrapper
    return decorator
