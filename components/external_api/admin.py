"""PR CCC-1b (1/3): admin endpoints for provisioning external API keys.

Adds a small admin surface for creating, listing, and revoking the
API keys used by /api/external/*.  Mounted at /api/admin/external-keys.

Auth: reuses the existing _require_auth() from dmai_core_complete
(JWT Bearer OR X-Master-Password), imported lazily to avoid a circular
import at module load time.

Endpoints:

  POST /api/admin/external-keys
      body: {label, scope, rate_limit_per_min?, service?}
      Mints a fresh dmai_ext_<32-hex> key, hashes it, stores the row,
      returns {ok, key, key_hash, label, scope, rate_limit_per_min}.
      The plaintext key is returned EXACTLY ONCE - warn the caller
      that this is their only chance to copy it.

  GET /api/admin/external-keys
      Lists all keys (without the plaintext) so the operator can see
      what's provisioned + rotate stale ones.

  POST /api/admin/external-keys/<key_hash>/revoke
      Sets revoked=1 on the matching row. Idempotent.

Two data-quality guards (per the session-long rules):
  * scope must be non-empty (never insert a row with no scope)
  * label must be non-empty (never insert a row with a nameless key)
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from .auth import KEY_PREFIX, hash_key

logger = logging.getLogger(__name__)

external_admin_bp = Blueprint(
    "external_admin", __name__, url_prefix="/api/admin/external-keys",
)


def _require_admin() -> bool:
    """Lazy-import _require_auth to avoid circular import at boot."""
    try:
        from dmai_core_complete import _require_auth  # type: ignore
        return bool(_require_auth())
    except Exception as e:
        logger.warning("external_admin: _require_auth import failed: %s", e)
        # Fail closed if we can't verify.
        return False


def _get_conn():
    """Match the auth.py connection strategy exactly."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        try:
            import psycopg  # noqa: F401
            from components.pg_storage import PGStorage
            return PGStorage(database_url).conn
        except Exception as e:
            logger.warning("admin conn: pg fallback -> sqlite: %s", e)
    return sqlite3.connect(os.environ.get("DMAI_DB_PATH", "data/dmai.db"))


def _placeholder(conn) -> str:
    mod = type(conn).__module__.lower()
    return "%s" if "psycopg" in mod else "?"


def _mint_plaintext() -> str:
    """Return a fresh dmai_ext_<32-hex> key.

    secrets.token_hex(16) = 32 hex chars = 128 bits of entropy, which is
    the minimum acceptable strength for an API key that never expires.
    """
    return KEY_PREFIX + secrets.token_hex(16)


def _validate_scope(scope: str) -> Optional[str]:
    """Return error string if scope malformed, None if OK."""
    if not scope or not scope.strip():
        return "scope_required"
    tokens = scope.split()
    for t in tokens:
        # scope tokens are 'resource:action' (e.g. insight:write)
        if ":" not in t or len(t) < 3:
            return f"malformed_scope_token:{t}"
    return None


# ---------------------------------------------------------------------------
# POST /api/admin/external-keys  -> create
# ---------------------------------------------------------------------------
@external_admin_bp.route("", methods=["POST"])
def create_key():
    if not _require_admin():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    label = (payload.get("label") or "").strip()
    scope = (payload.get("scope") or "").strip()
    # NOTE: use 'in payload' explicitly - `... or 60` would treat 0 as
    # unset and silently default, but the caller passing 0 is a bug we
    # want to reject.
    if "rate_limit_per_min" in payload and payload["rate_limit_per_min"] is not None:
        try:
            rate_limit = int(payload["rate_limit_per_min"])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "bad_rate_limit"}), 400
    else:
        rate_limit = 60
    service = (payload.get("service") or "external-partner").strip()

    if not label:
        return jsonify({"ok": False, "error": "label_required"}), 400
    scope_err = _validate_scope(scope)
    if scope_err:
        return jsonify({"ok": False, "error": scope_err}), 400
    if rate_limit < 1 or rate_limit > 100_000:
        return jsonify({"ok": False, "error": "bad_rate_limit"}), 400

    plaintext = _mint_plaintext()
    key_hash = hash_key(plaintext)
    conn = _get_conn()
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO api_keys
                (key, service, source, validated, key_hash, scope,
                 rate_limit_per_min, revoked, label)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
            (plaintext, service, "external-admin", 1, key_hash,
             scope, rate_limit, 0, label),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.exception("create_key insert failed: %s", e)
        return jsonify({"ok": False, "error": "db_insert_failed",
                        "detail": str(e)[:200]}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return jsonify({
        "ok": True,
        "key": plaintext,       # returned ONCE - operator must copy now
        "key_hash": key_hash,
        "label": label,
        "scope": scope,
        "rate_limit_per_min": rate_limit,
        "service": service,
        "warning": "This is the only time the plaintext key is shown. "
                   "Store it somewhere safe.",
    }), 201


# ---------------------------------------------------------------------------
# GET /api/admin/external-keys  -> list
# ---------------------------------------------------------------------------
@external_admin_bp.route("", methods=["GET"])
def list_keys():
    if not _require_admin():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT key_hash, service, scope, rate_limit_per_min, revoked,
                      label, created_at, last_used
               FROM api_keys
               WHERE key_hash IS NOT NULL
               ORDER BY created_at DESC"""
        )
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.exception("list_keys failed: %s", e)
        return jsonify({"ok": False, "error": "db_query_failed",
                        "detail": str(e)[:400]}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass
    keys: List[Dict[str, Any]] = []
    for r in rows:
        keys.append({
            "key_hash": r[0],
            "service": r[1],
            "scope": r[2],
            "rate_limit_per_min": r[3],
            "revoked": bool(r[4]),
            "label": r[5],
            "created_at": str(r[6]) if r[6] else None,
            "last_used": str(r[7]) if r[7] else None,
        })
    return jsonify({"ok": True, "count": len(keys), "keys": keys}), 200


# ---------------------------------------------------------------------------
# POST /api/admin/external-keys/<key_hash>/revoke
# ---------------------------------------------------------------------------
@external_admin_bp.route("/<key_hash>/revoke", methods=["POST"])
def revoke_key(key_hash: str):
    if not _require_admin():
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    if not key_hash or len(key_hash) != 64:
        return jsonify({"ok": False, "error": "bad_key_hash"}), 400
    conn = _get_conn()
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE api_keys SET revoked = 1 WHERE key_hash = {ph}",
            (key_hash,),
        )
        rowcount = cur.rowcount
        conn.commit()
        cur.close()
    except Exception as e:
        logger.exception("revoke_key failed: %s", e)
        return jsonify({"ok": False, "error": "db_update_failed"}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if rowcount == 0:
        return jsonify({"ok": False, "error": "key_not_found"}), 404
    return jsonify({
        "ok": True,
        "key_hash": key_hash,
        "revoked_at": datetime.now(timezone.utc).isoformat(),
    }), 200
