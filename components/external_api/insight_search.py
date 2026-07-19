"""PR CCC-1b (3/3): GET /api/external/insight/search — read endpoint.

Lets an external partner (holding a dmai_ext_<32-hex> API key with
scope 'insight:read') query DMAI's insights table.

Contract:
    GET /api/external/insight/search
    Headers: X-DMAI-Api-Key: dmai_ext_<32-hex>

    Query params (all optional, all combine with AND):
      q             substring match on insight_text (case-insensitive)
      entity_type   exact match
      source_topic  exact match
      domain        exact match
      provenance    exact match (or exact prefix if trailing '*')
      since         ISO date/datetime; created_at >= this
      limit         1..500, default 50
      offset        default 0

Response: 200 {ok, count, insights: [...]}
  count  = number of matching rows returned in this response (>= 0)
  Each insight includes: id, insight_text, entity_type, entities,
  relationship, confidence, source_topic, target_topic, source_url,
  source_title, source_type, domain, provenance, created_at.

Errors:
  400 bad_since / bad_limit / bad_offset
  401/403 handled upstream by _require_external_key
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

from .auth import _require_external_key

logger = logging.getLogger(__name__)

external_insight_search_bp = Blueprint(
    "external_insight_search", __name__, url_prefix="/api/external",
)


DEFAULT_LIMIT = 50
MAX_LIMIT = 500


# Columns returned to the caller (never leak internal-only fields).
SELECT_COLS = (
    "id, insight_text, entity_type, entities, relationship, confidence, "
    "source_topic, target_topic, source_url, source_title, source_type, "
    "domain, provenance, created_at"
)


def _get_conn():
    """Prod: pooled psycopg2 via pg_storage._get_conn.
    Local/tests: sqlite at DMAI_DB_PATH."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        try:
            from components.pg_storage import _get_conn as _pg_conn
            return _pg_conn()
        except Exception as e:
            logger.error("insight_search conn: pg failed, sqlite fallback: %s", e)
    return sqlite3.connect(os.environ.get("DMAI_DB_PATH", "data/dmai.db"))


def _placeholder(conn) -> str:
    mod = type(conn).__module__.lower()
    return "%s" if "psycopg" in mod else "?"


def _parse_since(raw: str) -> Optional[datetime]:
    """Accept ISO date (YYYY-MM-DD) or ISO datetime. Return None on unparseable."""
    if not raw:
        return None
    raw = raw.strip()
    # Try datetime first, then bare date.
    for parser in (
        lambda s: datetime.fromisoformat(s),
        lambda s: datetime.fromisoformat(s + "T00:00:00"),
    ):
        try:
            return parser(raw)
        except (ValueError, TypeError):
            continue
    return None


def _build_where(params: Dict[str, str], ph: str) -> Tuple[str, List[Any]]:
    """Return (where_sql, bindings). Empty where_sql if no filters."""
    clauses: List[str] = []
    binds: List[Any] = []
    if params.get("q"):
        clauses.append(f"LOWER(insight_text) LIKE {ph}")
        binds.append(f"%{params['q'].lower()}%")
    for col in ("entity_type", "source_topic", "domain"):
        if params.get(col):
            clauses.append(f"{col} = {ph}")
            binds.append(params[col])
    # provenance: exact match, or prefix match if user passed trailing '*'
    prov = params.get("provenance")
    if prov:
        if prov.endswith("*"):
            clauses.append(f"provenance LIKE {ph}")
            binds.append(prov[:-1] + "%")
        else:
            clauses.append(f"provenance = {ph}")
            binds.append(prov)
    if params.get("since_dt"):
        clauses.append(f"created_at >= {ph}")
        binds.append(params["since_dt"])
    if not clauses:
        return ("", [])
    return (" WHERE " + " AND ".join(clauses), binds)


@external_insight_search_bp.route("/insight/search", methods=["GET"])
@_require_external_key("insight:read")
def search_insights():
    args = request.args
    # ---- parse + validate ------------------------------------------------
    try:
        limit = int(args.get("limit", DEFAULT_LIMIT))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad_limit"}), 400
    if limit < 1 or limit > MAX_LIMIT:
        return jsonify({"ok": False, "error": "bad_limit",
                        "range": f"1..{MAX_LIMIT}"}), 400
    try:
        offset = int(args.get("offset", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad_offset"}), 400
    if offset < 0:
        return jsonify({"ok": False, "error": "bad_offset"}), 400

    since_dt = None
    if args.get("since"):
        since_dt = _parse_since(args["since"])
        if since_dt is None:
            return jsonify({"ok": False, "error": "bad_since",
                            "hint": "use YYYY-MM-DD or ISO datetime"}), 400

    # Data-quality guard: strip empty-string filters so callers passing
    # ?q=&entity_type= aren't accidentally filtering on empty values.
    filters = {
        "q":            (args.get("q") or "").strip(),
        "entity_type":  (args.get("entity_type") or "").strip(),
        "source_topic": (args.get("source_topic") or "").strip(),
        "domain":       (args.get("domain") or "").strip(),
        "provenance":   (args.get("provenance") or "").strip(),
        "since_dt":     since_dt,
    }

    conn = _get_conn()
    ph = _placeholder(conn)
    where_sql, binds = _build_where(filters, ph)

    sql = (
        f"SELECT {SELECT_COLS} FROM insights"
        f"{where_sql} "
        f"ORDER BY created_at DESC "
        f"LIMIT {ph} OFFSET {ph}"
    )
    binds.extend([limit, offset])

    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(binds))
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.exception("search_insights query failed: %s", e)
        return jsonify({"ok": False, "error": "db_query_failed",
                        "detail": str(e)[:200]}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    cols = [c.strip() for c in SELECT_COLS.split(",")]
    insights: List[Dict[str, Any]] = []
    for r in rows:
        row_dict = {}
        for idx, name in enumerate(cols):
            val = r[idx]
            # Serialise datetime-ish values as ISO strings.
            if hasattr(val, "isoformat"):
                val = val.isoformat()
            row_dict[name] = val
        insights.append(row_dict)

    return jsonify({
        "ok": True,
        "count": len(insights),
        "limit": limit,
        "offset": offset,
        "insights": insights,
    }), 200
