"""PR CCC-1b (2/3): POST /api/external/insight — write endpoint.

Lets an external partner (holding a dmai_ext_<32-hex> API key with
scope 'insight:write') submit a new insight into DMAI's insights table.

Contract:
    POST /api/external/insight
    Headers: X-DMAI-Api-Key: dmai_ext_<32-hex>

    Body (JSON):
      {
        "insight_text":  str,   # required, non-empty, <= 5000 chars
        "entity_type":   str,   # required, non-empty, <= 200 chars
        "entities":      str,   # optional, <= 1000 chars
        "relationship":  str,   # optional, <= 500 chars
        "confidence":    float, # optional, 0.0-1.0, default 0.5
        "source_topic":  str,   # optional
        "target_topic":  str,   # optional
        "source_url":    str,   # optional
        "source_title":  str,   # optional
        "source_type":   str,   # optional, default 'external-api'
        "domain":        str,   # optional
        "provenance":    str    # optional; if absent, auto = 'external_api:<label>'
      }

Returns 201 {ok:true, id:<uuid>} on success.

Data-quality guards (honours the three session-long rules):
  * insight_text required non-empty (never insert empty rows)
  * entity_type required non-empty (never insert without a type)
  * confidence bounded [0, 1] - out-of-range rejected, not clamped
  * All text fields truncated to sensible column sizes but never
    silently emptied

Auth: reuses `_require_external_key('insight:write')` from
components/external_api/auth.py which handles hash lookup,
scope check, rate-limit + audit logging.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, Optional

from flask import Blueprint, g, jsonify, request

from .auth import _require_external_key

logger = logging.getLogger(__name__)

external_insight_bp = Blueprint(
    "external_insight", __name__, url_prefix="/api/external",
)


# Field size caps mirror the columns in components/pg_storage.py
FIELD_LIMITS = {
    "insight_text":  5000,
    "entity_type":   200,
    "entities":      1000,
    "relationship":  500,
    "source_topic":  500,
    "target_topic":  500,
    "source_url":    2000,
    "source_title":  500,
    "source_type":   100,
    "domain":        200,
    "provenance":    500,
}


def _get_conn():
    """Match auth.py connection strategy exactly."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        try:
            import psycopg  # noqa: F401
            from components.pg_storage import PGStorage
            return PGStorage(database_url).conn
        except Exception as e:
            logger.warning("insight conn: pg fallback -> sqlite: %s", e)
    return sqlite3.connect(os.environ.get("DMAI_DB_PATH", "data/dmai.db"))


def _placeholder(conn) -> str:
    mod = type(conn).__module__.lower()
    return "%s" if "psycopg" in mod else "?"


def _validate(payload: Dict[str, Any]) -> Optional[str]:
    """Return None if payload is OK, else an error code string."""
    if not isinstance(payload, dict):
        return "bad_payload"
    text = (payload.get("insight_text") or "").strip()
    if not text:
        return "insight_text_required"
    etype = (payload.get("entity_type") or "").strip()
    if not etype:
        return "entity_type_required"
    if "confidence" in payload and payload["confidence"] is not None:
        try:
            c = float(payload["confidence"])
        except (TypeError, ValueError):
            return "bad_confidence"
        if c < 0.0 or c > 1.0:
            return "confidence_out_of_range"
    return None


def _clip(val: Any, limit: int) -> str:
    """Truncate a value to `limit` chars; return '' for None/empty."""
    if val is None:
        return ""
    s = str(val).strip()
    return s[:limit]


@external_insight_bp.route("/insight", methods=["POST"])
@_require_external_key("insight:write")
def create_insight():
    payload = request.get_json(silent=True) or {}
    err = _validate(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    insight_id = str(uuid.uuid4())
    insight_text = _clip(payload.get("insight_text"), FIELD_LIMITS["insight_text"])
    entity_type = _clip(payload.get("entity_type"), FIELD_LIMITS["entity_type"])
    entities = _clip(payload.get("entities"), FIELD_LIMITS["entities"])
    relationship = _clip(payload.get("relationship"), FIELD_LIMITS["relationship"])
    _c = payload.get("confidence")
    confidence = 0.5 if _c is None else float(_c)
    source_topic = _clip(payload.get("source_topic"), FIELD_LIMITS["source_topic"])
    target_topic = _clip(payload.get("target_topic"), FIELD_LIMITS["target_topic"])
    source_url = _clip(payload.get("source_url"), FIELD_LIMITS["source_url"])
    source_title = _clip(payload.get("source_title"), FIELD_LIMITS["source_title"])
    source_type = _clip(payload.get("source_type"), FIELD_LIMITS["source_type"]) or "external-api"
    domain = _clip(payload.get("domain"), FIELD_LIMITS["domain"])
    # Provenance defaults to 'external_api:<label>' so we can attribute
    # every ingested insight back to which partner key produced it.
    label = (getattr(g, "dmai_key", None) or {}).get("label") or "unknown"
    provenance = _clip(payload.get("provenance"),
                       FIELD_LIMITS["provenance"]) or f"external_api:{label}"

    conn = _get_conn()
    ph = _placeholder(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO insights
                (id, insight_text, entity_type, entities, relationship,
                 confidence, source_topic, target_topic, source_url,
                 source_title, source_type, domain, provenance)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph},
                        {ph}, {ph}, {ph}, {ph}, {ph})""",
            (insight_id, insight_text, entity_type, entities, relationship,
             confidence, source_topic, target_topic, source_url,
             source_title, source_type, domain, provenance),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.exception("create_insight insert failed: %s", e)
        return jsonify({"ok": False, "error": "db_insert_failed",
                        "detail": str(e)[:200]}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return jsonify({
        "ok": True,
        "id": insight_id,
        "confidence": confidence,
        "provenance": provenance,
    }), 201
