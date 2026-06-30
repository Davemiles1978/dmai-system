"""Shared JSON serialization helpers for DMAI Flask routes.

SQLite hands back ``bytes`` (e.g. ``MAX()`` over a BLOB-affinity column, or raw
pragma rows), which Flask's ``jsonify`` rejects with
``TypeError: Object of type bytes is not JSON serializable``. ``_jsonable``
recursively coerces a value into something ``json.dumps``/``jsonify`` accepts,
and ``safe_jsonify`` wraps that coercion so any route can return a payload
without re-discovering this bug (see PR #152, which first fixed it for
/api/admin/db-health).

``_jsonable`` is pure-stdlib and has no Flask dependency, so it stays importable
from CLI contexts. Flask is imported lazily inside ``safe_jsonify``.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path


def _jsonable(obj):
    """Recursively coerce a value into something ``json.dumps`` can serialize.

    SQLite can hand back ``bytes`` (e.g. ``MAX()`` over a BLOB-affinity column,
    or raw pragma rows), which Flask's ``jsonify`` rejects with
    ``TypeError: Object of type bytes is not JSON serializable``. Decode bytes,
    flatten ``sqlite3.Row``, and stringify other non-primitive types so the
    payload is always serializable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, bytearray):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, sqlite3.Row):
        return {k: _jsonable(obj[k]) for k in obj.keys()}
    if isinstance(obj, dict):
        return {(_jsonable(k) if not isinstance(k, str) else k): _jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def safe_jsonify(obj, status=200):
    """``jsonify`` a payload after coercing it with ``_jsonable``.

    Drop-in replacement for ``flask.jsonify`` on routes whose payloads may
    contain ``bytes`` or other non-JSON-native values from SQLite. Returns a
    Flask ``Response`` with ``status`` applied.
    """
    from flask import jsonify

    resp = jsonify(_jsonable(obj))
    resp.status_code = status
    return resp
