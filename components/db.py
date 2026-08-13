"""
DMAI Database Layer — PostgreSQL (production) with SQLite fallback for local dev.

All database access routes through safe_open_kdb().  On Render (DATABASE_URL set)
this returns a PostgreSQL connection.  Locally it falls back to SQLite so the
system can still run without a Postgres instance.

Callers use the same interface regardless of backend:
    conn = safe_open_kdb("data/dmai_knowledge.db")
    conn.execute("SELECT ...", params)
    conn.commit()
    conn.close()          # returns connection to pool
    conn.row_factory = ... # dict-like rows via _PGPseudoRow
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PostgreSQL connection pool
# ---------------------------------------------------------------------------
_pg_pool: List[Any] = []
_pg_pool_lock = threading.Lock()
_PG_POOL_MAX = 8
_pg_available: Optional[bool] = None  # None=unchecked, True/False=checked


def _get_dsn() -> Optional[str]:
    """Return PostgreSQL DSN if configured, else None."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return None
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _pg_connect():
    """Create a new PostgreSQL connection. Returns None if unavailable."""
    import psycopg2
    import psycopg2.extras
    dsn = _get_dsn()
    if not dsn:
        return None
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def _pg_get_conn():
    """Get a connection from the pool or create new. Returns None if PG unavailable."""
    global _pg_available, _pg_pool
    if _pg_available is False:
        return None
    if _pg_available is None:
        # First call — check if PostgreSQL is available
        dsn = _get_dsn()
        if not dsn:
            _pg_available = False
            logger.info("DATABASE_URL not set — using SQLite fallback")
            return None
        try:
            conn = _pg_connect()
            if conn is None:
                _pg_available = False
                logger.warning("PostgreSQL: _pg_connect returned None — using SQLite fallback")
                return None
            conn.cursor().execute("SELECT 1")
            with _pg_pool_lock:
                _pg_pool.append(conn)
            _pg_available = True
            logger.info("PostgreSQL connected — routing all queries to Postgres")
        except Exception as e:
            _pg_available = False
            logger.warning("PostgreSQL unavailable, using SQLite fallback: %s", e)
            return None

    with _pg_pool_lock:
        while _pg_pool:
            conn = _pg_pool.pop()
            try:
                conn.cursor().execute("SELECT 1")
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

    # Pool empty — create new
    try:
        return _pg_connect()
    except Exception as e:
        logger.warning("Failed to create PostgreSQL connection: %s", e)
        return None


def _pg_return_conn(conn):
    """Return a connection to the pool."""
    global _pg_pool
    try:
        conn.rollback()
    except Exception:
        pass
    with _pg_pool_lock:
        if len(_pg_pool) < _PG_POOL_MAX:
            _pg_pool.append(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# PostgreSQL connection wrapper — mimics sqlite3.Connection interface
# ---------------------------------------------------------------------------
class PGConnection:
    """Wraps a psycopg2 connection to look like sqlite3.Connection.

    Supports: execute(), commit(), rollback(), close(), row_factory,
    cursor(), __enter__/__exit__, and attribute forwarding for anything else.
    """

    __slots__ = ("_conn", "_row_factory")

    def __init__(self, pg_conn):
        object.__setattr__(self, "_conn", pg_conn)
        object.__setattr__(self, "_row_factory", None)

    # ── Core interface ───────────────────────────────────────────────
    def execute(self, sql: str, params=()) -> "PGCursor":
        """Execute SQL and return a cursor for fetchall/fetchone."""
        if params and not isinstance(params, (tuple, list, dict)):
            params = (params,)
        # Translate SQLite ? placeholders to PostgreSQL %s
        sql = sql.replace('?', '%s')
        # Handle SQLite-specific syntax that slips through
        if sql.strip().upper().startswith('PRAGMA'):
            return PGCursor(self._conn.cursor(), self._row_factory)  # no-op
        # Handle INSERT OR IGNORE → ON CONFLICT DO NOTHING
        if 'INSERT OR IGNORE' in sql.upper():
            sql = sql.replace('INSERT OR IGNORE', 'INSERT')
            if 'ON CONFLICT' not in sql.upper():
                sql = sql.rstrip(';') + ' ON CONFLICT DO NOTHING'
        # Handle INSERT OR REPLACE → ON CONFLICT DO UPDATE for PostgreSQL
        if 'INSERT OR REPLACE' in sql.upper():
            sql = sql.replace('INSERT OR REPLACE', 'INSERT')
            # Try to add ON CONFLICT clause for id primary key
            if 'ON CONFLICT' not in sql.upper():
                # Find the table name
                import re as _re
                _m = _re.search(r'INTO\s+(\w+)', sql, _re.IGNORECASE)
                if _m:
                    _table = _m.group(1)
                    # Check if we know the PK for this table
                    _pk_map = {
                        'capabilities': 'id', 'insights': 'id', 'system_state': 'key',
                        'mon_wallets': 'name', 'mon_tips': 'id', 'at_state': 'id',
                        'api_keys': 'key', 'syllabus_content': 'topic',
                    }
                    _pk = _pk_map.get(_table, 'id')
                    sql = sql.rstrip(';') + f' ON CONFLICT ({_pk}) DO UPDATE SET'
                    # Add all columns except the PK
                    _cols_match = _re.search(r'\(([^)]+)\)\s*VALUES', sql, _re.IGNORECASE)
                    if _cols_match:
                        _cols = [c.strip() for c in _cols_match.group(1).split(',') if c.strip() != _pk]
                        _updates = ', '.join(f'{c}=EXCLUDED.{c}' for c in _cols)
                        sql = sql + ' ' + _updates
        sql = sql.replace("datetime('now')", 'NOW()')
        sql = sql.replace("datetime('now')", 'NOW()')
        sql = sql.replace('datetime(\'now\')', 'NOW()')
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or None)
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise
        return PGCursor(cur, self._row_factory)

    def executemany(self, sql: str, seq_of_params) -> "PGCursor":
        sql = sql.replace('?', '%s')
        # Handle INSERT OR IGNORE → ON CONFLICT DO NOTHING
        if 'INSERT OR IGNORE' in sql.upper():
            sql = sql.replace('INSERT OR IGNORE', 'INSERT')
            if 'ON CONFLICT' not in sql.upper():
                sql = sql.rstrip(';') + ' ON CONFLICT DO NOTHING'
        # Handle INSERT OR REPLACE → ON CONFLICT DO UPDATE for PostgreSQL
        if 'INSERT OR REPLACE' in sql.upper():
            sql = sql.replace('INSERT OR REPLACE', 'INSERT')
            # Try to add ON CONFLICT clause for id primary key
            if 'ON CONFLICT' not in sql.upper():
                # Find the table name
                import re as _re
                _m = _re.search(r'INTO\s+(\w+)', sql, _re.IGNORECASE)
                if _m:
                    _table = _m.group(1)
                    # Check if we know the PK for this table
                    _pk_map = {
                        'capabilities': 'id', 'insights': 'id', 'system_state': 'key',
                        'mon_wallets': 'name', 'mon_tips': 'id', 'at_state': 'id',
                        'api_keys': 'key', 'syllabus_content': 'topic',
                    }
                    _pk = _pk_map.get(_table, 'id')
                    sql = sql.rstrip(';') + f' ON CONFLICT ({_pk}) DO UPDATE SET'
                    # Add all columns except the PK
                    _cols_match = _re.search(r'\(([^)]+)\)\s*VALUES', sql, _re.IGNORECASE)
                    if _cols_match:
                        _cols = [c.strip() for c in _cols_match.group(1).split(',') if c.strip() != _pk]
                        _updates = ', '.join(f'{c}=EXCLUDED.{c}' for c in _cols)
                        sql = sql + ' ' + _updates
        sql = sql.replace("datetime('now')", 'NOW()')
        sql = sql.replace("datetime('now')", 'NOW()')
        sql = sql.replace('datetime(\'now\')', 'NOW()')
        cur = self._conn.cursor()
        try:
            cur.executemany(sql, seq_of_params)
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise
        return PGCursor(cur, self._row_factory)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        """Return connection to pool (does not actually close)."""
        try:
            if not self._conn.closed:
                self._conn.rollback()
        except Exception:
            pass
        _pg_return_conn(self._conn)

    def cursor(self):
        return PGCursor(self._conn.cursor(), self._row_factory)

    def executescript(self, sql: str):
        """Execute multiple SQL statements. Splits on semicolons.
        Exists for compatibility with old SQLite callers."""
        for stmt in sql.split(';'):
            stmt = stmt.strip()
            if stmt and not stmt.startswith('--'):
                try:
                    cur = self._conn.cursor()
                    cur.execute(stmt)
                    cur.close()
                except Exception as e:
                    logger.debug("executescript stmt skipped: %s — %s", stmt[:60], e)
        self.commit()

    @property
    def row_factory(self):
        return self._row_factory

    @row_factory.setter
    def row_factory(self, value):
        object.__setattr__(self, "_row_factory", value)

    @property
    def in_transaction(self):
        return self._conn.status != 0  # psycopg2: 0 = idle

    # ── Context manager ──────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    # ── Attribute forwarding ─────────────────────────────────────────
    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name in ("_conn", "_row_factory"):
            object.__setattr__(self, name, value)
        elif name == "row_factory":
            object.__setattr__(self, "_row_factory", value)
        else:
            try:
                setattr(self._conn, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)


class PGCursor:
    """Wraps a psycopg2 cursor to look like sqlite3.Cursor.

    Supports: fetchall(), fetchone(), execute(), executemany(),
    description, rowcount, lastrowid, iteration, and close().
    """

    __slots__ = ("_cur", "_row_factory")

    def __init__(self, cur, row_factory=None):
        object.__setattr__(self, "_cur", cur)
        object.__setattr__(self, "_row_factory", row_factory)

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._cur.description:
            cols = [d[0] for d in self._cur.description]
            return [_PGPseudoRow(cols, row) for row in rows]
        return rows

    def fetchone(self):
        row = self._cur.fetchone()
        if row and self._cur.description:
            cols = [d[0] for d in self._cur.description]
            return _PGPseudoRow(cols, row)
        return row

    def execute(self, sql, params=()):
        """Re-execute on the same cursor. Returns self for chaining."""
        if params and not isinstance(params, (tuple, list, dict)):
            params = (params,)
        # Translate SQLite ? placeholders to PostgreSQL %s
        sql = sql.replace('?', '%s')
        self._cur.execute(sql, params or None)
        return self

    def executemany(self, sql, seq_of_params):
        self._cur.executemany(sql, seq_of_params)
        return self

    @property
    def description(self):
        return self._cur.description

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def lastrowid(self):
        try:
            if self._cur.description:
                return self._cur.fetchone()[0]
        except Exception:
            pass
        return None

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass


class _PGPseudoRow:
    """Mimics sqlite3.Row — dict-like key access + index access + iteration."""

    __slots__ = ("_cols", "_data")

    def __init__(self, cols: Tuple[str, ...], data: Tuple):
        object.__setattr__(self, "_cols", cols)
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[self._cols.index(key)]
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return list(self._cols)

    def __repr__(self):
        return str(dict(zip(self._cols, self._data)))

    def __eq__(self, other):
        if isinstance(other, _PGPseudoRow):
            return self._data == other._data
        return NotImplemented

    def __hash__(self):
        return hash(self._data)


# ---------------------------------------------------------------------------
# Public API — safe_open_kdb
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compatibility exports — callers that imported these from the old db.py
# ---------------------------------------------------------------------------
def acquire_write_lock(path):
    """No-op on PostgreSQL — concurrent writes are natively supported."""
    import contextlib
    return contextlib.nullcontext()

def is_priority_held(path=None):
    """No-op on PostgreSQL — no write lock contention."""
    return False

# For callers that do `from components.db import KeepOpenProxy`
# PGConnection is the replacement
KeepOpenProxy = PGConnection

def safe_open_kdb(
    path: str,
    *,
    timeout: float = 60.0,
    read_only: bool = False,
):
    """Open a connection to the DMAI database.

    On Render (DATABASE_URL set): returns a PGConnection wrapping psycopg2.
    Locally (no DATABASE_URL): falls back to SQLite via KeepOpenProxy.

    Args:
        path: Database path (used for SQLite fallback; ignored for PostgreSQL).
        timeout: Busy timeout in seconds (SQLite only).
        read_only: Open read-only (SQLite only).

    Returns:
        PGConnection (PostgreSQL) or KeepOpenProxy (SQLite fallback).
    """
    # Try PostgreSQL first
    pg_conn = _pg_get_conn()
    if pg_conn is not None:
        return PGConnection(pg_conn)

    # Fall back to SQLite for local development
    import sqlite3

    # Per-thread SQLite connection cache (only used in fallback mode)
    _TLS = threading.local()
    cache = getattr(_TLS, "conns", None)
    if cache is None:
        cache = _TLS.conns = {}
    key = (str(path), bool(read_only))

    cached = cache.get(key)
    if cached is not None:
        try:
            cached.execute("SELECT 1")
            return cached
        except sqlite3.Error:
            cache.pop(key, None)
            try:
                cached.close()
            except Exception:
                pass

    if read_only:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
    else:
        conn = sqlite3.connect(
            str(path),
            timeout=timeout,
            check_same_thread=False,
        )

    # Essential pragmas
    for pragma in [
        "PRAGMA journal_mode=WAL",
        "PRAGMA busy_timeout=30000",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA cache_size=-8000",
    ]:
        try:
            conn.execute(pragma)
        except Exception:
            pass

    cache[key] = conn
    return conn


# For backwards compatibility — callers that do `from components.db import KeepOpenProxy`
# won't break (they shouldn't be using it directly, but just in case)
KeepOpenProxy = None  # Removed — use PGConnection or sqlite3.connect directly
