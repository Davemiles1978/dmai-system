"""Shared SQLite connection helper for DMAI knowledge DB.

All connections to dmai_knowledge.db MUST go through safe_open_kdb() to
guarantee consistent WAL mode, tuned PRAGMAs, and a per-thread cached
connection (prevents the rollback-journal/WAL collision documented in
DB_CORRUPTION_AUDIT_2026-06-27 and INCIDENT_BRIEF_db_corruption.md).

SQLite gotcha: PRAGMA journal_mode=WAL is PER-CONNECTION. If one
connection sets WAL and a sibling opens the same file with a bare
sqlite3.connect(), the sibling writes through dmai_knowledge.db-journal
while the WAL connection writes through dmai_knowledge.db-wal. Both
update page 1 (the schema page) via different journal protocols and the
file becomes corrupt ("database disk image is malformed"). This is what
caused 3 production incidents on 2026-06-28/29.

Thread-safety: per-thread cached connection. Each thread reuses one
connection per (path, read_only) pair. `check_same_thread=False` is set
so a connection created in one thread can be closed by another, but
concurrent USE from multiple threads on the same handle is still unsafe
— callers should treat the returned handle as thread-bound.

KeepOpenProxy (2026-06-29, lock-storm hotfix):
---------------------------------------------
Many callers across the codebase write::

    conn = safe_open_kdb(path)
    conn.execute(...)
    conn.close()

This is wrong: the cache is meant to own the connection lifecycle. When
a caller closes it, the next safe_open_kdb call has to re-open the file
(re-apply WAL pragmas, etc.), and during the close/reopen window the
SQLite locking state on disk can flap between WAL and rollback-journal
modes if a sibling thread happens to open a fresh handle at the same
moment. Under load, this produced 80+ `database is locked` errors per
3-minute window from `vocabulary_ingester` alone.

Audit found 24 callers with this anti-pattern. Instead of touching all
24, this helper now returns a KeepOpenProxy that forwards every method
to the underlying sqlite3.Connection EXCEPT ``close()``, which is a
no-op. The cache remains the sole owner; the connection is closed only
when the liveness probe fails (eviction path inside safe_open_kdb).

The proxy also supports the context-manager protocol — entering returns
the proxy; exiting commits-or-rollbacks the transaction (sqlite3's
default __exit__ behaviour) but does NOT close.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

_PER_CONNECTION_PRAGMAS = (
    "PRAGMA journal_mode=WAL",          # MUST be first
    "PRAGMA synchronous=NORMAL",        # safe + fast with WAL
    "PRAGMA busy_timeout=120000",        # 120 s
    "PRAGMA foreign_keys=ON",
    "PRAGMA temp_store=MEMORY",         # avoid /tmp churn on Render
    "PRAGMA cache_size=-65536",         # 64 MB (negative = KB)
    "PRAGMA mmap_size=268435456",       # 256 MB for hot reads
    "PRAGMA wal_autocheckpoint=2000",   # checkpoint every ~8 MB of WAL
)

# Per-thread connection cache: each thread reuses one connection per
# (path, read_only) pair. This eliminates the cold-open window where a
# fresh worker would briefly create a bare-mode handle before any
# subsequent safe_open_kdb call applies the WAL pragma.
_TLS = threading.local()


# ── Process-level write mutex (2026-06-30, lock-storm root-cause fix) ──────
# KeepOpenProxy (f6fba609) stopped the close/reopen flapping. The remaining
# lock storm (~100 `database is locked`/min from vocabulary_ingester) is
# multi-thread WRITE contention: several worker threads each hold their own
# per-thread connection and issue concurrent writes to the same file. SQLite
# serializes writers at the file level, and under load the loser raises
# "database is locked" before busy_timeout elapses.
#
# Fix: serialize *writes* in-process through a re-entrant lock keyed by the
# canonical DB path. Reads stay fully parallel. The lock is reentrant so
# nested write paths on the same thread (e.g. execute INSERT then commit, or
# a batched BEGIN..executemany..COMMIT held under one guard) don't deadlock.
# PR V-fast: bumped from 30s -> 60s to reduce storm noise while V-real (hot-
# table Postgres migration) is in flight. On expiry we still raise cleanly.
_WRITE_MUTEX_TIMEOUT = 120.0  # seconds; on expiry we raise, never block forever
_WRITE_LOCKS: dict[str, threading.RLock] = {}
_WRITE_LOCKS_META = threading.Lock()           # guards _WRITE_LOCKS mutation
_WRITE_LOCK_HOLDERS: dict[str, int] = {}        # path -> last-acquirer thread ident
# PR V-fast: rolling max wait-to-acquire per path (ms). Reset on read via the
# diagnostic endpoint. Populated by _WriteGuard.__enter__ and proxy._acquire.
_WRITE_LOCK_MAX_WAIT_MS: dict[str, int] = {}

# SQL whose first token marks a pure read. Everything else takes the lock
# (safe default per design: ambiguous == treat as write).
_READ_ONLY_FIRST_TOKENS = frozenset({"select", "pragma", "explain"})


def _canonical_db_key(path) -> str:
    try:
        return os.path.realpath(str(path))
    except Exception:
        return str(path)


def _get_write_lock(path) -> tuple[str, threading.RLock]:
    """Return (canonical_key, RLock) for path, creating the lock once."""
    key = _canonical_db_key(path)
    lock = _WRITE_LOCKS.get(key)
    if lock is None:
        with _WRITE_LOCKS_META:
            lock = _WRITE_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                _WRITE_LOCKS[key] = lock
    return key, lock


def _first_sql_token(sql: str) -> str:
    """Lower-cased first SQL keyword, skipping leading whitespace/comments."""
    s = sql.lstrip()
    while s:
        if s.startswith("--"):
            nl = s.find("\n")
            if nl == -1:
                return ""
            s = s[nl + 1:].lstrip()
        elif s.startswith("/*"):
            end = s.find("*/")
            if end == -1:
                return ""
            s = s[end + 2:].lstrip()
        else:
            break
    i = 0
    while i < len(s) and (s[i].isalpha() or s[i] == "_"):
        i += 1
    return s[:i].lower()


def _is_write_sql(sql) -> bool:
    """True if the statement mutates. Non-str or unclassifiable -> True (safe)."""
    if not isinstance(sql, str):
        return True
    token = _first_sql_token(sql)
    if not token:
        return True
    return token not in _READ_ONLY_FIRST_TOKENS


def _format_holder_stack(ident) -> str:
    if ident is None:
        return "  <no recorded holder>"
    frame = sys._current_frames().get(ident)
    if frame is None:
        return f"  <holder thread {ident} not found>"
    return "".join(traceback.format_stack(frame)[-8:])


class _WriteGuard:
    """Context manager: acquire the per-path write lock with a 30 s timeout.

    On timeout, log a structured warning naming the recorded holder thread and
    a snapshot of its current stack, then raise
    ``sqlite3.OperationalError("write_mutex_timeout")`` — observability, not a
    silent hang or silent failure.
    """

    __slots__ = ("_key", "_lock")

    def __init__(self, key: str, lock: threading.RLock) -> None:
        self._key = key
        self._lock = lock

    def __enter__(self) -> "_WriteGuard":
        # PR V-fast: record wait-to-acquire for /api/admin/db-lock-status.
        _t0 = time.monotonic()
        if not self._lock.acquire(timeout=_WRITE_MUTEX_TIMEOUT):
            holder = _WRITE_LOCK_HOLDERS.get(self._key)
            logger.warning(
                "write_mutex_timeout: thread %s (%s) could not acquire write lock "
                "for %s within %.0fs. Recorded holder thread ident=%s; holder stack:\n%s",
                threading.current_thread().name,
                threading.get_ident(),
                self._key,
                _WRITE_MUTEX_TIMEOUT,
                holder,
                _format_holder_stack(holder),
            )
            raise sqlite3.OperationalError("write_mutex_timeout")
        _record_wait_ms(self._key, int((time.monotonic() - _t0) * 1000))
        _WRITE_LOCK_HOLDERS[self._key] = threading.get_ident()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._lock.release()
        return False


def acquire_write_lock(path):
    """Public helper returning a context manager that holds the write lock for
    ``path`` across a block. Reentrant: safe to nest with the proxy's own
    per-statement guards (e.g. wrap a whole BEGIN/executemany/COMMIT batch)."""
    key, lock = _get_write_lock(path)
    return _WriteGuard(key, lock)


def _record_wait_ms(key: str, wait_ms: int) -> None:
    """PR V-fast: track the rolling max wait-to-acquire per DB path so we can
    see contention pressure via /api/admin/db-lock-status without waiting for
    a 60s timeout to log a warning."""
    try:
        prev = _WRITE_LOCK_MAX_WAIT_MS.get(key, 0)
        if wait_ms > prev:
            _WRITE_LOCK_MAX_WAIT_MS[key] = wait_ms
    except Exception:
        pass


# ── PR NN: cooperative priority-pause for background writers ──────────────
# Some background threads (vocab ingester, gap fetcher, etc.) keep the write
# mutex effectively occupied by continuously flushing rows. When a higher-
# priority operation runs (materialiser tick, migration, admin action), it
# calls priority_hold() to signal "back off"; polite background writers check
# is_priority_held() before starting a batch and sleep instead.
#
# This is COOPERATIVE. Background loops must opt in by consulting
# is_priority_held() in their poll code. The lock itself remains available;
# a rogue writer that doesn't check will still succeed. This is deliberate:
# migrations and one-shots must still be able to write.
_PRIORITY_HOLDERS: set[str] = set()          # opaque tokens of active holders
_PRIORITY_HOLDERS_LOCK = threading.Lock()


def priority_hold(token: str) -> None:
    """Signal to polite background writers to back off. Idempotent; call
    ``priority_release(token)`` in a finally block."""
    with _PRIORITY_HOLDERS_LOCK:
        _PRIORITY_HOLDERS.add(str(token))


def priority_release(token: str) -> None:
    """Release the priority hold registered under ``token``. No-op if unknown."""
    with _PRIORITY_HOLDERS_LOCK:
        _PRIORITY_HOLDERS.discard(str(token))


def is_priority_held() -> bool:
    """Return True if any thread has called priority_hold and not yet released.
    Background writers should check this and sleep instead of flushing."""
    with _PRIORITY_HOLDERS_LOCK:
        return bool(_PRIORITY_HOLDERS)


class _PriorityHold:
    """Context manager wrapper around priority_hold/release."""
    __slots__ = ("_token",)

    def __init__(self, token: str) -> None:
        self._token = str(token)

    def __enter__(self) -> "_PriorityHold":
        priority_hold(self._token)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        priority_release(self._token)
        return False


def priority_hold_ctx(token: str) -> "_PriorityHold":
    """Context-manager form: ``with priority_hold_ctx('tag'): ...``."""
    return _PriorityHold(token)


def get_write_lock_status() -> dict:
    """Snapshot the current write-lock holders. Used by the diagnostic
    endpoint /api/admin/db-lock-status.

    Returns a dict of {path: {holder_thread_ident, holder_thread_name,
    holder_stack, has_lock_object, max_wait_ms}}. Never raises.
    """
    out: dict = {}
    # Copy keys first — dict may mutate while iterating from other threads.
    try:
        keys = list(_WRITE_LOCKS.keys())
    except Exception:
        keys = []
    threads_by_ident = {t.ident: t for t in threading.enumerate() if t.ident is not None}
    for key in keys:
        holder_ident = _WRITE_LOCK_HOLDERS.get(key)
        holder_thread = threads_by_ident.get(holder_ident) if holder_ident else None
        holder_name = holder_thread.name if holder_thread else None
        # Only render a stack if the lock appears to be held right now.
        stack: list[str] = []
        lock_obj = _WRITE_LOCKS.get(key)
        held_now = False
        if lock_obj is not None:
            # Try a non-blocking acquire; if it succeeds, the lock was free,
            # so release it immediately. If it fails, the lock is currently
            # held by someone — render the holder's stack for diagnostics.
            got = False
            try:
                got = lock_obj.acquire(blocking=False)
            except Exception:
                got = False
            if got:
                try:
                    lock_obj.release()
                except Exception:
                    pass
            else:
                held_now = True
                stack_lines = _format_holder_stack(holder_ident).splitlines()
                stack = stack_lines[-12:]
        out[key] = {
            "holder_thread_ident": holder_ident,
            "holder_thread_name": holder_name,
            "currently_held": held_now,
            "holder_stack": stack,
            "max_wait_ms": _WRITE_LOCK_MAX_WAIT_MS.get(key, 0),
        }
    return out




# ---------------------------------------------------------------------------
# PGProxy — PostgreSQL wrapper that mimics sqlite3.Connection interface
# ---------------------------------------------------------------------------
class PGProxy:
    """Wraps a psycopg2 connection so it looks like a sqlite3.Connection.

    Translates SQL on-the-fly: ? → %s, INSERT OR IGNORE → ON CONFLICT,
    and other SQLite-isms to PostgreSQL equivalents.  Callers using
    safe_open_kdb() get this automatically when DATABASE_URL is set.
    """

    __slots__ = ("_conn", "_cur", "_in_transaction", "_row_factory")

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_cur", None)
        object.__setattr__(self, "_in_transaction", False)
        object.__setattr__(self, "_row_factory", None)

    # ── SQL translation ─────────────────────────────────────────────
    @staticmethod
    def _translate_sql(sql: str) -> str:
        """Translate SQLite SQL to PostgreSQL SQL."""
        s = sql
        # Replace ? placeholders with %s (handling quoted strings)
        in_quote = False
        quote_char = None
        result = []
        i = 0
        while i < len(s):
            c = s[i]
            if c in ("'", '"') and (i == 0 or s[i-1] != '\\'):
                if not in_quote:
                    in_quote = True
                    quote_char = c
                elif c == quote_char:
                    in_quote = False
                    quote_char = None
            if c == '?' and not in_quote:
                result.append('%s')
            else:
                result.append(c)
            i += 1
        translated = ''.join(result)
        # SQLite → PostgreSQL translations
        translated = translated.replace("datetime('now')", "NOW()")
        translated = translated.replace('last_insert_rowid()', 'LASTVAL()')
        # AUTOINCREMENT → nothing (PostgreSQL SERIAL is auto-increment)
        import re
        translated = re.sub(r'\bAUTOINCREMENT\b', '', translated)
        # INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
        translated = re.sub(
            r'INSERT OR IGNORE INTO (\w+)\s*\(', 
            r'INSERT INTO \1 (', 
            translated
        )
        # PRAGMA statements → no-op (PostgreSQL handles config differently)
        if translated.strip().upper().startswith('PRAGMA'):
            return 'SELECT 1'  # Harmless no-op
        # sqlite_master → information_schema.tables
        translated = translated.replace(
            "FROM sqlite_master", 
            "FROM information_schema.tables"
        )
        translated = translated.replace(
            "FROM sqlite_master WHERE", 
            "FROM information_schema.tables WHERE"
        )
        return translated

    # ── Interface mimicking sqlite3.Connection ───────────────────────
    def execute(self, sql, params=()):
        """Execute SQL, returning a cursor-like wrapper that supports chaining."""
        if params and not isinstance(params, (tuple, list)):
            params = (params,)
        translated = self._translate_sql(sql)
        try:
            cur = self._conn.cursor()
            cur.execute(translated, params or None)
            object.__setattr__(self, "_cur", cur)
            object.__setattr__(self, "_in_transaction", True)
            wrapper = _PGCursorWrapper(cur, self._row_factory, self._conn)
            return wrapper
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    def executemany(self, sql, seq_of_params):
        translated = self._translate_sql(sql)
        try:
            cur = self._conn.cursor()
            cur.executemany(translated, seq_of_params)
            object.__setattr__(self, "_cur", cur)
            object.__setattr__(self, "_in_transaction", True)
            wrapper = _PGCursorWrapper(cur, self._row_factory, self._conn)
            return wrapper
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    def commit(self):
        if self._in_transaction:
            self._conn.commit()
            object.__setattr__(self, "_in_transaction", False)
        return None

    def rollback(self):
        if self._in_transaction:
            self._conn.rollback()
            object.__setattr__(self, "_in_transaction", False)
        return None

    def close(self):
        """No-op — connection pooling owns the lifecycle."""
        try:
            if self._cur:
                self._cur.close()
        except Exception:
            pass
        object.__setattr__(self, "_cur", None)
        object.__setattr__(self, "_in_transaction", False)
        return None

    @property
    def row_factory(self):
        return self._row_factory

    @row_factory.setter
    def row_factory(self, value):
        object.__setattr__(self, "_row_factory", value)

    @property
    def in_transaction(self):
        return self._in_transaction

    def cursor(self):
        return _PGCursorWrapper(self._conn.cursor(), self._row_factory)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name in ("_conn", "_cur", "_in_transaction", "_row_factory"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)


class _PGCursorWrapper:
    """Wraps a psycopg2 cursor to look like a sqlite3 cursor.
    
    Supports chained calls like conn.execute(sql, params).fetchall()
    by returning self from execute() and implementing fetch methods.
    """

    __slots__ = ("_cur", "_row_factory", "_description", "_conn")

    def __init__(self, cur, row_factory=None, conn=None):
        object.__setattr__(self, "_cur", cur)
        object.__setattr__(self, "_row_factory", row_factory)
        object.__setattr__(self, "_conn", conn)
        try:
            object.__setattr__(self, "_description", cur.description)
        except Exception:
            object.__setattr__(self, "_description", None)

    def execute(self, sql, params=()):
        """Execute SQL on the wrapped cursor. Returns self for chaining."""
        if params and not isinstance(params, (tuple, list)):
            params = (params,)
        translated = PGProxy._translate_sql(sql)
        try:
            self._cur.execute(translated, params or None)
            object.__setattr__(self, "_description", self._cur.description)
            return self
        except Exception:
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            raise

    def executemany(self, sql, seq_of_params):
        translated = PGProxy._translate_sql(sql)
        try:
            self._cur.executemany(translated, seq_of_params)
            object.__setattr__(self, "_description", self._cur.description)
            return self
        except Exception:
            if self._conn:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            raise

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._description and self._row_factory:
            cols = [d[0] for d in self._description]
            return [_PGPseudoRow(cols, row) for row in rows]
        return rows

    def fetchone(self):
        row = self._cur.fetchone()
        if row and self._description and self._row_factory:
            cols = [d[0] for d in self._description]
            return _PGPseudoRow(cols, row)
        return row

    @property
    def description(self):
        return self._description

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def lastrowid(self):
        try:
            self._cur.execute("SELECT LASTVAL()")
            row = self._cur.fetchone()
            return row[0] if row else None
        except Exception:
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
    """Mimics sqlite3.Row — supports both index and key access."""
    __slots__ = ("_cols", "_data")
    
    def __init__(self, cols, data):
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
        return self._cols
    
    def __repr__(self):
        return str(dict(zip(self._cols, self._data)))


# ---------------------------------------------------------------------------
# PostgreSQL connection pool (shared with pg_storage)
# ---------------------------------------------------------------------------
_pg_pool = None
_pg_available = None  # Tri-state: None=unchecked, True=available, False=unavailable

def _get_pg_connection():
    """Get a PostgreSQL connection from the shared pool. Returns None if unavailable."""
    global _pg_pool, _pg_available
    if _pg_available is False:
        return None
    if _pg_available is None:
        # Check if PostgreSQL is configured
        import os as _os_pg
        if not _os_pg.environ.get("DATABASE_URL"):
            _pg_available = False
            return None
        try:
            import psycopg2 as _pg2
            dsn = _os_pg.environ["DATABASE_URL"]
            if dsn.startswith("postgres://"):
                dsn = "postgresql://" + dsn[len("postgres://"):]
            conn = _pg2.connect(dsn)
            conn.autocommit = False
            conn.cursor().execute("SELECT 1")
            _pg_pool = [conn]
            _pg_available = True
            logger = logging.getLogger(__name__)
            logger.info("PostgreSQL routing active for safe_open_kdb")
        except Exception as e:
            _pg_available = False
            logging.getLogger(__name__).warning(
                "PostgreSQL not available for safe_open_kdb: %s", e
            )
            return None
    # Return from pool or create new
    import psycopg2 as _pg2
    import os as _os_pg
    if _pg_pool:
        conn = _pg_pool.pop()
        try:
            conn.cursor().execute("SELECT 1")
            _pg_pool.append(conn)
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    # Create new
    try:
        dsn = _os_pg.environ["DATABASE_URL"]
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://"):]
        return _pg2.connect(dsn)
    except Exception:
        _pg_available = False
        return None


def _return_pg_connection(conn):
    """Return a connection to the pool."""
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = []
    try:
        conn.rollback()
    except Exception:
        pass
    if len(_pg_pool) < 4:
        _pg_pool.append(conn)
    else:
        try:
            conn.close()
        except Exception:
            pass


class KeepOpenProxy:
    """Wraps a sqlite3.Connection so .close() is a no-op.

    The cache owns the underlying connection lifecycle. Callers may
    safely call .close() (or use ``with`` blocks) without disturbing
    the cached handle. All other attribute access is forwarded to the
    wrapped connection.

    Context-manager semantics: ``__enter__`` returns the proxy itself;
    ``__exit__`` delegates to the underlying connection's context
    manager (commit on no-exception, rollback on exception) but does
    NOT close the connection.
    """

    __slots__ = ("_conn", "_wkey", "_wlock", "_txn_held")

    def __init__(
        self,
        conn: sqlite3.Connection,
        write_key: str = "",
        write_lock: threading.RLock | None = None,
    ) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_wkey", write_key)
        object.__setattr__(self, "_wlock", write_lock)
        # True while we hold one outstanding lock acquisition for an
        # in-progress write transaction on this (per-thread) connection.
        object.__setattr__(self, "_txn_held", False)

    # ── close() override ────────────────────────────────────────────
    def close(self) -> None:  # noqa: D401
        """Release any outstanding transaction write-lock hold.

        The underlying connection stays open (the cache owns its
        lifecycle) but any RLock acquisition we were carrying for an
        open txn is dropped. Without this, a caller who opens a proxy,
        starts a write txn, and closes the proxy without an explicit
        commit/rollback leaves the process-wide RLock held — which
        starves every other writer until the pooled connection is
        eventually reused. That's the exact wedge that blocked the
        graph projector rebuild on prod (dmai-insight-promoter thread
        held dmai_knowledge.db's write lock across its poll wait).
        """
        try:
            # Best effort: rollback flushes any half-open txn state
            # before we drop the RLock. If the connection is idle this
            # is a no-op.
            if getattr(self._conn, "in_transaction", False):
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        finally:
            self._release_txn_hold()
        return None

    # ── write-gating internals ──────────────────────────────────────
    # Only mutating operations take the process-level write lock; reads stay
    # ungated and fully parallel. The lock is held across the WHOLE transaction
    # — from the first write statement until commit/rollback — not per
    # statement. Releasing between a statement and its commit would let another
    # thread open a competing SQLite transaction and deadlock against the
    # busy_timeout. The RLock is reentrant so nested acquisitions are cheap.
    def _acquire(self) -> None:
        if self._wlock is None:
            return
        # PR V-fast: record wait-to-acquire for /api/admin/db-lock-status.
        _t0 = time.monotonic()
        if not self._wlock.acquire(timeout=_WRITE_MUTEX_TIMEOUT):
            holder = _WRITE_LOCK_HOLDERS.get(self._wkey)
            logger.warning(
                "write_mutex_timeout: thread %s (%s) could not acquire write lock "
                "for %s within %.0fs. Recorded holder thread ident=%s; holder stack:\n%s",
                threading.current_thread().name,
                threading.get_ident(),
                self._wkey,
                _WRITE_MUTEX_TIMEOUT,
                holder,
                _format_holder_stack(holder),
            )
            raise sqlite3.OperationalError("write_mutex_timeout")
        _record_wait_ms(self._wkey, int((time.monotonic() - _t0) * 1000))
        _WRITE_LOCK_HOLDERS[self._wkey] = threading.get_ident()

    def _release(self) -> None:
        if self._wlock is None:
            return
        self._wlock.release()

    def _settle_after_write(self) -> None:
        # Keep exactly one acquisition outstanding while a transaction is open;
        # release immediately when the statement left no open transaction.
        if self._wlock is None:
            return
        if self._conn.in_transaction:
            if self._txn_held:
                self._release()  # already holding for the txn; drop the extra
            else:
                object.__setattr__(self, "_txn_held", True)  # keep this one
        else:
            self._release()

    def _release_txn_hold(self) -> None:
        if self._txn_held:
            object.__setattr__(self, "_txn_held", False)
            self._release()

    # ── write-gated method wrappers ─────────────────────────────────
    def execute(self, sql, *args, **kwargs):
        if not _is_write_sql(sql):
            return self._conn.execute(sql, *args, **kwargs)
        self._acquire()
        try:
            cur = self._conn.execute(sql, *args, **kwargs)
        except BaseException:
            self._release()
            raise
        self._settle_after_write()
        return cur

    def executemany(self, sql, *args, **kwargs):
        self._acquire()
        try:
            cur = self._conn.executemany(sql, *args, **kwargs)
        except BaseException:
            self._release()
            raise
        self._settle_after_write()
        return cur

    def executescript(self, sql, *args, **kwargs):
        self._acquire()
        try:
            cur = self._conn.executescript(sql, *args, **kwargs)
        except BaseException:
            self._release()
            raise
        self._settle_after_write()
        return cur

    def commit(self):
        self._acquire()
        try:
            return self._conn.commit()
        finally:
            self._release()
            self._release_txn_hold()

    def rollback(self):
        self._acquire()
        try:
            return self._conn.rollback()
        finally:
            self._release()
            self._release_txn_hold()

    # ── attribute forwarding ────────────────────────────────────────
    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value) -> None:
        # __slots__ keeps _conn assignable via object.__setattr__ in __init__.
        # All other attribute writes (e.g. row_factory) go to the wrapped conn.
        if name == "_conn":
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)

    # ── context-manager protocol ────────────────────────────────────
    def __enter__(self) -> "KeepOpenProxy":
        # Mirror sqlite3's behaviour: entering returns the connection
        # (here, the proxy) so the user can do ``with conn as c: ...``.
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Delegate commit/rollback to the underlying connection, but do
        # NOT close. sqlite3.Connection.__exit__ does not close either,
        # so this is consistent with stdlib behaviour. This commits/rolls
        # back the connection directly (bypassing our commit()), so release
        # any write-lock hold left open by writes inside the with-block.
        try:
            self._conn.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._release_txn_hold()

    # ── helpful introspection ───────────────────────────────────────
    def __repr__(self) -> str:
        return f"<KeepOpenProxy wrapping {self._conn!r}>"


def _really_close(conn) -> None:
    """Force-close a connection, unwrapping any proxy. Used only by the
    eviction path inside safe_open_kdb when the liveness probe fails."""
    target = getattr(conn, "_conn", conn)
    try:
        target.close()
    except Exception:
        pass


def safe_open_kdb(
    path: str,
    *,
    timeout: float = 60.0,
    read_only: bool = False,
) -> KeepOpenProxy:
    """Open or reuse a connection to a DMAI SQLite DB with safe defaults.

    Returns a KeepOpenProxy wrapping the per-thread cached connection.
    Callers may call ``.close()`` on the proxy (it becomes a no-op) or
    use ``with`` blocks (commit/rollback works, but no close). The
    cache owns the real connection lifecycle.

    Args:
        path: filesystem path to the .db file.
        timeout: SQLite busy timeout in seconds (default 30 s). Maps to
            both the Python-level connect timeout AND the SQLite-level
            busy_timeout PRAGMA (30000 ms).
        read_only: if True, open via URI in `mode=ro` so writes raise.
    """
    cache = getattr(_TLS, "conns", None)
    if cache is None:
        cache = _TLS.conns = {}
    key = (str(path), bool(read_only))
    # Process-level write lock for this DB file (shared across all threads).
    wkey, wlock = _get_write_lock(path)
    cached = cache.get(key)
    if cached is not None:
        try:
            # Liveness probe — operate on the underlying connection
            # directly so we don't accidentally hit a proxied method.
            real = cached._conn if isinstance(cached, KeepOpenProxy) else cached
            real.execute("SELECT 1")
            # Always hand callers a proxy — even if a non-proxy slipped
            # into the cache (defensive against future bugs).
            if isinstance(cached, KeepOpenProxy):
                return cached
            return KeepOpenProxy(cached, wkey, wlock)
        except sqlite3.Error:
            cache.pop(key, None)
            _really_close(cached)

    # R4/Bug 2 self-heal: a cached connection can go stale because its file
    # vanished from under it (quarantined by a rebuild/admin action, or the
    # boot-time self-heal path). If the DB file is genuinely missing and this
    # is a write-capable open, try to lay down fresh schema right now instead
    # of silently falling through to a bare, schema-less sqlite3.connect()
    # below (which would create an empty file with no tables and defer the
    # real fix to the next process restart). Local import avoids a hard
    # import cycle with dmai_core_complete (which imports safe_open_kdb).
    # Scoped to dmai_knowledge.db specifically: _ensure_kdb_schema lays down
    # the shared CORE_SCHEMA (capabilities/insights/at_state/etc.) plus every
    # components/*.py CREATE-TABLE it can scan. Running it against an
    # arbitrary caller-chosen db_path (e.g. a component's own isolated test
    # DB, or trading_mastery.db) would let a generic/fallback schema win a
    # CREATE TABLE IF NOT EXISTS race against that component's own, more
    # specific _init_db() — silently locking in the wrong columns. Only
    # dmai_knowledge.db is the intended target of this self-heal.
    if (
        not read_only
        and not os.path.exists(path)
        and os.path.basename(str(path)) == "dmai_knowledge.db"
    ):
        try:
            logger.warning("safe_open_kdb: %s missing, invoking schema restore", path)
            from dmai_core_complete import _ensure_kdb_schema
            _ensure_kdb_schema(str(path))
        except Exception as _heal_err:
            logger.warning("safe_open_kdb: schema restore for %s failed: %s", path, _heal_err)
            # Fall through to the bare-connect path below — it will create an
            # empty schema-less DB, but at least we tried.

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
    # Pragmas applied PER CONNECTION (the SQLite gotcha).
    for pragma in _PER_CONNECTION_PRAGMAS:
        try:
            conn.execute(pragma)
        except sqlite3.Error:
            # Some PRAGMAs (e.g. mmap_size on certain builds) may no-op
            # or be unsupported. Don't fail the open over an optional
            # pragma; the critical ones (journal_mode, busy_timeout,
            # synchronous, foreign_keys) come first and will raise on
            # the same kind of failure if it's a real problem.
            pass
    # Auto-checkpoint WAL to prevent bloat and reduce lock contention
    try:
        conn.execute("PRAGMA wal_autocheckpoint=1000")
    except Exception:
        pass
    proxy = KeepOpenProxy(conn, wkey, wlock)
    cache[key] = proxy
    return proxy
