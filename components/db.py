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

import sqlite3
import threading

_PER_CONNECTION_PRAGMAS = (
    "PRAGMA journal_mode=WAL",          # MUST be first
    "PRAGMA synchronous=NORMAL",        # safe + fast with WAL
    "PRAGMA busy_timeout=30000",        # 30 s
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

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    # ── close() override ────────────────────────────────────────────
    def close(self) -> None:  # noqa: D401
        """No-op. The cache owns the connection lifecycle."""
        return None

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
        # so this is consistent with stdlib behaviour.
        self._conn.__exit__(exc_type, exc_val, exc_tb)

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
    timeout: float = 30.0,
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
    cached = cache.get(key)
    if cached is not None:
        try:
            # Liveness probe — operate on the underlying connection
            # directly so we don't accidentally hit a proxied method.
            real = cached._conn if isinstance(cached, KeepOpenProxy) else cached
            real.execute("SELECT 1")
            # Always hand callers a proxy — even if a non-proxy slipped
            # into the cache (defensive against future bugs).
            return cached if isinstance(cached, KeepOpenProxy) else KeepOpenProxy(cached)
        except sqlite3.Error:
            cache.pop(key, None)
            _really_close(cached)

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
    proxy = KeepOpenProxy(conn)
    cache[key] = proxy
    return proxy
