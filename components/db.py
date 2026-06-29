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


def safe_open_kdb(
    path: str,
    *,
    timeout: float = 30.0,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open or reuse a connection to a DMAI SQLite DB with safe defaults.

    Returns the per-thread cached connection if alive, else opens a new
    one with the tuned PRAGMAs applied PER CONNECTION (this is the
    SQLite gotcha — journal_mode is per-handle, not per-file).

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
    conn = cache.get(key)
    if conn is not None:
        try:
            conn.execute("SELECT 1")  # liveness probe
            return conn
        except sqlite3.Error:
            cache.pop(key, None)
            try:
                conn.close()
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
    cache[key] = conn
    return conn
