"""Shared SQLite connection helper for DMAI knowledge DB.

All connections to dmai_knowledge.db MUST go through safe_open_kdb() to
guarantee consistent WAL mode, busy_timeout, and synchronous settings.

Root cause of recurring DB corruption (see DB_CORRUPTION_AUDIT_2026-06-27.md):
30 component files were opening dmai_knowledge.db with a bare
`sqlite3.connect(...)` that defaults to rollback-journal mode. In a 16-thread
single-process Gunicorn deployment, mixing WAL connections with rollback-journal
connections on the same file causes the SQLite engine to write through both a
`-wal` file and a `-journal` file simultaneously, corrupting the schema page
("database disk image is malformed").

SQLite gotcha: setting `PRAGMA journal_mode=WAL` on one connection does NOT
make sibling connections speak WAL. The pragma must be applied PER CONNECTION.
That is the whole point of this helper.

Usage:
    from components.db import safe_open_kdb
    conn = safe_open_kdb(self.db_path)
    # ... use conn ...

Thread-safety: each call returns a NEW connection. Do not share connections
across threads. `check_same_thread=False` is set so a connection created in
one thread can be closed by another, but concurrent USE from multiple threads
is still unsafe.
"""
from __future__ import annotations

import sqlite3


def safe_open_kdb(
    path: str,
    *,
    timeout: float = 30.0,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a connection to a DMAI SQLite DB with safe defaults.

    Args:
        path: filesystem path to the .db file.
        timeout: SQLite busy timeout in seconds (default 30s). Maps to both
            the Python-level connect timeout AND the SQLite-level
            busy_timeout PRAGMA (30000ms).
        read_only: if True, open via URI in `mode=ro` so writes raise.

    Returns:
        sqlite3.Connection with the following pragmas applied:
          - journal_mode = WAL          (prevents mixed-mode corruption)
          - busy_timeout = 30000        (30s, matches connect timeout)
          - synchronous  = NORMAL       (fast, safe with WAL)
          - foreign_keys = ON           (defensive)
    """
    if read_only:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
    else:
        conn = sqlite3.connect(
            path,
            timeout=timeout,
            check_same_thread=False,
        )
    # Pragmas applied PER CONNECTION (this is the SQLite gotcha).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
