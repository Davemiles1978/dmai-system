"""
ConversationMemory \u2014 SQLite-backed multi-turn chat history.

Stores every (session_id, role, content, ts) tuple in mon-style table
and exposes get/append/clear helpers + summary stats for the dashboard.

Tables:
  conv_messages (session_id, ts, role, content, meta_json)
  conv_sessions (session_id, started_ts, last_ts, msg_count, title)

Designed so the chat engine can call:
    memory.append(session_id, "user", "...")
    history = memory.recent(session_id, n=20)
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """SQLite multi-turn conversation store."""

    def __init__(self, data_path: str | Path = "data"):
        self.data_path = str(data_path).rstrip("/")
        self.db_path = os.path.join(self.data_path, "dmai_knowledge.db")
        self._lock = threading.RLock()
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    # ── schema ─────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL;")
        return c

    def _ensure_tables(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conv_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta_json TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_conv_messages_session
                    ON conv_messages(session_id, ts);

                CREATE TABLE IF NOT EXISTS conv_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_ts TEXT NOT NULL,
                    last_ts TEXT NOT NULL,
                    msg_count INTEGER NOT NULL DEFAULT 0,
                    title TEXT
                );
            """)

    # ── write ──────────────────────────────────────────────────────────
    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"invalid role: {role}")
        ts = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(meta) if meta else None
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO conv_messages (session_id, ts, role, content, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, ts, role, content, meta_json),
            )
            msg_id = cur.lastrowid
            conn.execute(
                "INSERT INTO conv_sessions (session_id, started_ts, last_ts, msg_count, title) "
                "VALUES (?, ?, ?, 1, NULL) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  last_ts = excluded.last_ts, "
                "  msg_count = msg_count + 1",
                (session_id, ts, ts),
            )
        return msg_id

    # ── read ───────────────────────────────────────────────────────────
    def recent(self, session_id: str, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, ts, role, content, meta_json FROM conv_messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, int(n)),
            ).fetchall()
        out = []
        for r in reversed(rows):
            out.append({
                "id": r[0],
                "ts": r[1],
                "role": r[2],
                "content": r[3],
                "meta": json.loads(r[4]) if r[4] else None,
            })
        return out

    def sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, started_ts, last_ts, msg_count, title "
                "FROM conv_sessions ORDER BY last_ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"session_id": r[0], "started_ts": r[1], "last_ts": r[2],
             "msg_count": r[3], "title": r[4]}
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._conn() as conn:
            sess = conn.execute("SELECT COUNT(*) FROM conv_sessions").fetchone()[0]
            msgs = conn.execute("SELECT COUNT(*) FROM conv_messages").fetchone()[0]
            last = conn.execute(
                "SELECT MAX(last_ts) FROM conv_sessions"
            ).fetchone()[0]
        return {"sessions": sess, "messages": msgs, "last_activity": last}

    # ── maintenance ────────────────────────────────────────────────────
    def clear(self, session_id: str) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM conv_messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "DELETE FROM conv_sessions WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount

    def set_title(self, session_id: str, title: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE conv_sessions SET title = ? WHERE session_id = ?",
                (title, session_id),
            )
