"""
pg_storage.py — PostgreSQL drop-in replacement for sqlite_storage.py

Implements exactly the same public interface as SQLiteStorage so every
caller can swap without changes.  Falls back to SQLiteStorage automatically
if DATABASE_URL is not set (local dev / missing env var).

Uses psycopg2 (already in requirements.txt) with a simple thread-safe
connection pool.  No SQLAlchemy dependency needed.

Environment variables:
  DATABASE_URL   — standard Render Postgres URL
                   e.g. postgresql://user:pass@host:5432/dbname
                   Also accepts postgres:// (rewritten to postgresql://)
"""

import os
import json
import logging
import threading
import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger("PGStorage")

# ── Connection pool (simple, thread-safe) ─────────────────────────────────────

_pool_lock = threading.Lock()
_connections: list = []
_MAX_POOL = 4


def _get_dsn() -> str:
    url = os.environ.get("DATABASE_URL", "")
    # Render uses postgres:// but psycopg2 needs postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _get_conn():
    """Get a connection from the pool, or create a new one."""
    import psycopg2
    import psycopg2.extras
    with _pool_lock:
        while _connections:
            conn = _connections.pop()
            try:
                conn.cursor().execute("SELECT 1")
                return conn
            except Exception:
                pass  # stale — discard
    conn = psycopg2.connect(_get_dsn())
    conn.autocommit = False
    return conn


def _return_conn(conn):
    with _pool_lock:
        if len(_connections) < _MAX_POOL:
            try:
                conn.rollback()
                _connections.append(conn)
                return
            except Exception:
                pass
    try:
        conn.close()
    except Exception:
        pass


# ── Schema DDL ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evolution_state (
    id          SERIAL PRIMARY KEY,
    state_json  TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    status      TEXT DEFAULT 'pending',
    priority    TEXT DEFAULT 'medium',
    data_json   TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS persona (
    id          SERIAL PRIMARY KEY,
    data_json   TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    user_msg    TEXT,
    message     TEXT,
    response    TEXT,
    context     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS funding_state (
    id              SERIAL PRIMARY KEY,
    revenue_avenues TEXT,
    learned_concepts TEXT,
    performance_data TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS funding_avenues (
    id          SERIAL PRIMARY KEY,
    avenue_json TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS funding_concepts (
    concept     TEXT PRIMARY KEY,
    added_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    key         TEXT PRIMARY KEY,
    service     TEXT,
    source      TEXT,
    validated   INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_used   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_service ON api_keys(service);

CREATE TABLE IF NOT EXISTS admin_api_keys (
    provider_id TEXT PRIMARY KEY,
    api_key     TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS insights (
    id              TEXT PRIMARY KEY,
    insight_text    TEXT,
    entity_type     TEXT,
    entities        TEXT,
    relationship    TEXT,
    confidence      REAL DEFAULT 0.5,
    source_url      TEXT,
    source_title    TEXT,
    source_type     TEXT DEFAULT 'web',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_insights_entity ON insights(entity_type);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at DESC);
"""


class PGStorage:
    """
    PostgreSQL-backed persistent storage for DMAI.
    Public interface is identical to SQLiteStorage — callers need no changes.
    """

    def __init__(self):
        self._available = False
        dsn = _get_dsn()
        if not dsn:
            logger.warning("DATABASE_URL not set — PGStorage unavailable, will fall back to SQLite")
            return
        try:
            self._init_schema()
            self._available = True
            logger.info("PGStorage connected and schema initialised")
        except Exception as e:
            logger.error("PGStorage init failed: %s", e)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _exec(self, sql: str, params=(), fetch: str = "none"):
        """Execute SQL, return rows if fetch='all'|'one', else None."""
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == "all":
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                    conn.commit()
                    return rows
                elif fetch == "one":
                    row = cur.fetchone()
                    if row:
                        cols = [d[0] for d in cur.description]
                        conn.commit()
                        return dict(zip(cols, row))
                    conn.commit()
                    return None
                else:
                    conn.commit()
                    return None
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise e
        finally:
            _return_conn(conn)

    def _init_schema(self):
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
                # ---- Idempotent migrations for older deployments ----
                # Add missing columns expected by the current schema.
                _migrations = [
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS service TEXT",
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS source TEXT",
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS validated INTEGER DEFAULT 0",
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS last_used TIMESTAMPTZ",
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_service ON api_keys(service)",
                ]
                for stmt in _migrations:
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        logger.warning("PGStorage migration skipped (%s): %s", stmt[:60], e)
                        conn.rollback()
                        # Reopen a clean cursor on the same connection.
                        cur.close()
                        cur = conn.cursor()
            conn.commit()
        finally:
            _return_conn(conn)

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available

    def close(self):
        with _pool_lock:
            for conn in _connections:
                try:
                    conn.close()
                except Exception:
                    pass
            _connections.clear()

    # ── Evolution State ───────────────────────────────────────────────────────

    def save_evolution_state(self, state: Dict) -> bool:
        try:
            self._exec(
                "DELETE FROM evolution_state"
            )
            self._exec(
                "INSERT INTO evolution_state (state_json) VALUES (%s)",
                (json.dumps(state),)
            )
            return True
        except Exception as e:
            logger.error("save_evolution_state: %s", e)
            return False

    def load_evolution_state(self) -> Optional[Dict]:
        try:
            row = self._exec(
                "SELECT state_json FROM evolution_state ORDER BY id DESC LIMIT 1",
                fetch="one"
            )
            return json.loads(row["state_json"]) if row else None
        except Exception as e:
            logger.error("load_evolution_state: %s", e)
            return None

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def save_task(self, task: Dict) -> bool:
        try:
            tid = task.get("id", task.get("title", "unknown"))
            self._exec(
                """INSERT INTO tasks (id, title, status, priority, data_json, updated_at)
                   VALUES (%s, %s, %s, %s, %s, NOW())
                   ON CONFLICT(id) DO UPDATE SET
                     title=EXCLUDED.title, status=EXCLUDED.status,
                     priority=EXCLUDED.priority, data_json=EXCLUDED.data_json,
                     updated_at=NOW()""",
                (tid, task.get("title", ""), task.get("status", "pending"),
                 task.get("priority", "medium"), json.dumps(task))
            )
            return True
        except Exception as e:
            logger.error("save_task: %s", e)
            return False

    def load_tasks(self, status: Optional[str] = None) -> List[Dict]:
        try:
            if status:
                rows = self._exec(
                    "SELECT data_json FROM tasks WHERE status=%s ORDER BY created_at DESC",
                    (status,), fetch="all"
                )
            else:
                rows = self._exec(
                    "SELECT data_json FROM tasks ORDER BY created_at DESC",
                    fetch="all"
                )
            return [json.loads(r["data_json"]) for r in (rows or [])]
        except Exception as e:
            logger.error("load_tasks: %s", e)
            return []

    # ── Persona ───────────────────────────────────────────────────────────────

    def save_persona(self, persona: Dict) -> bool:
        try:
            self._exec("DELETE FROM persona")
            self._exec(
                "INSERT INTO persona (data_json) VALUES (%s)",
                (json.dumps(persona),)
            )
            return True
        except Exception as e:
            logger.error("save_persona: %s", e)
            return False

    def load_persona(self) -> Optional[Dict]:
        try:
            row = self._exec(
                "SELECT data_json FROM persona ORDER BY id DESC LIMIT 1",
                fetch="one"
            )
            return json.loads(row["data_json"]) if row else None
        except Exception as e:
            logger.error("load_persona: %s", e)
            return None

    # ── Conversations ─────────────────────────────────────────────────────────

    def save_conversation(self, user: str, message: str, response: str,
                          context: str = "") -> bool:
        try:
            self._exec(
                "INSERT INTO conversations (user_msg, message, response, context) VALUES (%s,%s,%s,%s)",
                (user, message, response, context)
            )
            return True
        except Exception as e:
            logger.error("save_conversation: %s", e)
            return False

    def load_tasks_from_conversations(self) -> List[Dict]:
        try:
            rows = self._exec(
                "SELECT message, response, created_at FROM conversations ORDER BY created_at DESC LIMIT 50",
                fetch="all"
            )
            return rows or []
        except Exception as e:
            logger.error("load_tasks_from_conversations: %s", e)
            return []

    # ── Funding State ─────────────────────────────────────────────────────────

    def save_funding_state(self, revenue_avenues: Dict, learned_concepts: set,
                           performance_data: Dict) -> bool:
        try:
            self._exec("DELETE FROM funding_state")
            self._exec(
                "INSERT INTO funding_state (revenue_avenues, learned_concepts, performance_data) VALUES (%s,%s,%s)",
                (json.dumps(revenue_avenues), json.dumps(list(learned_concepts)),
                 json.dumps(performance_data))
            )
            return True
        except Exception as e:
            logger.error("save_funding_state: %s", e)
            return False

    def load_funding_state(self) -> Optional[Dict]:
        try:
            row = self._exec(
                "SELECT revenue_avenues, learned_concepts, performance_data FROM funding_state ORDER BY id DESC LIMIT 1",
                fetch="one"
            )
            if not row:
                return None
            return {
                "revenue_avenues":  json.loads(row["revenue_avenues"] or "{}"),
                "learned_concepts": set(json.loads(row["learned_concepts"] or "[]")),
                "performance_data": json.loads(row["performance_data"] or "{}"),
            }
        except Exception as e:
            logger.error("load_funding_state: %s", e)
            return None

    # ── API Keys (harvester bulk store) ───────────────────────────────────────

    def store_api_keys(self, keys: List) -> bool:
        try:
            for k in keys:
                self._exec(
                    """INSERT INTO api_keys (key, service, source, validated, created_at)
                       VALUES (%s,%s,%s,%s,NOW())
                       ON CONFLICT(key) DO UPDATE SET
                         service=EXCLUDED.service, validated=EXCLUDED.validated""",
                    (k.get("key",""), k.get("service",""), k.get("source",""),
                     1 if k.get("validated") else 0)
                )
            return True
        except Exception as e:
            logger.error("store_api_keys: %s", e)
            return False

    # ── Admin API Keys (per-provider set/get/delete) ──────────────────────────

    def get_api_key(self, provider_id: str) -> Optional[str]:
        try:
            row = self._exec(
                "SELECT api_key FROM admin_api_keys WHERE provider_id=%s",
                (provider_id,), fetch="one"
            )
            return row["api_key"] if row else None
        except Exception as e:
            logger.error("get_api_key: %s", e)
            return None

    def set_api_key(self, provider_id: str, key: str) -> bool:
        try:
            self._exec(
                """INSERT INTO admin_api_keys (provider_id, api_key, updated_at)
                   VALUES (%s,%s,NOW())
                   ON CONFLICT(provider_id) DO UPDATE SET
                     api_key=EXCLUDED.api_key, updated_at=NOW()""",
                (provider_id, key)
            )
            return True
        except Exception as e:
            logger.error("set_api_key: %s", e)
            return False

    def delete_api_key(self, provider_id: str) -> bool:
        try:
            self._exec(
                "DELETE FROM admin_api_keys WHERE provider_id=%s",
                (provider_id,)
            )
            return True
        except Exception as e:
            logger.error("delete_api_key: %s", e)
            return False

    # ── Insights (knowledge DB) ───────────────────────────────────────────────

    def insert_insight(self, insight_id: str, text: str, entity_type: str = "",
                       entities: str = "", relationship: str = "",
                       confidence: float = 0.5, source_url: str = "",
                       source_title: str = "", source_type: str = "web") -> bool:
        try:
            self._exec(
                """INSERT INTO insights
                   (id, insight_text, entity_type, entities, relationship,
                    confidence, source_url, source_title, source_type)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(id) DO NOTHING""",
                (insight_id, text, entity_type, entities, relationship,
                 confidence, source_url, source_title, source_type)
            )
            return True
        except Exception as e:
            logger.error("insert_insight: %s", e)
            return False

    def search_insights(self, query: str = "", limit: int = 20,
                        entity_type: str = "") -> List[Dict]:
        try:
            if entity_type:
                rows = self._exec(
                    """SELECT insight_text, source_title, entity_type, confidence, created_at
                       FROM insights WHERE entity_type=%s
                       ORDER BY created_at DESC LIMIT %s""",
                    (entity_type, limit), fetch="all"
                )
            elif query:
                rows = self._exec(
                    """SELECT insight_text, source_title, entity_type, confidence, created_at
                       FROM insights WHERE insight_text ILIKE %s
                       ORDER BY created_at DESC LIMIT %s""",
                    (f"%{query}%", limit), fetch="all"
                )
            else:
                rows = self._exec(
                    """SELECT insight_text, source_title, entity_type, confidence, created_at
                       FROM insights ORDER BY created_at DESC LIMIT %s""",
                    (limit,), fetch="all"
                )
            return rows or []
        except Exception as e:
            logger.error("search_insights: %s", e)
            return []

    def count_insights(self) -> int:
        try:
            row = self._exec("SELECT COUNT(*) AS c FROM insights", fetch="one")
            return row["c"] if row else 0
        except Exception as e:
            logger.error("count_insights: %s", e)
            return 0

    # ── Backup / Restore (compatibility shim) ────────────────────────────────

    def backup_all(self, local_state: Dict) -> bool:
        self.save_evolution_state(local_state.get("evolution", {}))
        self.save_persona(local_state.get("persona", {}))
        for task in local_state.get("tasks", []):
            self.save_task(task)
        logger.info("Full backup to PostgreSQL complete")
        return True

    def restore_all(self) -> Dict:
        return {
            "evolution":          self.load_evolution_state(),
            "persona":            self.load_persona(),
            "tasks":              self.load_tasks(),
            "conversation_tasks": self.load_tasks_from_conversations(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_pg_storage: Optional[PGStorage] = None
_storage_lock = threading.Lock()


def get_pg_storage() -> PGStorage:
    global _pg_storage
    with _storage_lock:
        if _pg_storage is None:
            _pg_storage = PGStorage()
    return _pg_storage


def get_storage():
    """
    Returns PGStorage if DATABASE_URL is set and available,
    otherwise falls back to SQLiteStorage.
    This is the preferred entry point for all callers.
    """
    pg = get_pg_storage()
    if pg.is_available():
        return pg
    logger.warning("PGStorage unavailable — falling back to SQLiteStorage")
    from components.sqlite_storage import get_sqlite_storage
    return get_sqlite_storage()


# Alias kept for backward compatibility with callers that import get_sqlite_storage
def get_sqlite_storage():
    return get_storage()


# Alias used by dmai_core_complete.py
get_neo4j_storage = get_storage
