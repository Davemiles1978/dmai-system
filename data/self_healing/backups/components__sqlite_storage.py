"""
SQLite Persistent Storage for DMAI
Replaces Neo4j — all critical state persisted locally in data/dmai.db
Survives Render restarts (Render mounts /opt/render/project/src as persistent disk when configured)
"""

import os
import json
import sqlite3
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger('dmai_sqlite')

DB_PATH = Path(os.getenv('SQLITE_DB_PATH', 'data/dmai.db'))


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class SQLiteStorage:
    """
    Drop-in replacement for Neo4jStorage.
    All methods have identical signatures so callers need no changes.
    """

    _lock = threading.Lock()

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()
        logger.info(f"✅ SQLite storage ready at {DB_PATH.resolve()}")

    # ── Connection ────────────────────────────────────────────────────────────

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _get_conn()
        return self._conn

    def is_available(self) -> bool:
        try:
            self._db().execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        db = _get_conn()
        with self._lock:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS evolution_state (
                    id TEXT PRIMARY KEY DEFAULT 'core',
                    consciousness REAL DEFAULT 0,
                    neurons INTEGER DEFAULT 0,
                    synapses INTEGER DEFAULT 0,
                    evolution_cycles INTEGER DEFAULT 0,
                    evolution_count INTEGER DEFAULT 0,
                    last_update TEXT
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    created TEXT,
                    updated TEXT,
                    user TEXT DEFAULT 'master',
                    priority TEXT DEFAULT 'normal'
                );

                CREATE TABLE IF NOT EXISTS persona (
                    id TEXT PRIMARY KEY DEFAULT 'dmai',
                    traits TEXT DEFAULT '{}',
                    speaking_style TEXT DEFAULT 'emerging',
                    emotional_state TEXT DEFAULT 'neutral',
                    consciousness_level REAL DEFAULT 0,
                    last_update TEXT
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    user TEXT,
                    message TEXT,
                    response TEXT,
                    is_task INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS funding_state (
                    id TEXT PRIMARY KEY DEFAULT 'core',
                    completed_avenues TEXT DEFAULT '[]',
                    concepts_learned INTEGER DEFAULT 0,
                    concepts_total INTEGER DEFAULT 0,
                    learning_active INTEGER DEFAULT 0,
                    training_complete INTEGER DEFAULT 0,
                    progress REAL DEFAULT 0,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS funding_avenues (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    progress REAL DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS funding_concepts (
                    id TEXT PRIMARY KEY,
                    learned_at TEXT
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    service TEXT,
                    source TEXT,
                    validated INTEGER DEFAULT 0,
                    created_at TEXT,
                    last_used TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_api_keys_service ON api_keys(service);
                CREATE TABLE IF NOT EXISTS admin_api_keys (
                    provider_id TEXT PRIMARY KEY,
                    api_key TEXT NOT NULL,
                    updated_at TEXT
                );
            """)

            db.commit()
        db.close()

    # ── Evolution State ───────────────────────────────────────────────────────

    def save_evolution_state(self, state: Dict) -> bool:
        try:
            with self._lock:
                db = self._db()
                db.execute("""
                    INSERT INTO evolution_state (id, consciousness, neurons, synapses,
                        evolution_cycles, evolution_count, last_update)
                    VALUES ('core', :consciousness, :neurons, :synapses, :cycles, :evolution_count, :ts)
                    ON CONFLICT(id) DO UPDATE SET
                        consciousness=excluded.consciousness,
                        neurons=excluded.neurons,
                        synapses=excluded.synapses,
                        evolution_cycles=excluded.evolution_cycles,
                        evolution_count=excluded.evolution_count,
                        last_update=excluded.last_update
                """, {
                    'consciousness': state.get('consciousness', 0),
                    'neurons': state.get('neurons', 0),
                    'synapses': state.get('synapses', 0),
                    'cycles': state.get('evolution_cycles', 0),
                    'evolution_count': state.get('evolution_count', 0),
                    'ts': datetime.now().isoformat(),
                })
                db.commit()
            return True
        except Exception as e:
            logger.error(f"save_evolution_state failed: {e}")
            return False

    def load_evolution_state(self) -> Optional[Dict]:
        try:
            row = self._db().execute(
                "SELECT * FROM evolution_state WHERE id='core'"
            ).fetchone()
            if row:
                return {
                    'consciousness': row['consciousness'],
                    'neurons': row['neurons'],
                    'synapses': row['synapses'],
                    'evolution_cycles': row['evolution_cycles'],
                    'evolution_count': row['evolution_count'],
                    'last_update': row['last_update'],
                }
        except Exception as e:
            logger.error(f"load_evolution_state failed: {e}")
        return None

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def save_task(self, task: Dict) -> bool:
        try:
            task_id = task.get('id', task.get('description', str(datetime.now().timestamp())))
            with self._lock:
                db = self._db()
                db.execute("""
                    INSERT INTO tasks (id, description, status, created, updated, user, priority)
                    VALUES (:id, :desc, :status, :created, :updated, :user, :priority)
                    ON CONFLICT(id) DO UPDATE SET
                        description=excluded.description,
                        status=excluded.status,
                        updated=excluded.updated,
                        priority=excluded.priority
                """, {
                    'id': task_id,
                    'desc': task.get('description', ''),
                    'status': task.get('status', 'pending'),
                    'created': task.get('created', datetime.now().isoformat()),
                    'updated': datetime.now().isoformat(),
                    'user': task.get('user', 'master'),
                    'priority': task.get('priority', 'normal'),
                })
                db.commit()
            return True
        except Exception as e:
            logger.error(f"save_task failed: {e}")
            return False

    def load_tasks(self, status: Optional[str] = None) -> List[Dict]:
        try:
            db = self._db()
            if status:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY created DESC", (status,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM tasks ORDER BY created DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"load_tasks failed: {e}")
            return []

    # ── Persona ───────────────────────────────────────────────────────────────

    def save_persona(self, persona: Dict) -> bool:
        try:
            with self._lock:
                db = self._db()
                db.execute("""
                    INSERT INTO persona (id, traits, speaking_style, emotional_state,
                        consciousness_level, last_update)
                    VALUES ('dmai', :traits, :style, :emotion, :consciousness, :ts)
                    ON CONFLICT(id) DO UPDATE SET
                        traits=excluded.traits,
                        speaking_style=excluded.speaking_style,
                        emotional_state=excluded.emotional_state,
                        consciousness_level=excluded.consciousness_level,
                        last_update=excluded.last_update
                """, {
                    'traits': json.dumps(persona.get('traits', {})),
                    'style': persona.get('speaking_style', 'emerging'),
                    'emotion': persona.get('emotional_state', 'neutral'),
                    'consciousness': persona.get('consciousness_level', 0),
                    'ts': datetime.now().isoformat(),
                })
                db.commit()
            return True
        except Exception as e:
            logger.error(f"save_persona failed: {e}")
            return False

    def load_persona(self) -> Optional[Dict]:
        try:
            row = self._db().execute(
                "SELECT * FROM persona WHERE id='dmai'"
            ).fetchone()
            if row:
                return {
                    'traits': json.loads(row['traits'] or '{}'),
                    'speaking_style': row['speaking_style'],
                    'emotional_state': row['emotional_state'],
                    'consciousness_level': row['consciousness_level'],
                    'last_update': row['last_update'],
                }
        except Exception as e:
            logger.error(f"load_persona failed: {e}")
        return None

    # ── Conversations ─────────────────────────────────────────────────────────

    def save_conversation(self, user: str, message: str, response: str,
                          important: bool = False) -> bool:
        is_task = any(w in message.lower() for w in ['task', 'todo', 'remind', 'remember'])
        if not (important or is_task):
            return False
        try:
            with self._lock:
                db = self._db()
                db.execute("""
                    INSERT INTO conversations (ts, user, message, response, is_task)
                    VALUES (?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), user, message[:500], response[:500], int(is_task)))
                db.commit()
            return True
        except Exception as e:
            logger.error(f"save_conversation failed: {e}")
            return False

    def load_tasks_from_conversations(self) -> List[Dict]:
        try:
            rows = self._db().execute("""
                SELECT * FROM conversations WHERE is_task=1
                ORDER BY ts DESC LIMIT 50
            """).fetchall()
            return [{'description': r['message'], 'status': 'pending',
                     'created': r['ts'], 'user': r['user']} for r in rows]
        except Exception as e:
            logger.error(f"load_tasks_from_conversations failed: {e}")
            return []

    # ── Funding State ─────────────────────────────────────────────────────────

    def save_funding_state(self, revenue_avenues: Dict, learned_concepts: set,
                           training_complete: bool, learning_active: bool) -> bool:
        try:
            with self._lock:
                db = self._db()
                completed = [k for k, v in revenue_avenues.items() if v.get('completed')]
                db.execute("""
                    INSERT INTO funding_state
                        (id, completed_avenues, concepts_learned, concepts_total,
                         learning_active, training_complete, progress, updated_at)
                    VALUES ('core', :completed, :learned, :total, :active, :complete, :progress, :ts)
                    ON CONFLICT(id) DO UPDATE SET
                        completed_avenues=excluded.completed_avenues,
                        concepts_learned=excluded.concepts_learned,
                        concepts_total=excluded.concepts_total,
                        learning_active=excluded.learning_active,
                        training_complete=excluded.training_complete,
                        progress=excluded.progress,
                        updated_at=excluded.updated_at
                """, {
                    'completed': json.dumps(completed),
                    'learned': len(learned_concepts),
                    'total': sum(len(d['topics']) for d in revenue_avenues.values()),
                    'active': int(learning_active),
                    'complete': int(training_complete),
                    'progress': (len(learned_concepts) /
                                 max(1, sum(len(d['topics']) for d in revenue_avenues.values()))) * 100,
                    'ts': datetime.now().isoformat(),
                })
                for name, av in revenue_avenues.items():
                    db.execute("""
                        INSERT INTO funding_avenues (id, name, progress, completed, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            progress=excluded.progress,
                            completed=excluded.completed,
                            updated_at=excluded.updated_at
                    """, (name, av['name'], av.get('progress', 0),
                          int(av.get('completed', False)), datetime.now().isoformat()))
                for concept in learned_concepts:
                    db.execute("""
                        INSERT OR IGNORE INTO funding_concepts (id, learned_at)
                        VALUES (?, ?)
                    """, (concept, datetime.now().isoformat()))
                db.commit()
            return True
        except Exception as e:
            logger.error(f"save_funding_state failed: {e}")
            return False

    def load_funding_state(self) -> Optional[Dict]:
        """Returns dict with keys: training_complete, avenue_progress, learned_concepts"""
        try:
            db = self._db()
            row = db.execute(
                "SELECT * FROM funding_state WHERE id='core'"
            ).fetchone()
            if not row:
                return None
            avenues = {r['id']: {'progress': r['progress'], 'completed': bool(r['completed'])}
                       for r in db.execute("SELECT * FROM funding_avenues").fetchall()}
            concepts = {r['id'] for r in db.execute("SELECT id FROM funding_concepts").fetchall()}
            return {
                'training_complete': bool(row['training_complete']),
                'avenue_progress': avenues,
                'learned_concepts': concepts,
            }
        except Exception as e:
            logger.error(f"load_funding_state failed: {e}")
            return None

    # ── API Keys ──────────────────────────────────────────────────────────────

    def store_api_keys(self, keys: List) -> bool:
        """Store validated APIKey objects (duck-typed: .key .service .source .validated .created_at)"""
        try:
            with self._lock:
                db = self._db()
                for k in keys:
                    db.execute("""
                        INSERT INTO api_keys (key, service, source, validated, created_at, last_used)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            validated=excluded.validated,
                            last_used=excluded.last_used
                    """, (k.key, k.service, k.source, int(k.validated),
                          k.created_at, None))
                db.commit()
            logger.info(f"💾 Stored {len(keys)} API keys in SQLite")
            return True
        except Exception as e:
            logger.error(f"store_api_keys failed: {e}")
            return False

    # ── Backup / Restore (compatibility shims) ────────────────────────────────

    # ── Admin API Key Management ────────────────────────────────────────────
    def get_api_key(self, provider_id: str) -> Optional[str]:
        """Retrieve an admin-set API key by provider_id."""
        try:
            db = self._db()
            row = db.execute(
                "SELECT api_key FROM admin_api_keys WHERE provider_id = ?",
                (provider_id,)
            ).fetchone()
            db.close()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"get_api_key failed: {e}")
            return None

    def set_api_key(self, provider_id: str, key: str) -> bool:
        """Persist an admin-set API key (upsert)."""
        import datetime
        try:
            db = self._db()
            db.execute(
                """INSERT INTO admin_api_keys (provider_id, api_key, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(provider_id) DO UPDATE SET
                   api_key=excluded.api_key, updated_at=excluded.updated_at""",
                (provider_id, key, datetime.datetime.utcnow().isoformat())
            )
            db.commit()
            db.close()
            return True
        except Exception as e:
            logger.error(f"set_api_key failed: {e}")
            return False

    def delete_api_key(self, provider_id: str) -> bool:
        """Remove an admin-set API key."""
        try:
            db = self._db()
            db.execute("DELETE FROM admin_api_keys WHERE provider_id = ?", (provider_id,))
            db.commit()
            db.close()
            return True
        except Exception as e:
            logger.error(f"delete_api_key failed: {e}")
            return False

    def backup_all(self, local_state: Dict) -> bool:
        self.save_evolution_state(local_state.get('evolution', {}))
        self.save_persona(local_state.get('persona', {}))
        for task in local_state.get('tasks', []):
            self.save_task(task)
        logger.info("✅ Full backup to SQLite complete")
        return True

    def restore_all(self) -> Dict:
        return {
            'evolution': self.load_evolution_state(),
            'persona': self.load_persona(),
            'tasks': self.load_tasks(),
            'conversation_tasks': self.load_tasks_from_conversations(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_sqlite_storage: Optional[SQLiteStorage] = None


def get_sqlite_storage() -> SQLiteStorage:
    global _sqlite_storage
    if _sqlite_storage is None:
        _sqlite_storage = SQLiteStorage()
    return _sqlite_storage


# Back-compat alias so any code that imports get_neo4j_storage still works
def get_neo4j_storage() -> SQLiteStorage:
    return get_sqlite_storage()
