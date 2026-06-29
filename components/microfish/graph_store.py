"""
SQLite-backed knowledge graph + prediction state store.
Replaces Zep Cloud + Neo4j entirely.
"""
from __future__ import annotations
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from components.db import safe_open_kdb

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mf_predictions (
    id TEXT PRIMARY KEY,
    requirement TEXT NOT NULL,
    seed_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    verdict_json TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);
CREATE TABLE IF NOT EXISTS mf_entities (
    prediction_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT,
    PRIMARY KEY (prediction_id, entity_id)
);
CREATE TABLE IF NOT EXISTS mf_relations (
    prediction_id TEXT NOT NULL,
    rel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT
);
CREATE TABLE IF NOT EXISTS mf_agents (
    prediction_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    persona_json TEXT NOT NULL,
    platform TEXT,
    PRIMARY KEY (prediction_id, agent_id)
);
CREATE TABLE IF NOT EXISTS mf_actions (
    prediction_id TEXT NOT NULL,
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    content TEXT,
    target_id TEXT,
    round_num INTEGER NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mf_actions_pred ON mf_actions(prediction_id, round_num);
CREATE INDEX IF NOT EXISTS idx_mf_entities_pred ON mf_entities(prediction_id);
CREATE INDEX IF NOT EXISTS idx_mf_relations_pred ON mf_relations(prediction_id);
"""


class GraphStore:
    def __init__(self, db_path: str = "data/dmai_knowledge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _conn(self):
        c = safe_open_kdb(self.db_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with _LOCK, self._conn() as c:
            c.executescript(_SCHEMA)

    # ----- predictions -----
    def create_prediction(self, requirement: str, seed_hash: str = "") -> str:
        pid = uuid.uuid4().hex[:16]
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mf_predictions (id, requirement, seed_hash, status, created_at) VALUES (?,?,?,?,?)",
                (pid, requirement, seed_hash, "running", time.time()),
            )
        return pid

    def finalize_prediction(self, pid: str, verdict: Dict[str, Any], status: str = "complete"):
        with _LOCK, self._conn() as c:
            c.execute(
                "UPDATE mf_predictions SET status=?, verdict_json=?, completed_at=? WHERE id=?",
                (status, json.dumps(verdict), time.time(), pid),
            )

    def get_prediction(self, pid: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM mf_predictions WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("verdict_json"):
            try:
                d["verdict"] = json.loads(d["verdict_json"])
            except Exception:
                d["verdict"] = None
        return d

    # ----- entities + relations -----
    def add_entities(self, pid: str, entities: List[Dict[str, Any]]):
        if not entities:
            return
        with _LOCK, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO mf_entities (prediction_id, entity_id, label, type, attrs_json) VALUES (?,?,?,?,?)",
                [(pid, e.get("id") or e.get("label"), e.get("label", ""), e.get("type", "entity"),
                  json.dumps(e.get("attrs", {}))) for e in entities],
            )

    def add_relations(self, pid: str, relations: List[Dict[str, Any]]):
        if not relations:
            return
        with _LOCK, self._conn() as c:
            c.executemany(
                "INSERT INTO mf_relations (prediction_id, from_id, to_id, type, attrs_json) VALUES (?,?,?,?,?)",
                [(pid, r.get("from", ""), r.get("to", ""), r.get("type", "related"),
                  json.dumps(r.get("attrs", {}))) for r in relations],
            )

    def get_entities(self, pid: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT entity_id, label, type, attrs_json FROM mf_entities WHERE prediction_id=?", (pid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["attrs"] = json.loads(d.pop("attrs_json") or "{}")
            except Exception:
                d["attrs"] = {}
            out.append(d)
        return out

    def get_relations(self, pid: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT from_id, to_id, type, attrs_json FROM mf_relations WHERE prediction_id=?", (pid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["attrs"] = json.loads(d.pop("attrs_json") or "{}")
            except Exception:
                d["attrs"] = {}
            out.append(d)
        return out

    # ----- agents + actions -----
    def add_agents(self, pid: str, agents: List[Dict[str, Any]]):
        if not agents:
            return
        with _LOCK, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO mf_agents (prediction_id, agent_id, persona_json, platform) VALUES (?,?,?,?)",
                [(pid, a.get("id"), json.dumps(a), a.get("platform", "generic")) for a in agents],
            )

    def add_action(self, pid: str, agent_id: str, action_type: str, content: str,
                   round_num: int, target_id: str = ""):
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO mf_actions (prediction_id, agent_id, action_type, content, target_id, round_num, ts) VALUES (?,?,?,?,?,?,?)",
                (pid, agent_id, action_type, content, target_id, round_num, time.time()),
            )

    def get_timeline(self, pid: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT agent_id, action_type, content, target_id, round_num, ts FROM mf_actions WHERE prediction_id=? ORDER BY round_num, action_id",
                (pid,),
            ).fetchall()
        return [dict(r) for r in rows]
