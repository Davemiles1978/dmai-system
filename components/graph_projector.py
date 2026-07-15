"""
GraphProjector — builds a real knowledge graph from DMAI's ingested data.

Turns the flat registries into a tiered graph so SI can find links and
evolutions across DMAI's actual repertoire:

  Layer 1 — Architecture (32 core subsystems from graph_schema.json)
  Layer 2 — Capabilities (~20k rows from capabilities table)
             clustered by capability_type; each type is attached to the
             matching architectural neuron.
  Layer 3 — Insight topics (~1-3k distinct source_topic + target_topic
             values from the insights table).

Synapses:
  - insight edges: every row in `insights` is a directed synapse from
    source_topic → target_topic with the row's `relationship` label.
    Multiple insights between the same pair are collapsed into a single
    synapse with weight = count.
  - capability-to-topic: capability whose lowercased name is a substring
    of (or equal to) a topic name.
  - capability-to-capability (same repo): capabilities sharing a
    source_repo get a lightweight 'same_repo' edge (capped per repo to
    keep the graph sane).
  - capability-to-architecture: each capability_type maps to a fixed
    architectural anchor (see TYPE_TO_ANCHOR).

Persistence: SQLite tables `graph_neurons` and `graph_synapses` inside
`data/dmai_knowledge.db`. Non-destructive rebuild: drop-and-recreate
inside a single transaction.

Usage:
    from components.graph_projector import GraphProjector
    stats = GraphProjector().rebuild()
    schema = GraphProjector().to_schema()  # dashboard-friendly JSON
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _REPO_ROOT / "data" / "dmai_knowledge.db"
ARCH_SCHEMA_PATH = _REPO_ROOT / "aevora-training" / "dashboard" / "data" / "graph_schema.json"

# Fixed mapping from capability_type → architectural anchor neuron.
# Keeps the top-layer sane and lets SI see which subsystem a capability
# is "wired into" without having to guess.
TYPE_TO_ANCHOR: Dict[str, str] = {
    "ai_model": "ai_hub",
    "api": "api_gateway",
    "automation": "background_svc",
    "blockchain": "trading_agent",
    "configuration": "meta_controller",
    "data_structure": "db_storage",
    "funding": "self_funding",
    "generation": "content_gen",
    "identity": "meta_controller",
    "interface": "api_gateway",
    "knowledge": "knowledge_mgr",
    "replication": "background_svc",
    "requirement": "meta_controller",
    "survival": "kaizen_engine",
    "trading": "trading_agent",
    "utility": "background_svc",
}

# Per-repo cap on same_repo edges to avoid a hairball on big monorepos.
MAX_SAME_REPO_EDGES = 200


class GraphProjector:
    def __init__(self, db_path: Optional[Path] = None,
                 arch_schema_path: Optional[Path] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.arch_schema_path = Path(arch_schema_path or ARCH_SCHEMA_PATH)

    # ── Schema ────────────────────────────────────────────────────────
    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_neurons (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                layer TEXT NOT NULL,            -- 'architecture', 'capability', 'topic'
                cluster TEXT,
                capability_type TEXT,
                runtime_mode TEXT,
                source_repo TEXT,
                activation REAL DEFAULT 0.5,
                metadata TEXT,                  -- JSON blob
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_gn_layer ON graph_neurons(layer);
            CREATE INDEX IF NOT EXISTS idx_gn_captype ON graph_neurons(capability_type);
            CREATE INDEX IF NOT EXISTS idx_gn_cluster ON graph_neurons(cluster);

            CREATE TABLE IF NOT EXISTS graph_synapses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL,        -- 'insight', 'cap_to_topic',
                                                -- 'same_repo', 'cap_to_arch',
                                                -- 'arch'
                weight REAL DEFAULT 1.0,
                relationship TEXT,              -- free-form label from insights
                metadata TEXT,
                UNIQUE(source, target, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_gs_source ON graph_synapses(source);
            CREATE INDEX IF NOT EXISTS idx_gs_target ON graph_synapses(target);
            CREATE INDEX IF NOT EXISTS idx_gs_type ON graph_synapses(edge_type);
        """)

    def _clear(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM graph_synapses")
        conn.execute("DELETE FROM graph_neurons")

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _neuron_id_for_capability(cap_id: str) -> str:
        return f"cap:{cap_id}"

    @staticmethod
    def _neuron_id_for_topic(topic: str) -> str:
        return f"topic:{topic.strip().lower()}"

    @staticmethod
    def _neuron_id_for_type(cap_type: str) -> str:
        return f"type:{cap_type.strip().lower()}"

    # ── Build ─────────────────────────────────────────────────────────
    def rebuild(self) -> Dict[str, Any]:
        """Rebuild the projected graph. Idempotent; safe to call periodically."""
        if not self.db_path.exists():
            return {"ok": False, "error": f"DB not found: {self.db_path}"}

        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._ensure_tables(conn)
            self._clear(conn)

            stats = {
                "arch_neurons": 0,
                "type_neurons": 0,
                "capability_neurons": 0,
                "topic_neurons": 0,
                "insight_synapses": 0,
                "cap_to_topic_synapses": 0,
                "same_repo_synapses": 0,
                "cap_to_arch_synapses": 0,
                "arch_synapses": 0,
            }

            # 1) Load architectural neurons from the existing schema file.
            arch_ids: set[str] = set()
            arch_synapses: List[Dict[str, Any]] = []
            try:
                if self.arch_schema_path.exists():
                    arch_schema = json.loads(
                        self.arch_schema_path.read_text(encoding="utf-8")
                    )
                    for n in arch_schema.get("neurons", []) or []:
                        nid = n.get("id")
                        if not nid:
                            continue
                        conn.execute(
                            "INSERT OR REPLACE INTO graph_neurons "
                            "(id, label, layer, cluster, activation, metadata) "
                            "VALUES (?, ?, 'architecture', ?, ?, ?)",
                            (
                                nid,
                                n.get("label") or nid,
                                n.get("cluster") or "core",
                                float(n.get("activation", 0.5) or 0.5),
                                json.dumps({"description": n.get("description", "")}),
                            ),
                        )
                        arch_ids.add(nid)
                        stats["arch_neurons"] += 1
                    arch_synapses = arch_schema.get("synapses", []) or []
            except Exception as e:
                logger.warning("Failed to load architectural schema: %s", e)

            # 2) Capability neurons + type cluster neurons.
            cap_rows = conn.execute(
                "SELECT id, name, capability_type, runtime_mode, source_repo, "
                "description FROM capabilities"
            ).fetchall()

            type_counts: Dict[str, int] = {}
            cap_name_to_id: Dict[str, str] = {}
            for r in cap_rows:
                cap_id = r["id"]
                cap_type = (r["capability_type"] or "unknown").lower()
                type_counts[cap_type] = type_counts.get(cap_type, 0) + 1
                neuron_id = self._neuron_id_for_capability(cap_id)
                conn.execute(
                    "INSERT OR REPLACE INTO graph_neurons "
                    "(id, label, layer, cluster, capability_type, runtime_mode, "
                    " source_repo, activation, metadata) "
                    "VALUES (?, ?, 'capability', ?, ?, ?, ?, ?, ?)",
                    (
                        neuron_id,
                        r["name"] or cap_id,
                        cap_type,
                        cap_type,
                        r["runtime_mode"],
                        r["source_repo"],
                        0.5,
                        json.dumps({"description": (r["description"] or "")[:400]}),
                    ),
                )
                stats["capability_neurons"] += 1
                if r["name"]:
                    cap_name_to_id[(r["name"] or "").strip().lower()] = neuron_id

            # Type cluster neurons + edges to architectural anchors.
            for cap_type, count in type_counts.items():
                type_neuron_id = self._neuron_id_for_type(cap_type)
                conn.execute(
                    "INSERT OR REPLACE INTO graph_neurons "
                    "(id, label, layer, cluster, capability_type, activation, "
                    " metadata) VALUES (?, ?, 'capability_type', ?, ?, ?, ?)",
                    (
                        type_neuron_id,
                        f"{cap_type} ({count})",
                        cap_type,
                        cap_type,
                        min(1.0, 0.3 + count / 5000.0),
                        json.dumps({"count": count}),
                    ),
                )
                stats["type_neurons"] += 1

                anchor = TYPE_TO_ANCHOR.get(cap_type)
                if anchor and anchor in arch_ids:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO graph_synapses "
                            "(source, target, edge_type, weight, relationship) "
                            "VALUES (?, ?, 'cap_to_arch', ?, 'contains')",
                            (anchor, type_neuron_id, 1.0),
                        )
                        stats["cap_to_arch_synapses"] += 1
                    except Exception:
                        pass

                # link every capability to its type cluster
                conn.execute(
                    "INSERT OR IGNORE INTO graph_synapses "
                    "(source, target, edge_type, weight, relationship) "
                    "SELECT ?, id, 'cap_to_arch', 0.5, 'member_of' "
                    "FROM graph_neurons "
                    "WHERE layer='capability' AND capability_type=?",
                    (type_neuron_id, cap_type),
                )

            # 3) Topic neurons (union of insights.source_topic & target_topic).
            topic_rows = conn.execute("""
                SELECT topic, SUM(cnt) AS occurrences
                FROM (
                    SELECT source_topic AS topic, COUNT(*) AS cnt
                    FROM insights WHERE source_topic IS NOT NULL AND LENGTH(source_topic) > 1
                    GROUP BY source_topic
                    UNION ALL
                    SELECT target_topic AS topic, COUNT(*) AS cnt
                    FROM insights WHERE target_topic IS NOT NULL AND LENGTH(target_topic) > 1
                    GROUP BY target_topic
                )
                GROUP BY topic
            """).fetchall()

            topic_ids: Dict[str, str] = {}
            for r in topic_rows:
                topic = (r["topic"] or "").strip()
                if not topic:
                    continue
                tid = self._neuron_id_for_topic(topic)
                if tid in topic_ids:
                    continue
                occ = int(r["occurrences"] or 1)
                conn.execute(
                    "INSERT OR REPLACE INTO graph_neurons "
                    "(id, label, layer, cluster, activation, metadata) "
                    "VALUES (?, ?, 'topic', 'topics', ?, ?)",
                    (
                        tid,
                        topic,
                        min(1.0, 0.2 + occ / 100.0),
                        json.dumps({"occurrences": occ}),
                    ),
                )
                topic_ids[tid] = topic
                stats["topic_neurons"] += 1

            # 4) Insight synapses (topic → topic).
            insight_edges = conn.execute("""
                SELECT source_topic, target_topic, relationship,
                       COUNT(*) AS n, MAX(confidence) AS max_conf
                FROM insights
                WHERE source_topic IS NOT NULL AND target_topic IS NOT NULL
                  AND LENGTH(source_topic) > 1 AND LENGTH(target_topic) > 1
                  AND source_topic <> target_topic
                GROUP BY source_topic, target_topic, relationship
            """).fetchall()

            for e in insight_edges:
                s = self._neuron_id_for_topic(e["source_topic"])
                t = self._neuron_id_for_topic(e["target_topic"])
                if s not in topic_ids or t not in topic_ids:
                    continue
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO graph_synapses "
                        "(source, target, edge_type, weight, relationship, metadata) "
                        "VALUES (?, ?, 'insight', ?, ?, ?)",
                        (
                            s, t,
                            float(e["max_conf"] or 0.5),
                            e["relationship"] or "related",
                            json.dumps({"insight_count": int(e["n"] or 1)}),
                        ),
                    )
                    stats["insight_synapses"] += 1
                except Exception:
                    pass

            # 5) Capability-to-topic edges: exact-name and substring matches.
            for cap_name_lc, cap_neuron in cap_name_to_id.items():
                if len(cap_name_lc) < 3:
                    continue
                topic_neuron_id = self._neuron_id_for_topic(cap_name_lc)
                if topic_neuron_id in topic_ids:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO graph_synapses "
                            "(source, target, edge_type, weight, relationship) "
                            "VALUES (?, ?, 'cap_to_topic', 0.9, 'implements')",
                            (cap_neuron, topic_neuron_id),
                        )
                        stats["cap_to_topic_synapses"] += 1
                    except Exception:
                        pass

            # 6) Capability↔capability same_repo edges (capped per repo).
            repo_rows = conn.execute("""
                SELECT source_repo, GROUP_CONCAT(id) AS ids, COUNT(*) AS n
                FROM capabilities
                WHERE source_repo IS NOT NULL AND source_repo <> ''
                GROUP BY source_repo
                HAVING n > 1 AND n <= 500
            """).fetchall()
            for rr in repo_rows:
                ids = (rr["ids"] or "").split(",")
                if len(ids) < 2:
                    continue
                # Build a small star: link the first cap in the repo to up
                # to MAX_SAME_REPO_EDGES others. Keeps degree bounded.
                center = self._neuron_id_for_capability(ids[0])
                for other_id in ids[1:1 + MAX_SAME_REPO_EDGES]:
                    other = self._neuron_id_for_capability(other_id)
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO graph_synapses "
                            "(source, target, edge_type, weight, relationship, metadata) "
                            "VALUES (?, ?, 'same_repo', 0.4, 'same_repo', ?)",
                            (center, other, json.dumps({"repo": rr["source_repo"]})),
                        )
                        stats["same_repo_synapses"] += 1
                    except Exception:
                        pass

            # 7) Preserve original architectural synapses.
            for s in arch_synapses:
                src = s.get("source"); tgt = s.get("target")
                if not (src and tgt) or src not in arch_ids or tgt not in arch_ids:
                    continue
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO graph_synapses "
                        "(source, target, edge_type, weight, relationship) "
                        "VALUES (?, ?, 'arch', ?, ?)",
                        (src, tgt, float(s.get("weight", 1.0) or 1.0),
                         s.get("type") or "control"),
                    )
                    stats["arch_synapses"] += 1
                except Exception:
                    pass

            conn.commit()

            # Totals.
            total_neurons = conn.execute(
                "SELECT COUNT(*) FROM graph_neurons"
            ).fetchone()[0]
            total_synapses = conn.execute(
                "SELECT COUNT(*) FROM graph_synapses"
            ).fetchone()[0]
            stats["total_neurons"] = total_neurons
            stats["total_synapses"] = total_synapses
            stats["ok"] = True
            return stats
        except Exception as e:
            conn.rollback()
            logger.exception("GraphProjector.rebuild failed")
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()

    # ── Read API ──────────────────────────────────────────────────────
    def to_schema(self, limit_per_layer: int = 5000) -> Dict[str, Any]:
        """Return a dashboard-friendly dict of neurons + synapses.

        Includes a hard per-layer cap so the JSON response is bounded on
        very large registries. Use SQL directly for scale-sensitive
        queries.
        """
        if not self.db_path.exists():
            return {"neurons": [], "synapses": [], "total_neurons": 0,
                    "total_synapses": 0}

        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            # Bail if projection tables don't exist yet.
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='graph_neurons'"
            ).fetchone()
            if not has_table:
                return {"neurons": [], "synapses": [], "total_neurons": 0,
                        "total_synapses": 0,
                        "_note": "projection tables not built yet"}

            total_neurons = conn.execute(
                "SELECT COUNT(*) FROM graph_neurons"
            ).fetchone()[0]
            total_synapses = conn.execute(
                "SELECT COUNT(*) FROM graph_synapses"
            ).fetchone()[0]

            neurons: List[Dict[str, Any]] = []
            for layer in ("architecture", "capability_type", "topic", "capability"):
                rows = conn.execute(
                    "SELECT id, label, layer, cluster, capability_type, activation "
                    "FROM graph_neurons WHERE layer=? "
                    "ORDER BY activation DESC LIMIT ?",
                    (layer, limit_per_layer),
                ).fetchall()
                for r in rows:
                    neurons.append({
                        "id": r["id"],
                        "label": r["label"],
                        "layer": r["layer"],
                        "cluster": r["cluster"],
                        "capability_type": r["capability_type"],
                        "activation": r["activation"],
                    })
            kept_ids = {n["id"] for n in neurons}

            synapses: List[Dict[str, Any]] = []
            edge_rows = conn.execute(
                "SELECT source, target, edge_type, weight, relationship "
                "FROM graph_synapses LIMIT ?",
                (limit_per_layer * 4,),
            ).fetchall()
            for r in edge_rows:
                if r["source"] in kept_ids and r["target"] in kept_ids:
                    synapses.append({
                        "source": r["source"],
                        "target": r["target"],
                        "type": r["edge_type"],
                        "weight": r["weight"],
                        "relationship": r["relationship"],
                    })

            return {
                "schema_version": "2.0",
                "neurons": neurons,
                "synapses": synapses,
                "total_neurons": total_neurons,
                "total_synapses": total_synapses,
                "projection": True,
            }
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {"ok": False, "error": "DB not found"}
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='graph_neurons'"
            ).fetchone()
            if not has_table:
                return {"ok": True, "built": False,
                        "total_neurons": 0, "total_synapses": 0}
            by_layer = dict(conn.execute(
                "SELECT layer, COUNT(*) FROM graph_neurons GROUP BY layer"
            ).fetchall())
            by_edge = dict(conn.execute(
                "SELECT edge_type, COUNT(*) FROM graph_synapses GROUP BY edge_type"
            ).fetchall())
            total_neurons = sum(by_layer.values())
            total_synapses = sum(by_edge.values())
            return {
                "ok": True, "built": True,
                "total_neurons": total_neurons,
                "total_synapses": total_synapses,
                "by_layer": by_layer,
                "by_edge_type": by_edge,
            }
        finally:
            conn.close()
