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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Prefer DMAI's shared connection proxy so we honour the shared write
    # mutex + long busy_timeout used across the codebase. Falls back to a
    # raw sqlite3.connect for the unit tests, which supply a tmp DB.
    from components.db import safe_open_kdb as _safe_open_kdb  # type: ignore
except Exception:  # pragma: no cover
    _safe_open_kdb = None  # type: ignore

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _REPO_ROOT / "data" / "dmai_knowledge.db"
ARCH_SCHEMA_PATH = _REPO_ROOT / "data" / "graph_schema.json"

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

    # ── Connection helpers ────────────────────────────────────────────
    def _open_conn(self, timeout: float = 30.0):
        """Return a connection that respects DMAI's shared write mutex.

        Uses safe_open_kdb from components.db when available (i.e. when
        running inside the live app). Falls back to a raw sqlite3
        connection for unit tests, which use tmp_path databases and
        don't need the process-wide mutex.
        """
        if _safe_open_kdb is not None:
            try:
                return _safe_open_kdb(str(self.db_path), timeout=timeout)
            except Exception as e:
                logger.warning(
                    "safe_open_kdb failed, falling back to raw sqlite3: %s", e
                )
        conn = safe_open_kdb(str(self.db_path), timeout=timeout)
        return conn

    def _acquire_write_slot(self, conn, *, attempts: int = 5,
                            initial_delay: float = 0.5) -> None:
        """Force acquisition of the SQLite write lock with backoff.

        Immediately BEGIN IMMEDIATE — if the DB is locked we back off
        with exponential delays instead of hitting an unbounded wait
        inside the huge bulk-clear DELETE. Fails loud after N attempts
        so the caller can report a friendly error rather than a 30s
        hang.
        """
        delay = initial_delay
        for i in range(attempts):
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as e:
                if "lock" not in str(e).lower():
                    raise
                if i == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

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

        conn = self._open_conn(timeout=90.0)
        # sqlite3.Row for both proxy and raw paths.
        try:
            underlying = getattr(conn, "_conn", conn)
            underlying.row_factory = sqlite3.Row
        except Exception:
            pass
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=90000")
            self._ensure_tables(conn)
            self._acquire_write_slot(conn)
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

            # 3) Topic neurons — union of every populated topic-like column.
            #
            # Reality on prod: the InsightPromoter (the majority writer)
            # only fills (concept, insight_text, confidence, domain,
            # source), leaving source_topic/target_topic NULL. Other
            # writers fill source_topic/target_topic. So we build the
            # topic set from four columns and treat 'concept' as the
            # primary topic when it exists.
            #
            # Detect which columns exist so we don't SELECT a missing
            # column and blow up on legacy DBs.
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(insights)").fetchall()
            }
            topic_selects: list[str] = []
            for col in ("source_topic", "target_topic", "concept", "domain"):
                if col in cols:
                    topic_selects.append(
                        f"SELECT {col} AS topic, COUNT(*) AS cnt "
                        f"FROM insights WHERE {col} IS NOT NULL "
                        f"AND LENGTH({col}) > 1 GROUP BY {col}"
                    )
            if topic_selects:
                topic_sql = (
                    "SELECT topic, SUM(cnt) AS occurrences FROM (\n"
                    + "\nUNION ALL\n".join(topic_selects)
                    + "\n) GROUP BY topic"
                )
                topic_rows = conn.execute(topic_sql).fetchall()
            else:
                topic_rows = []

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
            #
            # Two paths:
            #   (a) legacy: source_topic → target_topic + relationship
            #   (b) main promoter path: concept → domain (both fields
            #       populated on every promoter insert). We treat that
            #       as a 'concept_in_domain' relationship.
            insight_edges: list[dict] = []
            if {"source_topic", "target_topic", "relationship"} <= cols:
                for r in conn.execute("""
                    SELECT source_topic AS s, target_topic AS t,
                           relationship AS rel,
                           COUNT(*) AS n, MAX(confidence) AS max_conf
                    FROM insights
                    WHERE source_topic IS NOT NULL AND target_topic IS NOT NULL
                      AND LENGTH(source_topic) > 1 AND LENGTH(target_topic) > 1
                      AND source_topic <> target_topic
                    GROUP BY source_topic, target_topic, relationship
                """).fetchall():
                    insight_edges.append({
                        "s": r["s"], "t": r["t"],
                        "rel": r["rel"], "n": r["n"], "max_conf": r["max_conf"],
                    })
            if {"concept", "domain"} <= cols:
                for r in conn.execute("""
                    SELECT concept AS s, domain AS t,
                           COUNT(*) AS n, MAX(confidence) AS max_conf
                    FROM insights
                    WHERE concept IS NOT NULL AND domain IS NOT NULL
                      AND LENGTH(concept) > 1 AND LENGTH(domain) > 1
                      AND LOWER(concept) <> LOWER(domain)
                    GROUP BY concept, domain
                """).fetchall():
                    insight_edges.append({
                        "s": r["s"], "t": r["t"],
                        "rel": "concept_in_domain", "n": r["n"],
                        "max_conf": r["max_conf"],
                    })

            for e in insight_edges:
                s = self._neuron_id_for_topic(e["s"])
                t = self._neuron_id_for_topic(e["t"])
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
                            e["rel"] or "related",
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
            try:
                conn.rollback()
            except Exception:
                pass
            logger.exception("GraphProjector.rebuild failed")
            return {"ok": False, "error": str(e)}
        finally:
            try:
                conn.close()
            except Exception:
                pass

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

        conn = self._open_conn(timeout=30.0)
        try:
            underlying = getattr(conn, "_conn", conn)
            underlying.row_factory = sqlite3.Row
        except Exception:
            pass
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

    def drilldown(
        self,
        view: str = "overview",
        expand_type: Optional[str] = None,
        expand_cap: Optional[str] = None,
        expand_topic: Optional[str] = None,
        limit: int = 60,
    ) -> Dict[str, Any]:
        """Return a bounded, layered slice of the graph for drilldown UIs.

        Views:
          - ``overview``: architecture neurons + capability_type neurons
            + top-N topics by degree. This is the default readable view.
          - ``type``: architecture + all type neurons + top ``limit``
            capabilities of ``expand_type`` (by activation).
          - ``capability``: the requested capability + its
            immediate neighbours (type parent + linked topics + same-repo
            peers, all bounded by ``limit``).
          - ``topic``: the requested topic + capabilities linked to it +
            other topics one hop away.

        Returns the same shape as ``to_schema`` (neurons + synapses +
        totals) so the frontend renderer can consume both uniformly.
        """
        if not self.db_path.exists():
            return {"neurons": [], "synapses": [], "total_neurons": 0,
                    "total_synapses": 0, "view": view}

        conn = self._open_conn(timeout=30.0)
        try:
            underlying = getattr(conn, "_conn", conn)
            underlying.row_factory = sqlite3.Row
        except Exception:
            pass

        try:
            has_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='graph_neurons'"
            ).fetchone()
            if not has_table:
                return {"neurons": [], "synapses": [], "total_neurons": 0,
                        "total_synapses": 0, "view": view,
                        "_note": "projection tables not built yet"}

            total_neurons = conn.execute(
                "SELECT COUNT(*) FROM graph_neurons"
            ).fetchone()[0]
            total_synapses = conn.execute(
                "SELECT COUNT(*) FROM graph_synapses"
            ).fetchone()[0]

            def _pack(row: sqlite3.Row) -> Dict[str, Any]:
                return {
                    "id": row["id"],
                    "label": row["label"],
                    "layer": row["layer"],
                    "cluster": row["cluster"],
                    "capability_type": row["capability_type"],
                    "activation": row["activation"],
                    "has_children": False,  # set below per-view
                    "child_count": 0,
                }

            neurons: Dict[str, Dict[str, Any]] = {}
            counts_by_layer: Dict[str, int] = {}
            for row in conn.execute(
                "SELECT layer, COUNT(*) AS n FROM graph_neurons GROUP BY layer"
            ).fetchall():
                counts_by_layer[row["layer"]] = int(row["n"])

            def _include(row: sqlite3.Row) -> None:
                if row["id"] not in neurons:
                    neurons[row["id"]] = _pack(row)

            # 1) Every view includes architecture + capability_type as the
            # legible top layers.
            for row in conn.execute(
                "SELECT id, label, layer, cluster, capability_type, activation "
                "FROM graph_neurons WHERE layer IN ('architecture','capability_type') "
                "ORDER BY layer, activation DESC"
            ).fetchall():
                _include(row)

            # 2) View-specific expansion.
            if view == "overview":
                # Add top topics by degree so the overview has some
                # topical texture without dumping 20k caps into the DOM.
                for row in conn.execute(
                    """SELECT n.id, n.label, n.layer, n.cluster,
                              n.capability_type, n.activation
                       FROM graph_neurons n
                       LEFT JOIN (
                         SELECT source AS nid, COUNT(*) AS d FROM graph_synapses GROUP BY source
                         UNION ALL
                         SELECT target AS nid, COUNT(*) AS d FROM graph_synapses GROUP BY target
                       ) e ON e.nid = n.id
                       WHERE n.layer='topic'
                       GROUP BY n.id
                       ORDER BY COALESCE(SUM(e.d), 0) DESC, n.activation DESC
                       LIMIT ?""",
                    (max(20, min(limit, 200)),),
                ).fetchall():
                    _include(row)

            elif view == "type" and expand_type:
                for row in conn.execute(
                    "SELECT id, label, layer, cluster, capability_type, activation "
                    "FROM graph_neurons WHERE layer='capability' "
                    "AND capability_type=? ORDER BY activation DESC LIMIT ?",
                    (expand_type, max(10, min(limit, 500))),
                ).fetchall():
                    _include(row)

            elif view == "capability" and expand_cap:
                target_row = conn.execute(
                    "SELECT id, label, layer, cluster, capability_type, activation "
                    "FROM graph_neurons WHERE id=?",
                    (expand_cap,),
                ).fetchone()
                if target_row:
                    _include(target_row)
                    # Neighbours via synapses.
                    for row in conn.execute(
                        """SELECT n.id, n.label, n.layer, n.cluster,
                                  n.capability_type, n.activation
                           FROM graph_synapses s
                           JOIN graph_neurons n
                             ON n.id = CASE WHEN s.source=? THEN s.target ELSE s.source END
                           WHERE s.source=? OR s.target=?
                           ORDER BY s.weight DESC
                           LIMIT ?""",
                        (expand_cap, expand_cap, expand_cap,
                         max(10, min(limit, 200))),
                    ).fetchall():
                        _include(row)

            elif view == "topic" and expand_topic:
                target_row = conn.execute(
                    "SELECT id, label, layer, cluster, capability_type, activation "
                    "FROM graph_neurons WHERE id=?",
                    (expand_topic,),
                ).fetchone()
                if target_row:
                    _include(target_row)
                    for row in conn.execute(
                        """SELECT n.id, n.label, n.layer, n.cluster,
                                  n.capability_type, n.activation
                           FROM graph_synapses s
                           JOIN graph_neurons n
                             ON n.id = CASE WHEN s.source=? THEN s.target ELSE s.source END
                           WHERE s.source=? OR s.target=?
                           ORDER BY s.weight DESC
                           LIMIT ?""",
                        (expand_topic, expand_topic, expand_topic,
                         max(10, min(limit, 200))),
                    ).fetchall():
                        _include(row)

            # 3) Compute child counts for every visible neuron so the UI
            # can show '+N' badges (has_children indicator).
            visible_ids = list(neurons.keys())
            if visible_ids:
                # For architecture neurons: count capability_type children linked via cap_to_arch.
                for row in conn.execute(
                    "SELECT source AS arch, COUNT(DISTINCT target) AS n "
                    "FROM graph_synapses WHERE edge_type='cap_to_arch' "
                    "GROUP BY source"
                ).fetchall():
                    # arch -> capability edges reversed elsewhere; keep it simple.
                    pass
                # For capability_type neurons: count capabilities of that type.
                for row in conn.execute(
                    "SELECT capability_type, COUNT(*) AS n "
                    "FROM graph_neurons WHERE layer='capability' "
                    "GROUP BY capability_type"
                ).fetchall():
                    type_id = f"type:{row['capability_type']}"
                    if type_id in neurons:
                        neurons[type_id]["has_children"] = True
                        neurons[type_id]["child_count"] = int(row["n"])
                # For capability neurons: 'has_children' = degree > 0.
                deg_rows = conn.execute(
                    "SELECT nid, COUNT(*) AS d FROM ("
                    "  SELECT source AS nid FROM graph_synapses UNION ALL"
                    "  SELECT target AS nid FROM graph_synapses"
                    ") GROUP BY nid"
                ).fetchall()
                deg_by_id = {r["nid"]: int(r["d"]) for r in deg_rows}
                for nid, n in neurons.items():
                    if n["layer"] in ("capability", "topic") and deg_by_id.get(nid, 0) > 0:
                        n["has_children"] = True
                        n["child_count"] = deg_by_id[nid]

            # 4) Synapses between visible neurons only.
            visible_set = set(neurons.keys())
            synapses: List[Dict[str, Any]] = []
            if visible_set:
                placeholders = ",".join("?" for _ in visible_set)
                for row in conn.execute(
                    f"SELECT source, target, edge_type, weight, relationship "
                    f"FROM graph_synapses "
                    f"WHERE source IN ({placeholders}) AND target IN ({placeholders}) "
                    f"LIMIT ?",
                    (*visible_set, *visible_set, len(visible_set) * 6),
                ).fetchall():
                    synapses.append({
                        "source": row["source"],
                        "target": row["target"],
                        "type": row["edge_type"],
                        "weight": row["weight"],
                        "relationship": row["relationship"],
                    })

            return {
                "schema_version": "2.1",
                "view": view,
                "expand_type": expand_type,
                "expand_cap": expand_cap,
                "expand_topic": expand_topic,
                "neurons": list(neurons.values()),
                "synapses": synapses,
                "visible_neurons": len(neurons),
                "visible_synapses": len(synapses),
                "total_neurons": int(total_neurons),
                "total_synapses": int(total_synapses),
                "counts_by_layer": counts_by_layer,
                "projection": True,
            }
        finally:
            conn.close()

    def stats(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {"ok": False, "error": "DB not found"}
        conn = self._open_conn(timeout=30.0)
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
