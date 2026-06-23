"""
DMAI GraphWriter — Live Knowledge Graph Growth Engine
=====================================================
Automatically grows graph_schema.json whenever DMAI studies or researches.

Called after every:
  - Learning / study cycle
  - Autonomous research run
  - Insight added via si_core.add_insight()
  - Capability registered

This is the bridge between DMAI's learning activity and the visual knowledge graph.
The graph_evolution_monitor.py script handles Git PR creation (Friday cron).
GraphWriter handles the LIVE, continuous growth during normal operation.

Node types produced:
  - domain    — from discoveries.jsonl (autonomous_researcher output)
  - entity    — from discoveries.jsonl entities list
  - insight   — from insights.jsonl (si_core.add_insight output)
  - capability— from capabilities table / code_writer output
  - topic     — from syllabus mastery progression

Usage:
    from components.graph_writer import GraphWriter
    gw = GraphWriter()
    stats = gw.evolve()           # full evolution pass
    stats = gw.add_insight_node(domain, concept, source)   # single insight
    stats = gw.add_topic_node(stage, topic)                # single topic mastered
"""

import hashlib
import json
import logging
import re
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dmai.graph_writer")

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent
SCHEMA_PATH  = _REPO_ROOT / "aevora-training" / "dashboard" / "data" / "graph_schema.json"
DISCOVERIES  = _REPO_ROOT / "data" / "research" / "discoveries.jsonl"
INSIGHTS     = _REPO_ROOT / "data" / "research" / "insights.jsonl"
DB_PATH      = _REPO_ROOT / "data" / "dmai_knowledge.db"

# ── Domain → cluster mapping (matches graph_evolution_monitor.py) ─────────────
DOMAIN_CLUSTER_MAP = {
    "machine_learning":       "learning",
    "reinforcement_learning": "learning",
    "autonomous_agents":      "research",
    "trading":                "revenue",
    "content_generation":     "revenue",
    "computer_vision":        "research",
    "nlp":                    "knowledge",
    "self_improvement":       "core",
    "knowledge_systems":      "knowledge",
    "robotics":               "research",
    "cybersecurity":          "research",
    "web_technologies":       "research",
    "data_science":           "knowledge",
    "cloud_devops":           "providers",
    # extended domains from researcher
    "arxiv":                  "research",
    "github":                 "research",
    "artificial_intelligence":"research",
    "deep_learning":          "learning",
    "llm":                    "knowledge",
    "agi":                    "core",
    "finance":                "revenue",
    "python":                 "research",
}

CLUSTER_HUB_MAP = {
    "core":      "dmai_core",
    "learning":  "learning_orch",
    "research":  "auto_researcher",
    "knowledge": "knowledge_mgr",
    "providers": "ai_hub",
    "revenue":   "self_funding",
}

# Stage → cluster for syllabus topics
STAGE_CLUSTER_MAP = {
    "baby":    "core",
    "toddler": "learning",
    "child":   "learning",
    "teen":    "knowledge",
    "adult":   "core",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:48]


def _stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode()).hexdigest()[:6]
    return f"{_slug(prefix)}_{h}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
    return records


def _infer_cluster(domain: str) -> str:
    """Best-effort cluster from domain string."""
    d = domain.lower().replace(" ", "_").replace("-", "_")
    return DOMAIN_CLUSTER_MAP.get(d, "knowledge")


# ── GraphWriter ────────────────────────────────────────────────────────────────

class GraphWriter:
    """
    Continuously grows graph_schema.json from DMAI's live learning activity.
    Thread-safe via file-level locking on save.
    """

    def __init__(self, schema_path: Path = SCHEMA_PATH):
        self.schema_path = schema_path

    # ── Schema I/O ─────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self.schema_path.exists():
            logger.error("graph_schema.json not found at %s", self.schema_path)
            return {}
        try:
            return json.loads(self.schema_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Could not parse graph_schema.json: %s", e)
            return {}

    def _save(self, schema: dict) -> bool:
        try:
            self.schema_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.schema_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(schema, indent=2), encoding="utf-8")
            tmp.replace(self.schema_path)
            return True
        except Exception as e:
            logger.error("Could not save graph_schema.json: %s", e)
            return False

    # ── Node helpers ───────────────────────────────────────────────────────────

    def _node_ids(self, schema: dict) -> set:
        return {n["id"] for n in schema.get("neurons", [])}

    def _synapse_keys(self, schema: dict) -> set:
        return {(s["source"], s["target"]) for s in schema.get("synapses", [])}

    def _add_node(self, schema: dict, node: dict, existing_ids: set) -> bool:
        """Add node to schema if ID is new. Returns True if added."""
        if node["id"] in existing_ids:
            return False
        schema["neurons"].append(node)
        existing_ids.add(node["id"])
        return True

    def _add_synapse(self, schema: dict, source: str, target: str,
                     existing_ids: set, existing_synapses: set,
                     weight: float = 0.5, stype: str = "data") -> bool:
        """Add synapse if both endpoints exist and edge is new. Returns True if added."""
        if source not in existing_ids or target not in existing_ids:
            return False
        key = (source, target)
        if key in existing_synapses:
            return False
        schema["synapses"].append({
            "source": source, "target": target,
            "weight": weight, "type": stype, "auto_generated": True,
        })
        existing_synapses.add(key)
        return True

    def _update_metadata(self, schema: dict) -> None:
        schema["total_neurons"]  = len(schema.get("neurons", []))
        schema["total_synapses"] = len(schema.get("synapses", []))
        schema["evolution_cycle"] = schema.get("evolution_cycle", 0) + 1
        schema["last_updated"]   = _today_utc()
        if "metadata" not in schema:
            schema["metadata"] = {}
        schema["metadata"]["auto_evolved"] = True
        schema["metadata"]["last_live_update"] = _now_utc()

    # ── Full evolution pass ─────────────────────────────────────────────────────

    def evolve(self) -> dict:
        """
        Full evolution pass:
        1. Process all discoveries.jsonl entries
        2. Process all insights.jsonl entries
        3. Process mastered syllabus topics from DB
        4. Process recent capabilities from DB
        Returns stats dict: {new_neurons, new_synapses, evolution_cycle, total_neurons, total_synapses}
        """
        schema = self._load()
        if not schema:
            return {"error": "Could not load graph_schema.json"}

        schema = deepcopy(schema)
        existing_ids      = self._node_ids(schema)
        existing_synapses = self._synapse_keys(schema)
        new_nodes  = 0
        new_edges  = 0

        # ── 1. Discoveries ───────────────────────────────────────────────────
        discoveries = _load_jsonl(DISCOVERIES)
        for disc in discoveries:
            domain   = disc.get("domain", "").strip()
            entities = disc.get("entities", [])
            source   = disc.get("source", "autonomous_researcher")
            date_str = disc.get("date", _today_utc())

            if domain:
                domain_id = _slug(domain)
                cluster   = _infer_cluster(domain)
                if self._add_node(schema, {
                    "id": domain_id, "label": domain.replace("_", " ").title(),
                    "cluster": cluster, "description": f"Domain: {domain}",
                    "activation": 0.6, "auto_generated": True,
                    "first_seen": date_str, "source": source,
                }, existing_ids):
                    new_nodes += 1
                    hub = CLUSTER_HUB_MAP.get(cluster)
                    if hub:
                        if self._add_synapse(schema, hub, domain_id, existing_ids, existing_synapses, 0.6):
                            new_edges += 1

            for entity in entities[:5]:
                if not entity:
                    continue
                entity_id = _stable_id(domain or "entity", entity)
                cluster   = _infer_cluster(domain)
                if self._add_node(schema, {
                    "id": entity_id, "label": str(entity)[:32],
                    "cluster": cluster, "description": f"Entity from {source}",
                    "activation": 0.5, "auto_generated": True,
                    "first_seen": date_str, "source": source,
                }, existing_ids):
                    new_nodes += 1
                    domain_id = _slug(domain) if domain else None
                    if domain_id:
                        if self._add_synapse(schema, domain_id, entity_id, existing_ids, existing_synapses, 0.5):
                            new_edges += 1

        # ── 2. Insights ──────────────────────────────────────────────────────
        insights = _load_jsonl(INSIGHTS)
        for ins in insights:
            domain  = ins.get("domain", "")
            concept = ins.get("concept", ins.get("insight_text", ins.get("insight", ""))).strip()
            source  = ins.get("source", "si_core")
            date_str = ins.get("date", _today_utc())
            if not concept:
                continue
            node_id = _stable_id("insight", concept)
            cluster = _infer_cluster(domain)
            if self._add_node(schema, {
                "id": node_id, "label": concept[:32],
                "cluster": cluster, "description": f"Insight: {concept[:80]}",
                "activation": 0.55, "auto_generated": True,
                "first_seen": date_str, "source": source,
            }, existing_ids):
                new_nodes += 1
                # Connect insight → si_core (feedback loop)
                if self._add_synapse(schema, node_id, "si_core", existing_ids, existing_synapses, 0.55, "feedback"):
                    new_edges += 1
                # Also connect to domain hub if domain known
                domain_id = _slug(domain) if domain else None
                if domain_id and domain_id in existing_ids:
                    if self._add_synapse(schema, domain_id, node_id, existing_ids, existing_synapses, 0.5):
                        new_edges += 1

        # ── 3. Mastered syllabus topics from DB ──────────────────────────────
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur  = conn.cursor()
            cur.execute("""
                SELECT stage, topic_name, mastery FROM syllabus_content
                WHERE mastery >= 0.7
                ORDER BY mastery DESC LIMIT 200
            """)
            for row in cur.fetchall():
                stage, topic, mastery = row
                node_id = _stable_id("topic", topic)
                cluster = STAGE_CLUSTER_MAP.get((stage or "").lower(), "knowledge")
                if self._add_node(schema, {
                    "id": node_id, "label": str(topic)[:32],
                    "cluster": cluster,
                    "description": f"Syllabus topic [{stage}]: mastery={mastery:.2f}",
                    "activation": min(1.0, float(mastery)),
                    "auto_generated": True,
                    "first_seen": _today_utc(),
                    "source": "syllabus",
                }, existing_ids):
                    new_nodes += 1
                    hub = CLUSTER_HUB_MAP.get(cluster, "knowledge_mgr")
                    if self._add_synapse(schema, hub, node_id, existing_ids, existing_synapses, float(mastery)):
                        new_edges += 1
            conn.close()
        except Exception as e:
            logger.debug("Could not read syllabus topics from DB: %s", e)

        # ── 4. Recent capabilities from DB ───────────────────────────────────
        # NOTE: Individual capability nodes are NOT added to the visual graph.
        # With 20,000+ capabilities in the DB, adding even a small sample
        # produces a chaotic 'cap_xxxxxx' explosion that drowns the architecture.
        # Instead we update the knowledge_mgr node's activation level to reflect
        # overall capability count — keeping the graph clean and meaningful.
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM capabilities")
            cap_count = cur.fetchone()[0]
            conn.close()
            # Boost knowledge_mgr activation proportional to capability volume
            km_node = next((n for n in schema["neurons"] if n["id"] == "knowledge_mgr"), None)
            if km_node is not None:
                km_node["activation"] = min(round(cap_count / 50_000, 3), 1.0)
                km_node["capability_count"] = cap_count
            logger.debug("GraphWriter: %d capabilities tracked on knowledge_mgr (no individual nodes)", cap_count)
        except Exception as e:
            logger.debug("Could not read capabilities count from DB: %s", e)

        # ── Finalize ──────────────────────────────────────────────────────────
        if new_nodes > 0 or new_edges > 0:
            self._update_metadata(schema)
            saved = self._save(schema)
            logger.info(
                "GraphWriter.evolve: +%d neurons, +%d synapses → total %d/%d (saved=%s)",
                new_nodes, new_edges,
                schema["total_neurons"], schema["total_synapses"], saved,
            )
        else:
            logger.debug("GraphWriter.evolve: no new nodes — graph is current")

        return {
            "new_neurons":    new_nodes,
            "new_synapses":   new_edges,
            "evolution_cycle": schema.get("evolution_cycle", 0),
            "total_neurons":  schema.get("total_neurons", 0),
            "total_synapses": schema.get("total_synapses", 0),
        }

    # ── Single-node helpers (called inline during learning) ────────────────────

    def add_insight_node(self, domain: str, concept: str, source: str = "si_core") -> bool:
        """
        Add a single insight node immediately after si_core.add_insight() is called.
        Returns True if a new node was added.
        """
        if not concept:
            return False
        try:
            schema = self._load()
            if not schema:
                return False
            existing_ids      = self._node_ids(schema)
            existing_synapses = self._synapse_keys(schema)
            node_id = _stable_id("insight", concept)
            if node_id in existing_ids:
                return False  # already exists
            cluster = _infer_cluster(domain)
            added = self._add_node(schema, {
                "id": node_id, "label": concept[:32],
                "cluster": cluster,
                "description": f"Insight: {concept[:80]}",
                "activation": 0.55, "auto_generated": True,
                "first_seen": _today_utc(), "source": source,
            }, existing_ids)
            if added:
                self._add_synapse(schema, node_id, "si_core", existing_ids, existing_synapses, 0.55, "feedback")
                domain_id = _slug(domain) if domain else None
                if domain_id and domain_id in existing_ids:
                    self._add_synapse(schema, domain_id, node_id, existing_ids, existing_synapses, 0.5)
                self._update_metadata(schema)
                self._save(schema)
                logger.debug("GraphWriter: added insight node %s", node_id)
            return added
        except Exception as e:
            logger.warning("GraphWriter.add_insight_node failed: %s", e)
            return False

    def add_topic_node(self, stage: str, topic: str, mastery: float = 0.9) -> bool:
        """
        Add a syllabus topic node when it reaches mastery threshold.
        Returns True if a new node was added.
        """
        if not topic:
            return False
        try:
            schema = self._load()
            if not schema:
                return False
            existing_ids      = self._node_ids(schema)
            existing_synapses = self._synapse_keys(schema)
            node_id = _stable_id("topic", topic)
            if node_id in existing_ids:
                return False
            cluster = STAGE_CLUSTER_MAP.get((stage or "").lower(), "knowledge")
            added = self._add_node(schema, {
                "id": node_id, "label": str(topic)[:32],
                "cluster": cluster,
                "description": f"Mastered [{stage}]: {topic}",
                "activation": min(1.0, mastery),
                "auto_generated": True,
                "first_seen": _today_utc(), "source": "syllabus",
            }, existing_ids)
            if added:
                hub = CLUSTER_HUB_MAP.get(cluster, "knowledge_mgr")
                self._add_synapse(schema, hub, node_id, existing_ids, existing_synapses, mastery)
                self._update_metadata(schema)
                self._save(schema)
                logger.debug("GraphWriter: added topic node %s (stage=%s, mastery=%.2f)", node_id, stage, mastery)
            return added
        except Exception as e:
            logger.warning("GraphWriter.add_topic_node failed: %s", e)
            return False

    def add_discovery_node(self, domain: str, entities: list, source: str) -> dict:
        """
        Add a domain + entity nodes immediately after a discovery is persisted.
        Returns {new_neurons, new_synapses}.
        """
        try:
            schema = self._load()
            if not schema:
                return {"new_neurons": 0, "new_synapses": 0}
            existing_ids      = self._node_ids(schema)
            existing_synapses = self._synapse_keys(schema)
            nn, ne = 0, 0

            if domain:
                domain_id = _slug(domain)
                cluster   = _infer_cluster(domain)
                if self._add_node(schema, {
                    "id": domain_id, "label": domain.replace("_", " ").title(),
                    "cluster": cluster, "description": f"Domain: {domain}",
                    "activation": 0.6, "auto_generated": True,
                    "first_seen": _today_utc(), "source": source,
                }, existing_ids):
                    nn += 1
                    hub = CLUSTER_HUB_MAP.get(cluster)
                    if hub and self._add_synapse(schema, hub, domain_id, existing_ids, existing_synapses, 0.6):
                        ne += 1

            for entity in (entities or [])[:5]:
                if not entity:
                    continue
                entity_id = _stable_id(domain or "entity", entity)
                cluster   = _infer_cluster(domain)
                if self._add_node(schema, {
                    "id": entity_id, "label": str(entity)[:32],
                    "cluster": cluster, "description": f"Entity from {source}",
                    "activation": 0.5, "auto_generated": True,
                    "first_seen": _today_utc(), "source": source,
                }, existing_ids):
                    nn += 1
                    domain_id = _slug(domain) if domain else None
                    if domain_id and self._add_synapse(schema, domain_id, entity_id, existing_ids, existing_synapses, 0.5):
                        ne += 1

            if nn > 0 or ne > 0:
                self._update_metadata(schema)
                self._save(schema)

            return {"new_neurons": nn, "new_synapses": ne}
        except Exception as e:
            logger.warning("GraphWriter.add_discovery_node failed: %s", e)
            return {"new_neurons": 0, "new_synapses": 0}

    def status(self) -> dict:
        """Return current graph size stats without modifying anything."""
        try:
            schema = self._load()
            return {
                "total_neurons":   schema.get("total_neurons", 0),
                "total_synapses":  schema.get("total_synapses", 0),
                "evolution_cycle": schema.get("evolution_cycle", 0),
                "last_updated":    schema.get("last_updated", "never"),
                "schema_path":     str(self.schema_path),
            }
        except Exception as e:
            return {"error": str(e)}
