"""
DMAI GraphWriter — Live Knowledge Graph Activation Engine
=========================================================
Keeps graph_schema.json in sync with DMAI's live learning activity.

The graph topology is LOCKED to the core neuron set defined in the schema
(see metadata.graph_locked). GraphWriter ONLY updates `activation` on existing
neurons (and optionally `weight` on existing synapses). It NEVER adds new
neurons or synapses — this prevents the runaway node explosion that previously
grew the graph from 32 to 400+ nodes from discoveries/insights/syllabus data.

Called after every:
  - Learning / study cycle
  - Autonomous research run
  - Insight added via si_core.add_insight()
  - Capability registered

Usage:
    from components.graph_writer import GraphWriter
    gw = GraphWriter()
    stats = gw.evolve()                                    # full activation pass
    gw.add_insight_node(domain, concept, source)           # nudge insight nodes
    gw.add_topic_node(stage, topic, mastery)               # nudge learning nodes
"""

import json
import logging
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("dmai.graph_writer")

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent
SCHEMA_PATH  = _REPO_ROOT / "aevora-training" / "dashboard" / "data" / "graph_schema.json"
DISCOVERIES  = _REPO_ROOT / "data" / "research" / "discoveries.jsonl"
INSIGHTS     = _REPO_ROOT / "data" / "research" / "insights.jsonl"
DB_PATH      = _REPO_ROOT / "data" / "dmai_knowledge.db"


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ── GraphWriter ────────────────────────────────────────────────────────────────

class GraphWriter:
    """
    Updates activation levels on the LOCKED core knowledge graph.

    The neuron/synapse topology never changes — only `activation` floats on
    existing neurons are refreshed from live learning activity.
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

    # ── Activation helpers (UPDATE-ONLY — never add nodes/synapses) ─────────────

    @staticmethod
    def _by_id(schema: dict) -> dict:
        return {n["id"]: n for n in schema.get("neurons", [])}

    @staticmethod
    def _set_activation(node: dict, target: float) -> None:
        """Move a neuron's activation toward target (existing node only)."""
        node["activation"] = round(_clamp(target), 3)

    @staticmethod
    def _nudge(node: dict, delta: float = 0.05) -> None:
        """Incrementally raise a neuron's activation (existing node only)."""
        node["activation"] = round(_clamp(float(node.get("activation", 0.5)) + delta), 3)

    def _touch_metadata(self, schema: dict) -> None:
        """
        Refresh counts/timestamps. Counts are derived from the (unchanged)
        topology, so they stay pinned at the locked baseline — never grown.
        evolution_cycle is still incremented to record that a cycle ran.
        """
        schema["total_neurons"]  = len(schema.get("neurons", []))
        schema["total_synapses"] = len(schema.get("synapses", []))
        schema["evolution_cycle"] = schema.get("evolution_cycle", 0) + 1
        schema["last_updated"]    = _today_utc()
        meta = schema.setdefault("metadata", {})
        meta["last_live_update"] = _now_utc()
        meta.setdefault("graph_locked", True)

    # ── Full activation pass ─────────────────────────────────────────────────────

    def evolve(self) -> dict:
        """
        Refresh activation levels on existing core neurons from live data:
          1. discoveries.jsonl   → research-cluster activation
          2. insights.jsonl      → insight/self-improvement activation
          3. syllabus mastery DB → learning-cluster activation
          4. capabilities DB     → knowledge_mgr activation
        Never adds neurons or synapses. Returns a stats dict.
        """
        schema = self._load()
        if not schema:
            return {"error": "Could not load graph_schema.json"}

        schema = deepcopy(schema)
        by_id = self._by_id(schema)
        updated = 0

        def bump(node_id: str, target: float) -> None:
            nonlocal updated
            node = by_id.get(node_id)
            if node is not None:
                self._set_activation(node, target)
                updated += 1

        # ── 1. Discoveries → research activity ───────────────────────────────
        discoveries = _load_jsonl(DISCOVERIES)
        if discoveries:
            intensity = 0.5 + min(len(discoveries), 50) / 100.0
            bump("auto_researcher", intensity)
            bump("graph_evolution", intensity)

        # ── 2. Insights → insight processing + self-improvement ──────────────
        insights = _load_jsonl(INSIGHTS)
        if insights:
            intensity = 0.5 + min(len(insights), 50) / 100.0
            bump("insight_processor", intensity)
            bump("si_core", 0.6 + min(len(insights), 40) / 100.0)

        # ── 3. Mastered syllabus topics → learning activation ────────────────
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur  = conn.cursor()
            cur.execute("SELECT AVG(mastery) FROM syllabus_content WHERE mastery >= 0.7")
            avg_mastery = cur.fetchone()[0]
            conn.close()
            if avg_mastery:
                bump("learning_orch", float(avg_mastery))
                bump("stage_learner", float(avg_mastery))
                bump("training_pipeline", float(avg_mastery))
        except Exception as e:
            logger.debug("Could not read syllabus topics from DB: %s", e)

        # ── 4. Capability volume → knowledge_mgr activation ──────────────────
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM capabilities")
            cap_count = cur.fetchone()[0]
            conn.close()
            km = by_id.get("knowledge_mgr")
            if km is not None:
                km["activation"] = min(round(cap_count / 50_000, 3), 1.0)
                km["capability_count"] = cap_count
                updated += 1
        except Exception as e:
            logger.debug("Could not read capabilities count from DB: %s", e)

        # ── Finalize ──────────────────────────────────────────────────────────
        if updated > 0:
            self._touch_metadata(schema)
            saved = self._save(schema)
            logger.info(
                "GraphWriter.evolve: updated %d neuron activations → cycle %d, "
                "topology locked at %d/%d (saved=%s)",
                updated, schema.get("evolution_cycle"),
                schema["total_neurons"], schema["total_synapses"], saved,
            )
        else:
            logger.debug("GraphWriter.evolve: no live data — activations unchanged")

        return {
            "new_neurons":     0,
            "new_synapses":    0,
            "updated_neurons": updated,
            "evolution_cycle": schema.get("evolution_cycle", 0),
            "total_neurons":   schema.get("total_neurons", 0),
            "total_synapses":  schema.get("total_synapses", 0),
        }

    # ── Single-node nudges (called inline during learning) ─────────────────────
    # These NEVER add nodes; they only raise activation on existing core neurons.

    def add_insight_node(self, domain: str, concept: str, source: str = "si_core") -> bool:
        """
        Nudge the insight/self-improvement neurons when a new insight is added.
        Returns False — the locked topology never gains a node.
        """
        if not concept:
            return False
        try:
            schema = self._load()
            if not schema:
                return False
            schema = deepcopy(schema)
            by_id = self._by_id(schema)
            changed = False
            for nid in ("insight_processor", "si_core"):
                node = by_id.get(nid)
                if node is not None:
                    self._nudge(node, 0.05)
                    changed = True
            if changed:
                self._touch_metadata(schema)
                self._save(schema)
                logger.debug("GraphWriter: nudged insight activation (locked topology)")
            return False
        except Exception as e:
            logger.warning("GraphWriter.add_insight_node failed: %s", e)
            return False

    def add_topic_node(self, stage: str, topic: str, mastery: float = 0.9) -> bool:
        """
        Raise learning-cluster activation when a syllabus topic is mastered.
        Returns False — the locked topology never gains a node.
        """
        if not topic:
            return False
        try:
            schema = self._load()
            if not schema:
                return False
            schema = deepcopy(schema)
            by_id = self._by_id(schema)
            changed = False
            for nid in ("learning_orch", "stage_learner"):
                node = by_id.get(nid)
                if node is not None:
                    self._set_activation(node, max(float(node.get("activation", 0.5)), mastery))
                    changed = True
            if changed:
                self._touch_metadata(schema)
                self._save(schema)
                logger.debug("GraphWriter: nudged learning activation (locked topology)")
            return False
        except Exception as e:
            logger.warning("GraphWriter.add_topic_node failed: %s", e)
            return False

    def add_discovery_node(self, domain: str, entities: list, source: str) -> dict:
        """
        Raise research-cluster activation when a discovery is persisted.
        Returns zero counts — the locked topology never gains nodes/synapses.
        """
        try:
            schema = self._load()
            if not schema:
                return {"new_neurons": 0, "new_synapses": 0}
            schema = deepcopy(schema)
            by_id = self._by_id(schema)
            changed = False
            for nid in ("auto_researcher", "graph_evolution"):
                node = by_id.get(nid)
                if node is not None:
                    self._nudge(node, 0.04)
                    changed = True
            if changed:
                self._touch_metadata(schema)
                self._save(schema)
            return {"new_neurons": 0, "new_synapses": 0}
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
                "graph_locked":    schema.get("metadata", {}).get("graph_locked", False),
                "schema_path":     str(self.schema_path),
            }
        except Exception as e:
            return {"error": str(e)}
