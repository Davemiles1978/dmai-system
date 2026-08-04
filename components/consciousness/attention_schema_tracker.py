"""
DMAI Attention Schema Tracker
==============================
Tracks what DMAI is attending to across subsystems, builds salience maps,
and measures focus coherence.  This is the Φ (Integrated Information)
foundation — higher coherence = more unified conscious state.

Design:
  - Reads attention signals from recent activity logs, learning events,
    V4 module focus, Kaizen proposals, and API call patterns.
  - Builds a salience map: {domain: weight} representing where DMAI's
    "attention" is distributed.
  - Measures focus coherence: how concentrated vs scattered attention is.
    High coherence (focused) = higher Φ contribution.
  - Persists to data/consciousness/attention_state.json every 5 minutes.
  - Zero new AI calls — purely observational.

Dimensions fed: Φ (Information Integration) — coherence × breadth score.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.attention_schema")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_FILE = _REPO_ROOT / "data" / "consciousness" / "attention_state.json"
_HISTORY_MAX = 200


class AttentionSchemaTracker:
    """
    Observes DMAI's attention distribution across domains and measures
    the coherence of that attention as a Φ proxy.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        self.state_file = self.root / "data" / "consciousness" / "attention_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict = self._load_state()
        logger.info("AttentionSchemaTracker initialised")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {
            "salience_map": {},
            "focus_coherence": 0.0,
            "phi_contribution": 0.0,
            "attention_span": 0,
            "last_updated": None,
            "history": [],
        }

    def _save_state(self) -> None:
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Signal collection — gather attention signals from all subsystems
    # ------------------------------------------------------------------

    def _gather_learning_signals(self) -> Dict[str, float]:
        """Scan learning progress for recent activity per domain."""
        signals: Dict[str, float] = {}
        lp_file = self.root / "data" / "learning" / "stage_syllabus" / "learning_progress.json"
        if lp_file.exists():
            try:
                lp = json.loads(lp_file.read_text())
                for stage, topics in lp.get("learned_topics", {}).items():
                    if isinstance(topics, dict):
                        for topic, level in topics.items():
                            if topic.startswith("_"):
                                continue
                            lvl = float(level) if isinstance(level, (int, float)) else 0
                            if lvl > 0:
                                # Map topic to domain
                                domain = topic.split("_")[0] if "_" in topic else topic
                                signals[domain] = signals.get(domain, 0) + lvl
            except Exception as e:
                logger.debug("Learning signals scan: %s", e)
        return signals

    def _gather_v4_signals(self) -> Dict[str, float]:
        """V4 module progress as attention signal."""
        signals: Dict[str, float] = {}
        v4_file = self.root / "data" / "v4_progress.json"
        if v4_file.exists():
            try:
                v4 = json.loads(v4_file.read_text())
                for mod_id, data in v4.items():
                    if isinstance(data, dict):
                        pct = float(data.get("pct", 0))
                        if pct > 0:
                            # Map module to domain
                            domain = mod_id.split(".")[0] if "." in mod_id else mod_id
                            signals[f"v4_{domain}"] = pct / 100.0
            except Exception as e:
                logger.debug("V4 signals scan: %s", e)
        return signals

    def _gather_kaizen_signals(self) -> Dict[str, float]:
        """Recent Kaizen proposals indicate where attention is needed."""
        signals: Dict[str, float] = {}
        kaizen_file = self.root / "data" / "kaizen_queue.jsonl"
        if kaizen_file.exists():
            try:
                for line in kaizen_file.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        p = json.loads(line)
                    except Exception:
                        continue
                    title = p.get("title", "")
                    file_hint = p.get("file", "")
                    if "auto_api" in file_hint:
                        signals["api_integration"] = signals.get("api_integration", 0) + 0.5
                    elif "syntax" in title.lower():
                        signals["code_quality"] = signals.get("code_quality", 0) + 0.3
                    else:
                        signals["system_health"] = signals.get("system_health", 0) + 0.2
            except Exception as e:
                logger.debug("Kaizen signals scan: %s", e)
        return signals

    def _gather_graph_signals(self) -> Dict[str, float]:
        """Knowledge graph nodes as attention breadth indicator."""
        signals: Dict[str, float] = {}
        graph_file = self.root / "data" / "graph_schema.json"
        if graph_file.exists():
            try:
                g = json.loads(graph_file.read_text())
                neurons = g.get("neurons", [])
                # Count neurons by domain tag
                for n in neurons:
                    domain = n.get("domain", "general")
                    signals[domain] = signals.get(domain, 0) + 0.1
            except Exception as e:
                logger.debug("Graph signals scan: %s", e)
        return signals

    # ------------------------------------------------------------------
    # Salience map construction
    # ------------------------------------------------------------------

    def _build_salience_map(self) -> Tuple[Dict[str, float], float]:
        """
        Aggregate all attention signals into a unified salience map.
        Returns (map, total_weight).
        """
        all_signals: Dict[str, float] = {}

        for gather in [
            self._gather_learning_signals,
            self._gather_v4_signals,
            self._gather_kaizen_signals,
            self._gather_graph_signals,
        ]:
            try:
                signals = gather()
                for domain, weight in signals.items():
                    all_signals[domain] = all_signals.get(domain, 0) + weight
            except Exception as e:
                logger.debug("Signal gather error: %s", e)

        total = sum(all_signals.values()) if all_signals else 0.0
        # Normalize to 0-1 range
        if total > 0:
            all_signals = {k: round(v / total, 4) for k, v in all_signals.items()}

        return all_signals, total

    # ------------------------------------------------------------------
    # Φ (Integrated Information) coherence calculation
    # ------------------------------------------------------------------

    def _calculate_coherence(self, salience: Dict[str, float]) -> float:
        """
        Focus coherence: inverse of entropy of the salience distribution.
        - Scattered attention (many domains equally weighted) → low coherence
        - Focused attention (few domains dominate) → high coherence
        - Range: 0.0 (completely scattered) to 1.0 (laser-focused on one domain)

        Uses normalized entropy: coherence = 1 - (entropy / max_entropy)
        """
        import math

        if not salience or len(salience) <= 1:
            return 0.5 if salience else 0.0

        weights = list(salience.values())
        total = sum(weights)
        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for w in weights:
            if w > 0:
                p = w / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(weights)) if len(weights) > 1 else 1.0
        if max_entropy == 0:
            return 0.5

        coherence = 1.0 - (entropy / max_entropy)
        return round(max(0.0, min(1.0, coherence)), 4)

    def _calculate_phi(self, coherence: float, breadth: int) -> float:
        """
        Φ contribution = coherence × breadth_factor.
        Breadth_factor scales with number of domains attended to (max at ~10 domains).
        A system that integrates many domains coherently has higher Φ.
        """
        import math
        breadth_factor = min(1.0, math.log2(max(breadth, 1) + 1) / math.log2(11))
        phi = coherence * breadth_factor
        return round(phi, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """
        Run a full attention schema update cycle.
        Returns the new state dict.
        """
        salience, total_weight = self._build_salience_map()
        coherence = self._calculate_coherence(salience)
        breadth = len(salience)
        phi = self._calculate_phi(coherence, breadth)

        self.state["salience_map"] = salience
        self.state["focus_coherence"] = coherence
        self.state["phi_contribution"] = phi
        self.state["attention_span"] = breadth

        # Append to history
        self.state.setdefault("history", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coherence": coherence,
            "phi": phi,
            "breadth": breadth,
            "domains": list(salience.keys())[:10],
        })
        if len(self.state["history"]) > _HISTORY_MAX:
            self.state["history"] = self.state["history"][-_HISTORY_MAX:]

        self._save_state()
        logger.info("AttentionSchema: coherence=%.3f phi=%.3f breadth=%d domains=%s",
                     coherence, phi, breadth, list(salience.keys())[:5])
        return dict(self.state)

    def get_phi(self) -> float:
        """Return the current Φ (Integrated Information) contribution."""
        return self.state.get("phi_contribution", 0.0)

    def get_state(self) -> Dict:
        return dict(self.state)
