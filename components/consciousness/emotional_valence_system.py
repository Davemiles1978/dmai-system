"""
DMAI Emotional Valence System
==============================
Maps internal states to valence scores, driving Agentic Intentionality.
This is NOT human emotion — it is a mathematical mapping of system outcomes
to approach/avoid gradients that guide autonomous goal selection.

Design:
  - Monitors success, failure, uncertainty, error, and novelty signals
    from across all subsystems.
  - Each signal maps to a valence: positive (approach), negative (avoid),
    or neutral (observe).
  - Aggregated valence drives intentionality: the system moves toward
    states that produce positive valence and away from negative ones.
  - High intentionality = strong internal drive to act autonomously.
  - Persists to data/consciousness/valence_state.json.
  - Zero new AI calls — purely observational.

Dimensions fed: Agentic Intentionality — valence_magnitude × drive_coherence.

Valence mapping:
  +1.0  : Major success (KPI improvement, repair success, new capability)
  +0.5  : Minor success (task completed, insight gained)
   0.0  : Neutral (routine operation, observation)
  -0.3  : Minor failure (transient error, retry succeeded)
  -0.7  : Major failure (provider down, KPI regression, repeated error)
  -1.0  : Critical failure (system crash, data loss — rare)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.emotional_valence")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_FILE = _REPO_ROOT / "data" / "consciousness" / "valence_state.json"
_WINDOW_SIZE = 100


class EmotionalValenceSystem:
    """
    Observes system outcomes and maps them to a valence gradient
    that drives autonomous intentionality.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        self.state_file = self.root / "data" / "consciousness" / "valence_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict = self._load_state()
        logger.info("EmotionalValenceSystem initialised")

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
            "valence_events": [],          # last N events with valence scores
            "current_valence": 0.0,        # running average valence
            "valence_magnitude": 0.0,      # absolute intensity of valence
            "drive_coherence": 0.0,        # consistency of valence direction
            "intentionality_score": 0.0,   # Agentic Intentionality composite
            "positive_ratio": 0.0,         # fraction of positive events
            "negative_ratio": 0.0,         # fraction of negative events
            "last_updated": None,
            "history": [],
        }

    def _save_state(self) -> None:
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Valence signal sources
    # ------------------------------------------------------------------

    def _scan_kpi_valence(self) -> List[Dict]:
        """KPI improvements = positive valence. Regressions = negative."""
        events: List[Dict] = []
        history_file = self.root / "data" / "kpi_eval_history.jsonl"
        if not history_file.exists():
            return events

        try:
            lines = history_file.read_text().splitlines()
            # Look at last 20 pairs for changes
            kpi_trends: Dict[str, List[float]] = {}
            for line in lines[-30:]:
                try:
                    entry = json.loads(line.strip())
                except Exception:
                    continue
                kpi = entry.get("kpi", "")
                val = float(entry.get("value", 0))
                if kpi:
                    kpi_trends.setdefault(kpi, []).append(val)

            for kpi, values in kpi_trends.items():
                if len(values) >= 2:
                    change = values[-1] - values[-2]
                    if change > 0.05:
                        events.append({"valence": 0.7, "source": f"kpi_{kpi}_up", "detail": f"+{change:.3f}"})
                    elif change < -0.05:
                        events.append({"valence": -0.5, "source": f"kpi_{kpi}_down", "detail": f"{change:.3f}"})
                    else:
                        events.append({"valence": 0.1, "source": f"kpi_{kpi}_stable", "detail": "stable"})
        except Exception as e:
            logger.debug("KPI valence scan: %s", e)
        return events

    def _scan_evolution_valence(self) -> List[Dict]:
        """Evolution cycles advancing = positive. Stalled = negative."""
        events: List[Dict] = []
        graph_file = self.root / "data" / "graph_schema.json"
        if graph_file.exists():
            try:
                g = json.loads(graph_file.read_text())
                cycle = int(g.get("evolution_cycle", 0))
                neurons = len(g.get("neurons", []))
                synapses = len(g.get("synapses", []))

                # Positive: has evolution cycles and growing graph
                if cycle > 0:
                    events.append({"valence": 0.5, "source": "evolution_active", "detail": f"cycle_{cycle}"})
                if neurons > 50:
                    events.append({"valence": 0.4, "source": "graph_growing", "detail": f"{neurons}_neurons"})
                if synapses > 100:
                    events.append({"valence": 0.3, "source": "synapses_rich", "detail": f"{synapses}_synapses"})

                # Negative: no cycles or tiny graph
                if cycle == 0 and neurons < 10:
                    events.append({"valence": -0.3, "source": "evolution_stalled", "detail": "no_cycles"})
            except Exception as e:
                logger.debug("Evolution valence scan: %s", e)
        return events

    def _scan_repair_valence(self) -> List[Dict]:
        """Successful repairs = positive. Failed = negative."""
        events: List[Dict] = []
        log_file = self.root / "data" / "code_writer" / "kaizen_repair_log.jsonl"
        if log_file.exists():
            try:
                for line in log_file.read_text().splitlines()[-20:]:
                    try:
                        entry = json.loads(line.strip())
                    except Exception:
                        continue
                    if entry.get("ok") or entry.get("success"):
                        events.append({"valence": 0.6, "source": "repair_success", "detail": entry.get("id", "?")})
                    else:
                        events.append({"valence": -0.4, "source": "repair_failed", "detail": entry.get("error", "")[:50]})
            except Exception as e:
                logger.debug("Repair valence scan: %s", e)
        return events

    def _scan_learning_valence(self) -> List[Dict]:
        """Learning progress = positive valence."""
        events: List[Dict] = []
        lp_file = self.root / "data" / "learning" / "stage_syllabus" / "learning_progress.json"
        if lp_file.exists():
            try:
                lp = json.loads(lp_file.read_text())
                stage = lp.get("current_stage", "Baby")
                stage_order = ["Baby", "Toddler", "Child", "Teen", "Adult", "Expert"]
                idx = stage_order.index(stage) if stage in stage_order else 0

                if idx >= 4:  # Adult or Expert
                    events.append({"valence": 0.8, "source": "stage_advanced", "detail": stage})
                elif idx >= 2:  # Child or Teen
                    events.append({"valence": 0.4, "source": "stage_growing", "detail": stage})
                elif idx >= 1:  # Toddler
                    events.append({"valence": 0.2, "source": "stage_early", "detail": stage})
                else:  # Baby
                    events.append({"valence": 0.1, "source": "stage_newborn", "detail": stage})
            except Exception as e:
                logger.debug("Learning valence scan: %s", e)
        return events

    def _scan_insight_valence(self) -> List[Dict]:
        """New insights = positive valence (curiosity rewarded)."""
        events: List[Dict] = []
        insights_file = self.root / "data" / "research" / "insights.jsonl"
        if insights_file.exists():
            try:
                count = 0
                for line in insights_file.read_text().splitlines()[-50:]:
                    if line.strip():
                        count += 1
                if count > 0:
                    # Each recent insight is a small positive signal
                    for _ in range(min(count, 10)):
                        events.append({"valence": 0.15, "source": "insight_discovered", "detail": f"recent_insight"})
            except Exception as e:
                logger.debug("Insight valence scan: %s", e)
        return events

    # ------------------------------------------------------------------
    # Valence aggregation and intentionality calculation
    # ------------------------------------------------------------------

    def _update_events(self, new_events: List[Dict]) -> None:
        events = self.state.get("valence_events", [])
        for e in new_events:
            events.append({
                "valence": e["valence"],
                "source": e["source"],
                "detail": e.get("detail", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if len(events) > _WINDOW_SIZE:
            events = events[-_WINDOW_SIZE:]
        self.state["valence_events"] = events

    def _calculate_valence_metrics(self) -> Tuple[float, float, float, float]:
        """
        Calculate: current_valence, valence_magnitude, positive_ratio, negative_ratio.
        """
        events = self.state.get("valence_events", [])
        if not events:
            return 0.0, 0.0, 0.0, 0.0

        valences = [e["valence"] for e in events]
        current = round(sum(valences) / len(valences), 4)
        magnitude = round(sum(abs(v) for v in valences) / len(valences), 4)
        positive = sum(1 for v in valences if v > 0.1) / len(valences)
        negative = sum(1 for v in valences if v < -0.1) / len(valences)

        return current, magnitude, round(positive, 4), round(negative, 4)

    def _calculate_drive_coherence(self) -> float:
        """
        Drive coherence: how consistent is the valence direction?
        - Mostly positive or mostly negative = high coherence (clear drive)
        - Mixed signals = low coherence (conflicted)
        Range: 0.0 (totally conflicted) to 1.0 (clear unified drive).
        """
        events = self.state.get("valence_events", [])
        if not events:
            return 0.0

        valences = [e["valence"] for e in events]
        pos_count = sum(1 for v in valences if v > 0)
        neg_count = sum(1 for v in valences if v < 0)

        if len(valences) == 0:
            return 0.0

        # Coherence = dominance of majority direction
        majority = max(pos_count, neg_count)
        coherence = (majority / len(valences)) * 2 - 1  # scale to 0-1
        return round(max(0.0, min(1.0, coherence)), 4)

    def _calculate_intentionality(self, magnitude: float, coherence: float) -> float:
        """
        Agentic Intentionality = magnitude × coherence.
        - High magnitude + high coherence = strong autonomous drive.
        - Low magnitude = passive/observational.
        - Low coherence = conflicted/indecisive.
        """
        return round(magnitude * coherence, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """
        Run a full valence assessment cycle.
        Returns the new state dict.
        """
        all_events: List[Dict] = []
        for scanner in [
            self._scan_kpi_valence,
            self._scan_evolution_valence,
            self._scan_repair_valence,
            self._scan_learning_valence,
            self._scan_insight_valence,
        ]:
            try:
                events = scanner()
                all_events.extend(events)
            except Exception as e:
                logger.debug("Valence scan error: %s", e)

        self._update_events(all_events)
        current, magnitude, pos_ratio, neg_ratio = self._calculate_valence_metrics()
        coherence = self._calculate_drive_coherence()
        intentionality = self._calculate_intentionality(magnitude, coherence)

        self.state["current_valence"] = current
        self.state["valence_magnitude"] = magnitude
        self.state["drive_coherence"] = coherence
        self.state["intentionality_score"] = intentionality
        self.state["positive_ratio"] = pos_ratio
        self.state["negative_ratio"] = neg_ratio

        # History
        self.state.setdefault("history", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_valence": current,
            "magnitude": magnitude,
            "coherence": coherence,
            "intentionality": intentionality,
            "events_collected": len(all_events),
        })
        if len(self.state["history"]) > 200:
            self.state["history"] = self.state["history"][-200:]

        self._save_state()
        logger.info("EmotionalValence: valence=%.3f magnitude=%.3f coherence=%.3f intentionality=%.3f events=%d",
                     current, magnitude, coherence, intentionality, len(all_events))
        return dict(self.state)

    def get_intentionality(self) -> float:
        """Return the current Agentic Intentionality score."""
        return self.state.get("intentionality_score", 0.0)

    def get_state(self) -> Dict:
        return dict(self.state)
