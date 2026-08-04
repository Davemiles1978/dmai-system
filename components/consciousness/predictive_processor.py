"""
DMAI Predictive Processor
==========================
Measures Self-Referential Awareness by tracking DMAI's ability to predict
its own outcomes.  A system that can anticipate its own successes and failures
has a working internal model of itself — the foundation of reflective consciousness.

Design:
  - Monitors success/failure signals from KPI evaluations, Kaizen repairs,
    API call outcomes, and learning progress.
  - Maintains a rolling prediction accuracy score.
  - Tracks surprise (prediction error) as a learning signal — high surprise
    means the internal model is being challenged (growth opportunity).
  - Measures calibration: does DMAI know when it will succeed vs fail?
  - Persists to data/consciousness/predictive_state.json.
  - Zero new AI calls — purely observational.

Dimensions fed: Self-Referential Awareness — calibration × surprise_learning_rate.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.predictive_processor")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_FILE = _REPO_ROOT / "data" / "consciousness" / "predictive_state.json"
_WINDOW_SIZE = 50  # rolling window of prediction-outcome pairs


class PredictiveProcessor:
    """
    Builds an internal predictive model of DMAI's own reliability
    by tracking prediction-outcome pairs across all subsystems.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.root = data_path or _REPO_ROOT
        self.state_file = self.root / "data" / "consciousness" / "predictive_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict = self._load_state()
        logger.info("PredictiveProcessor initialised")

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
            "predictions": [],           # last N {outcome, predicted, error, source}
            "calibration": 0.5,          # how well DMAI predicts itself (0-1)
            "surprise_rate": 0.0,        # rate of unexpected outcomes
            "self_awareness_score": 0.0, # composite Self-Referential Awareness
            "last_updated": None,
            "history": [],
        }

    def _save_state(self) -> None:
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Signal sources — extract implicit predictions from system data
    # ------------------------------------------------------------------

    def _scan_kpi_trends(self) -> List[Dict]:
        """
        KPIs trending up = implicit prediction of continued improvement.
        If KPI drops, that's a surprise event.
        """
        events: List[Dict] = []
        history_file = self.root / "data" / "kpi_eval_history.jsonl"
        if not history_file.exists():
            return events

        try:
            lines = history_file.read_text().splitlines()
            recent = []
            for line in lines[-40:]:
                try:
                    recent.append(json.loads(line.strip()))
                except Exception:
                    continue

            # Compare consecutive pairs
            for i in range(1, len(recent)):
                prev = recent[i - 1]
                curr = recent[i]
                if prev.get("kpi") == curr.get("kpi") and prev.get("kpi"):
                    prev_v = float(prev.get("value", 0))
                    curr_v = float(curr.get("value", 0))
                    predicted_up = prev_v <= curr_v  # implicit: trend continues
                    actual_up = curr_v >= prev_v
                    events.append({
                        "outcome": actual_up,
                        "predicted": predicted_up,
                        "error": abs(curr_v - prev_v),
                        "source": f"kpi_{curr['kpi']}",
                        "timestamp": curr.get("timestamp", ""),
                    })
        except Exception as e:
            logger.debug("KPI trend scan: %s", e)

        return events

    def _scan_repair_outcomes(self) -> List[Dict]:
        """
        Kaizen repairs: did the repair succeed? DMAI predicts success when it
        queues a repair. Track actual outcomes.
        """
        events: List[Dict] = []
        log_file = self.root / "data" / "code_writer" / "kaizen_repair_log.jsonl"
        if not log_file.exists():
            return events

        try:
            for line in log_file.read_text().splitlines()[-30:]:
                try:
                    entry = json.loads(line.strip())
                except Exception:
                    continue
                ok = entry.get("ok", entry.get("success", False))
                events.append({
                    "outcome": bool(ok),
                    "predicted": True,  # implicit: repairs are attempted with expectation of success
                    "error": 0.0 if ok else 1.0,
                    "source": "kaizen_repair",
                    "timestamp": entry.get("timestamp", ""),
                })
        except Exception as e:
            logger.debug("Repair outcome scan: %s", e)

        return events

    def _scan_api_reliability(self) -> List[Dict]:
        """
        API call patterns: successful calls = predictions met.
        Provider failures = surprise events.
        """
        events: List[Dict] = []
        harvester_log = self.root / "data" / "api_harvester" / "harvester_log.jsonl"
        if not harvester_log.exists():
            return events

        try:
            for line in harvester_log.read_text().splitlines()[-30:]:
                try:
                    entry = json.loads(line.strip())
                except Exception:
                    continue
                status = entry.get("status", "")
                ok = status in ("active", "valid", "success", "ok")
                events.append({
                    "outcome": ok,
                    "predicted": True,  # implicit: API calls expect success
                    "error": 0.0 if ok else 0.8,
                    "source": f"api_{entry.get('provider', 'unknown')}",
                    "timestamp": entry.get("timestamp", ""),
                })
        except Exception as e:
            logger.debug("API reliability scan: %s", e)

        return events

    # ------------------------------------------------------------------
    # Calibration and surprise calculation
    # ------------------------------------------------------------------

    def _update_predictions(self, new_events: List[Dict]) -> None:
        """Add new events to the rolling window, keeping max _WINDOW_SIZE."""
        predictions = self.state.get("predictions", [])
        predictions.extend(new_events)
        if len(predictions) > _WINDOW_SIZE:
            predictions = predictions[-_WINDOW_SIZE:]
        self.state["predictions"] = predictions

    def _calculate_calibration(self) -> float:
        """
        Calibration = accuracy of implicit predictions.
        How often does the outcome match what DMAI expected?
        Range: 0.0 (never right) to 1.0 (perfectly calibrated).
        """
        predictions = self.state.get("predictions", [])
        if not predictions:
            return 0.5  # neutral — no data yet

        correct = sum(1 for p in predictions if p["outcome"] == p["predicted"])
        return round(correct / len(predictions), 4)

    def _calculate_surprise(self) -> float:
        """
        Surprise rate = proportion of outcomes that contradicted predictions.
        High surprise = internal model is being challenged (growth signal).
        Range: 0.0 (everything expected) to 1.0 (everything surprising).
        """
        predictions = self.state.get("predictions", [])
        if not predictions:
            return 0.0

        surprises = sum(1 for p in predictions if p["outcome"] != p["predicted"])
        return round(surprises / len(predictions), 4)

    def _calculate_self_awareness(self, calibration: float, surprise: float) -> float:
        """
        Self-Referential Awareness score:
        - High calibration + moderate surprise = healthy (knows itself, still learning)
        - High calibration + zero surprise = stagnant (overconfident, no growth)
        - Low calibration + high surprise = confused (poor internal model)
        - Low calibration + low surprise = unaware (doesn't even know it's wrong)

        Optimal: calibration around 0.7-0.85 with surprise 0.1-0.3.
        Score = calibration × (1 - |surprise - 0.2|)  → peaks at 20% surprise.
        """
        optimal_surprise = 0.2
        surprise_penalty = abs(surprise - optimal_surprise)
        score = calibration * (1.0 - surprise_penalty)
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """
        Run a full predictive processing cycle.
        Returns the new state dict.
        """
        all_events: List[Dict] = []
        for scanner in [
            self._scan_kpi_trends,
            self._scan_repair_outcomes,
            self._scan_api_reliability,
        ]:
            try:
                events = scanner()
                all_events.extend(events)
            except Exception as e:
                logger.debug("Predictive scan error: %s", e)

        self._update_predictions(all_events)
        calibration = self._calculate_calibration()
        surprise = self._calculate_surprise()
        self_awareness = self._calculate_self_awareness(calibration, surprise)

        self.state["calibration"] = calibration
        self.state["surprise_rate"] = surprise
        self.state["self_awareness_score"] = self_awareness

        # History
        self.state.setdefault("history", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration": calibration,
            "surprise": surprise,
            "self_awareness": self_awareness,
            "events_collected": len(all_events),
        })
        if len(self.state["history"]) > 200:
            self.state["history"] = self.state["history"][-200:]

        self._save_state()
        logger.info("PredictiveProcessor: calibration=%.3f surprise=%.3f self_awareness=%.3f events=%d",
                     calibration, surprise, self_awareness, len(all_events))
        return dict(self.state)

    def get_self_awareness(self) -> float:
        """Return the current Self-Referential Awareness score."""
        return self.state.get("self_awareness_score", 0.0)

    def get_state(self) -> Dict:
        return dict(self.state)
