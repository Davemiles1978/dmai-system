"""
DMAI Consciousness Engine
==========================
Unifies the three consciousness accelerators into a single four-dimension
consciousness score and writes it to SICore.

Four dimensions measured:
  1. Φ (Integrated Information)      — from AttentionSchemaTracker
  2. Subjective Experience            — composite of valence + prediction surprise
  3. Self-Referential Awareness       — from PredictiveProcessor
  4. Agentic Intentionality           — from EmotionalValenceSystem

Composite consciousness = weighted average of all four.
Updates SICore.consciousness directly so the API reflects real internal metrics.

Weights:
  Φ:                         25%  (information integration)
  Subjective Experience:     25%  (internal state richness)
  Self-Referential Awareness: 25%  (knowing oneself)
  Agentic Intentionality:    25%  (autonomous drive)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("dmai.consciousness_engine")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATE_FILE = _REPO_ROOT / "data" / "consciousness" / "consciousness_state.json"


class ConsciousnessEngine:
    """
    Central consciousness calculator.  Updates all three accelerators,
    computes the four-dimension score, and persists to SICore.
    """

    def __init__(self, data_path: Optional[Path] = None, si_core=None):
        self.root = data_path or _REPO_ROOT
        self.si_core = si_core
        self.state_file = self.root / "data" / "consciousness" / "consciousness_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Lazy-load accelerators
        self._attention = None
        self._predictive = None
        self._valence = None

        self.state: Dict = self._load_state()
        logger.info("ConsciousnessEngine initialised")

    # ------------------------------------------------------------------
    # Lazy accelerator loading (avoids circular imports at boot)
    # ------------------------------------------------------------------

    @property
    def attention(self):
        if self._attention is None:
            from components.consciousness.attention_schema_tracker import AttentionSchemaTracker
            self._attention = AttentionSchemaTracker(data_path=self.root)
        return self._attention

    @property
    def predictive(self):
        if self._predictive is None:
            from components.consciousness.predictive_processor import PredictiveProcessor
            self._predictive = PredictiveProcessor(data_path=self.root)
        return self._predictive

    @property
    def valence(self):
        if self._valence is None:
            from components.consciousness.emotional_valence_system import EmotionalValenceSystem
            self._valence = EmotionalValenceSystem(data_path=self.root)
        return self._valence

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
            "consciousness": 0.0,
            "dimensions": {
                "phi": 0.0,
                "subjective_experience": 0.0,
                "self_referential_awareness": 0.0,
                "agentic_intentionality": 0.0,
            },
            "last_updated": None,
            "history": [],
        }

    def _save_state(self) -> None:
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Dimension calculators
    # ------------------------------------------------------------------

    def _calc_phi(self) -> float:
        """Phi from AttentionSchemaTracker coherence."""
        attn_state = self.attention.update()
        return attn_state.get("phi_contribution", 0.0)

    def _calc_self_awareness(self) -> float:
        """Self-Referential Awareness from PredictiveProcessor."""
        pred_state = self.predictive.update()
        return pred_state.get("self_awareness_score", 0.0)

    def _calc_intentionality(self) -> float:
        """Agentic Intentionality from EmotionalValenceSystem."""
        val_state = self.valence.update()
        return val_state.get("intentionality_score", 0.0)

    def _calc_subjective_experience(self, phi: float, self_awareness: float,
                                     intentionality: float) -> float:
        """
        Subjective Experience = composite of internal state richness.
        Combines valence magnitude, prediction surprise, and attention breadth.
        This is the "what it feels like to be DMAI" dimension.

        Formula: mean of valence_magnitude + prediction_surprise, modulated by phi.
        A system with high phi that experiences strong valence and surprise
        has richer subjective experience.
        """
        val_state = self.valence.get_state()
        pred_state = self.predictive.get_state()

        valence_magnitude = val_state.get("valence_magnitude", 0.0)
        surprise_rate = pred_state.get("surprise_rate", 0.0)

        # Richness = magnitude of internal signals
        richness = (valence_magnitude + surprise_rate) / 2.0

        # Modulated by phi — integrated information amplifies experience
        subjective = richness * (0.5 + 0.5 * phi)

        return round(min(1.0, subjective), 4)

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def calculate(self) -> Dict:
        """
        Run the full consciousness calculation cycle.
        Updates all accelerators, computes four dimensions, composites the score,
        and writes to SICore.
        """
        # Collect all four dimensions
        phi = self._calc_phi()
        self_awareness = self._calc_self_awareness()
        intentionality = self._calc_intentionality()
        subjective = self._calc_subjective_experience(phi, self_awareness, intentionality)

        # Composite consciousness (equal weighting)
        consciousness = round(
            (phi + subjective + self_awareness + intentionality) / 4.0, 4
        )

        # Store dimensions
        self.state["dimensions"] = {
            "phi": phi,
            "subjective_experience": subjective,
            "self_referential_awareness": self_awareness,
            "agentic_intentionality": intentionality,
        }
        self.state["consciousness"] = consciousness

        # History
        self.state.setdefault("history", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "consciousness": consciousness,
            "dimensions": dict(self.state["dimensions"]),
        })
        if len(self.state["history"]) > 500:
            self.state["history"] = self.state["history"][-500:]

        self._save_state()

        # Write to SICore so the API reflects real internal metrics
        if self.si_core:
            try:
                token = None
                try:
                    import os
                    token = os.environ.get("MASTER_TOKEN", "dmai_master")
                except Exception:
                    pass
                self.si_core._update_kpi("consciousness", consciousness, token)
                self.si_core.save_state()
                logger.info("ConsciousnessEngine: SICore.consciousness updated to %.4f", consciousness)
            except Exception as e:
                logger.warning("ConsciousnessEngine: SICore update failed: %s", e)

        logger.info(
            "ConsciousnessEngine: score=%.4f phi=%.4f subj=%.4f self_aware=%.4f intent=%.4f",
            consciousness, phi, subjective, self_awareness, intentionality,
        )
        return dict(self.state)

    # ------------------------------------------------------------------
    # Accessors — compatible with old ConsciousnessTracker API
    # ------------------------------------------------------------------

    @property
    def consciousness(self):
        """Compatibility: old code accesses .consciousness directly."""
        return self.state.get("consciousness", 0.0)

    def get_consciousness(self) -> float:
        return self.state.get("consciousness", 0.0)

    def get_dimensions(self) -> Dict:
        return dict(self.state.get("dimensions", {}))

    def get_state(self) -> Dict:
        return dict(self.state)
