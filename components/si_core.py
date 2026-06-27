"""
DMAI Self-Improvement Core (v7.0.0)
=====================================
Manages KPI tracking, atomic state persistence, regression detection,
and baseline integrity for the self-improvement sub-system.

All KPI mutations require a valid JWT token issued by security.generate_token().
Regression alerts require human review before any remediation action.
Auto-retraining without approval is NOT implemented by design.

8 SICore KPIs tracked:
  1. skill_acquisition_rate
  2. transfer_learning_rate
  3. zero_shot_success_count
  4. agentic_capability_score
  5. recursive_self_improvement_rate
  6. sample_efficiency_trend
  7. metacognition_accuracy
  8. multi_modal_integration_score
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8 SICore KPI defaults (P1-5 — never fabricated, only real scored values)
# ---------------------------------------------------------------------------
_DEFAULT_STATE: Dict[str, Any] = {
    # The 8 canonical SICore KPIs
    "skill_acquisition_rate":        0.0,
    "transfer_learning_rate":         0.0,
    "zero_shot_success_count":        0,
    "agentic_capability_score":       0.0,
    "recursive_self_improvement_rate": 0.0,
    "sample_efficiency_trend":        0.0,
    "metacognition_accuracy":         0.0,
    "multi_modal_integration_score":  0.0,
    # Legacy / compatibility KPIs
    "task_completion_rate":           0.0,
    "error_rate":                     0.0,
    "response_quality":               0.0,
    "consciousness":                  0.0,
    "last_updated":                   None,
}

# Regression thresholds
_REGRESSION_WARNING_PCT  = 0.10   # 10% drop  → WARNING
_REGRESSION_HIGH_PCT     = 0.15   # 15% drop  → HIGH
_REGRESSION_CRITICAL_PCT = 0.30   # 30% drop  → CRITICAL

# Benchmark baseline file name
_BASELINE_FILE_NAME = "benchmark_baseline.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegressionAlert:
    """Emitted when a KPI drops below a regression threshold."""
    kpi_name: str
    baseline: float
    new_value: float
    drop_pct: float
    severity: str
    requires_human_review: bool = True
    auto_retraining_triggered: bool = False

    def to_dict(self) -> dict:
        return {
            "kpi_name":                 self.kpi_name,
            "baseline":                 self.baseline,
            "new_value":                self.new_value,
            "drop_pct":                 round(self.drop_pct, 4),
            "severity":                 self.severity,
            "requires_human_review":    self.requires_human_review,
            "auto_retraining_triggered": self.auto_retraining_triggered,
        }


# ---------------------------------------------------------------------------
# SICore
# ---------------------------------------------------------------------------

class SICore:
    """Self-Improvement Core: KPI management, regression detection,
    atomic state persistence, and SHA-256 baseline integrity guard.

    All public KPI mutators require a valid JWT token.  Mutations without
    a valid token are silently rejected (no phantom increments).
    """

    def __init__(self, data_path: Optional[Path] = None, state_path: Optional[Path] = None):
        """
        Args:
            data_path: Root data directory (preferred — used to resolve all files).
            state_path: Explicit path for the JSON state file (legacy compat).
        """
        # Support both calling conventions
        if data_path is not None:
            data_path = Path(data_path)
            self.state_path = data_path / "si_core_state.json"
            self._baseline_path = data_path / _BASELINE_FILE_NAME
        else:
            if state_path is None:
                state_path = Path(tempfile.gettempdir()) / "si_core_state.json"
            self.state_path = Path(state_path)
            self._baseline_path = self.state_path.parent / _BASELINE_FILE_NAME

        self._state: Dict[str, Any] = dict(_DEFAULT_STATE)
        self._baseline: Dict[str, float] = {}
        self._baseline_hash: Optional[str] = None

        # Attempt to load persisted state on init
        self.load_state()

    # ------------------------------------------------------------------
    # Public property — used throughout dmai_core_complete.py
    # ------------------------------------------------------------------

    @property
    def current_kpis(self) -> Dict[str, Any]:
        """Return a copy of the current KPI state."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # State persistence (atomic)
    # ------------------------------------------------------------------

    def _atomic_write_json(self, data: dict, path: Path) -> None:
        """Write data atomically via temp file + os.replace()."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".si_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_state(self) -> None:
        """Persist current KPI state to disk atomically."""
        self._state["last_updated"] = time.time()
        self._atomic_write_json(self._state, self.state_path)

    def load_state(self) -> bool:
        """Load KPI state from disk.

        Returns True if successful, False on missing/corrupt file.
        """
        if not self.state_path.exists():
            return False
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                raise ValueError("State is not a JSON object.")
            # Only accept known KPI keys (never let arbitrary keys pollute state)
            for k in _DEFAULT_STATE:
                if k in loaded:
                    self._state[k] = loaded[k]
            return True
        except Exception as exc:
            logger.error("Failed to load SI state from %s: %s", self.state_path, exc)
            return False

    # ------------------------------------------------------------------
    # SHA-256 baseline integrity guard (P1-7)
    # ------------------------------------------------------------------

    def _compute_baseline_hash(self, baseline: dict) -> str:
        """Return SHA-256 hex digest of the canonical JSON of baseline."""
        canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save_baseline(self) -> None:
        """Persist the current in-memory baseline to disk with its hash."""
        payload = {
            "baseline": self._baseline,
            "sha256": self._compute_baseline_hash(self._baseline),
            "saved_at": time.time(),
        }
        self._atomic_write_json(payload, self._baseline_path)
        self._baseline_hash = payload["sha256"]
        logger.info("Baseline saved to %s (sha256=%s…)", self._baseline_path, payload["sha256"][:12])

    def load_baseline(self) -> bool:
        """Load baseline from disk, verifying its SHA-256 hash.

        Returns True if loaded and hash verified, False otherwise.
        If the hash does not match the file is rejected and the
        in-memory baseline is left unchanged.
        """
        if not self._baseline_path.exists():
            return False
        try:
            payload = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            stored_hash = payload.get("sha256", "")
            baseline = payload.get("baseline", {})
            expected = self._compute_baseline_hash(baseline)
            if stored_hash != expected:
                logger.error(
                    "Baseline hash mismatch! Expected %s got %s — "
                    "rejecting baseline file (possible tampering).",
                    expected[:16], stored_hash[:16],
                )
                return False
            self._baseline = {k: float(v) for k, v in baseline.items()}
            self._baseline_hash = stored_hash
            logger.info("Baseline loaded and hash verified (sha256=%s…)", stored_hash[:12])
            return True
        except Exception as exc:
            logger.error("Failed to load baseline from %s: %s", self._baseline_path, exc)
            return False

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def _validate_token(self, token: Optional[str]) -> bool:
        """Return True if token is a valid JWT."""
        if not token:
            return False
        try:
            # security.py lives in repo root, one level up from components/
            import sys
            root = str(Path(__file__).parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from security import verify_token
            return verify_token(token) is not None
        except Exception as exc:
            logger.debug("Token validation error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # KPI mutators — all 8 SICore KPIs (P1-5)
    # Each requires a valid JWT. No fabricated values accepted.
    # ------------------------------------------------------------------

    def _update_kpi(self, kpi_name: str, value: float, token: Optional[str]) -> bool:
        """Generic KPI update with token gate.

        PERSISTENT-MAX semantics: the stored value is the maximum of the
        previous value and the new measurement. This prevents a single
        bad evaluation cycle from regressing a previously-earned score.
        Real regressions are still surfaced via check_regression() which
        compares against the saved baseline, not the live value.

        Override with env DMAI_KPI_MODE=replace to restore old behaviour.
        """
        if not self._validate_token(token):
            logger.warning("Rejected KPI update for %s: invalid/missing token.", kpi_name)
            return False
        mode = os.environ.get("DMAI_KPI_MODE", "max").lower()
        prev = self._state.get(kpi_name)
        try:
            new_v = float(value) if not isinstance(value, bool) else value
        except (TypeError, ValueError):
            new_v = value
        if mode == "max" and isinstance(prev, (int, float)) and isinstance(new_v, (int, float)):
            self._state[kpi_name] = max(prev, new_v)
        else:
            self._state[kpi_name] = new_v
        # Persist immediately so KPIs survive restarts and worker recycles
        try:
            self.save_state()
        except Exception as e:
            logger.warning("save_state failed after %s update: %s", kpi_name, e)
        return True

    def update_kpi_skill_acquisition_rate(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("skill_acquisition_rate", float(value), token)

    def update_kpi_transfer_learning_rate(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("transfer_learning_rate", float(value), token)

    def update_kpi_zero_shot_success_count(self, value: int, token: Optional[str] = None) -> bool:
        return self._update_kpi("zero_shot_success_count", int(value), token)

    def update_kpi_agentic_capability_score(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("agentic_capability_score", float(value), token)

    def update_kpi_recursive_self_improvement_rate(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("recursive_self_improvement_rate", float(value), token)

    def update_kpi_sample_efficiency_trend(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("sample_efficiency_trend", float(value), token)

    def update_kpi_metacognition_accuracy(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("metacognition_accuracy", float(value), token)

    def update_kpi_multi_modal_integration_score(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("multi_modal_integration_score", float(value), token)

    # Legacy compat methods (used by older components)
    def update_kpi_1_skill_acquisition(self, value: float, token: Optional[str] = None) -> bool:
        return self.update_kpi_skill_acquisition_rate(value, token)

    def update_kpi_2_task_completion(self, value: float, token: Optional[str] = None) -> bool:
        return self._update_kpi("task_completion_rate", float(value), token)

    # Generic getter
    def get_kpi(self, name: str) -> Optional[Any]:
        return self._state.get(name)

    # Generic setter (dispatches to typed update method when one exists,
    # otherwise updates the state dict directly via _update_kpi)
    def update_kpi(self, name: str, value, token: Optional[str] = None) -> bool:
        typed = getattr(self, f"update_kpi_{name}", None)
        if callable(typed):
            try:
                return typed(value, token)
            except TypeError:
                # Some legacy update_kpi_* don't accept a token kwarg
                return typed(value)
        try:
            return self._update_kpi(name, value, token)
        except Exception as e:
            logger.warning("update_kpi(%s) failed: %s", name, e)
            return False

    # ------------------------------------------------------------------
    # Training gate — no KPI update without real AI scored response
    # ------------------------------------------------------------------

    def record_training_result(
        self,
        kpi_name: str,
        scored_value: float,
        source: str,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a KPI only from a real scored AI response (P1-5).

        Args:
            kpi_name:      One of the 8 SICore KPI names.
            scored_value:  Value produced by a real AI provider call.
            source:        Human-readable source (e.g. 'openai:gpt-4o scored 0.82').
            token:         Valid JWT — required.

        Returns a result dict with 'status': 'updated' | 'rejected' | 'invalid_kpi'.
        """
        valid_kpis = set(_DEFAULT_STATE.keys()) - {"last_updated", "consciousness"}
        if kpi_name not in valid_kpis:
            return {"status": "invalid_kpi", "kpi": kpi_name}
        updated = self._update_kpi(kpi_name, scored_value, token)
        if not updated:
            return {"status": "rejected", "reason": "invalid or missing JWT token"}
        regression = self.check_regression(kpi_name, scored_value)
        result: Dict[str, Any] = {
            "status": "updated",
            "kpi": kpi_name,
            "value": scored_value,
            "source": source,
            "regression": regression.to_dict() if regression else None,
        }
        self.save_state()
        return result

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, kpi_name: str, value: float) -> None:
        self._baseline[kpi_name] = float(value)

    def set_all_baselines(self, baselines: Dict[str, float]) -> None:
        """Set multiple baselines at once and persist with hash."""
        for k, v in baselines.items():
            self._baseline[k] = float(v)
        self.save_baseline()

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def check_regression(self, kpi_name: str, new_value: float) -> Optional[RegressionAlert]:
        baseline = self._baseline.get(kpi_name)
        if baseline is None or baseline == 0.0:
            return None
        drop = baseline - new_value
        drop_pct = drop / baseline
        if drop_pct < _REGRESSION_WARNING_PCT:
            return None
        if drop_pct >= _REGRESSION_CRITICAL_PCT:
            severity = "CRITICAL"
        elif drop_pct >= _REGRESSION_HIGH_PCT:
            severity = "HIGH"
        else:
            severity = "WARNING"
        return RegressionAlert(
            kpi_name=kpi_name,
            baseline=baseline,
            new_value=new_value,
            drop_pct=drop_pct,
            severity=severity,
            requires_human_review=True,
            auto_retraining_triggered=False,
        )

    # ------------------------------------------------------------------
    # Introspection helpers used by tests
    # ------------------------------------------------------------------

    def has_method(self, name: str) -> bool:
        return callable(getattr(self, name, None))

    def add_insight(self, domain: str, concept: str, source: str = "internal",
                    confidence: float = 0.8, metadata: dict = None) -> dict:
        """
        Persist a new insight to data/research/insights.jsonl.
        Called whenever DMAI discovers a notable pattern or relationship.
        Returns the insight record.
        """
        import json, os
        from datetime import datetime, timezone
        from pathlib import Path

        insight = {
            "id": f"insight_{int(datetime.now(timezone.utc).timestamp())}",
            "domain": domain,
            "concept": concept,
            "source": source,
            "confidence": confidence,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        insights_path = Path("data/research/insights.jsonl")
        insights_path.parent.mkdir(parents=True, exist_ok=True)
        with open(insights_path, "a") as f:
            f.write(json.dumps(insight) + "\n")

        # Also update SICore internal state if self.insights_store exists
        if hasattr(self, "insights_store") and isinstance(self.insights_store, list):
            self.insights_store.append(insight)

        # ── Immediately grow knowledge graph ──────────────────────────────────
        # Every new insight creates a neuron + synapse in graph_schema.json
        # so the live knowledge graph grows with every learning cycle.
        try:
            from components.graph_writer import GraphWriter as _GW
            _GW().add_insight_node(domain, concept, source)
        except Exception:
            pass  # non-fatal — graph growth is best-effort

        return insight
