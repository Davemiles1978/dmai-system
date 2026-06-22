"""
DMAI Self-Improvement Core (Patched)
=====================================
Manages KPI tracking, atomic state persistence, and regression detection
for the self-improvement sub-system.

All KPI mutations require a valid JWT token issued by security.generate_token().
Regression alerts require human review before any remediation action is taken.
Auto-retraining without approval is NOT implemented by design.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KPI state defaults
# ---------------------------------------------------------------------------

_DEFAULT_STATE = {
    "skill_acquisition_rate": 0.0,
    "task_completion_rate": 0.0,
    "error_rate": 0.0,
    "response_quality": 0.0,
    "last_updated": None,
}

# Regression thresholds
_REGRESSION_WARNING_PCT = 0.10    # 10% drop -> WARNING
_REGRESSION_HIGH_PCT = 0.15       # 15% drop -> HIGH
_REGRESSION_CRITICAL_PCT = 0.30   # 30% drop -> CRITICAL


@dataclass
class RegressionAlert:
    """An alert produced when a KPI drops below a regression threshold.

    Attributes:
        kpi_name: The name of the KPI that regressed.
        baseline: The reference value the regression is measured against.
        new_value: The newly observed value.
        drop_pct: Fractional drop (e.g. 0.20 for 20%).
        severity: One of WARNING, HIGH, CRITICAL.
        requires_human_review: Always True; no auto-retraining is permitted.
        auto_retraining_triggered: Always False; present for audit purposes.
    """

    kpi_name: str
    baseline: float
    new_value: float
    drop_pct: float
    severity: str
    requires_human_review: bool = True
    auto_retraining_triggered: bool = False

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this alert."""
        return {
            "kpi_name": self.kpi_name,
            "baseline": self.baseline,
            "new_value": self.new_value,
            "drop_pct": round(self.drop_pct, 4),
            "severity": self.severity,
            "requires_human_review": self.requires_human_review,
            "auto_retraining_triggered": self.auto_retraining_triggered,
        }


class SICore:
    """Self-Improvement Core: KPI management and regression detection.

    All public KPI mutators require a valid JWT token.  Mutations without
    a valid token are silently rejected (no phantom increments).

    State is persisted atomically via os.replace() to prevent corruption.
    """

    def __init__(self, state_path: Optional[Path] = None):
        """Initialise SICore with optional persistent state path.

        Args:
            state_path: Path for the JSON state file.  If None a temp path
                is used (state is still written atomically but not permanent).
        """
        if state_path is None:
            state_path = Path(tempfile.gettempdir()) / "si_core_state.json"
        self.state_path = Path(state_path)
        self._state: dict = dict(_DEFAULT_STATE)
        self._baseline: dict = {}

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _atomic_write_json(self, data: dict, path: Path) -> None:
        """Write data to path atomically using a temp file + os.replace().

        Args:
            data: JSON-serialisable dict to write.
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".si_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up the temp file on failure; do not corrupt destination
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

        Returns:
            True if state was loaded successfully, False if the file is
            missing or contains invalid JSON.  On failure the in-memory
            state is left unchanged and the corrupt file is NOT overwritten.
        """
        if not self.state_path.exists():
            logger.debug("State file not found: %s", self.state_path)
            return False
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            # Validate basic structure before accepting
            if not isinstance(loaded, dict):
                raise ValueError("State is not a JSON object.")
            self._state.update(loaded)
            return True
        except Exception as exc:
            logger.error(
                "Failed to load SI state from %s: %s. "
                "Retaining previous in-memory state.",
                self.state_path,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Token validation (delegates to security module)
    # ------------------------------------------------------------------

    def _validate_token(self, token: Optional[str]) -> bool:
        """Return True if token is a valid JWT issued by security.generate_token().

        Args:
            token: Encoded JWT string, or None.
        """
        if not token:
            return False
        try:
            import sys
            _fixes_dir = str(Path(__file__).parent)
            if _fixes_dir not in sys.path:
                sys.path.insert(0, _fixes_dir)
            from security import verify_token
            payload = verify_token(token)
            return payload is not None
        except Exception as exc:
            logger.debug("Token validation error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # KPI mutators — each requires a valid token
    # ------------------------------------------------------------------

    def update_kpi_1_skill_acquisition(
        self, value: float, token: Optional[str] = None
    ) -> bool:
        """Update the skill_acquisition_rate KPI.

        Args:
            value: New KPI value (0.0 - 100.0 scale).
            token: Valid JWT token; mutation is rejected without it.

        Returns:
            True if the KPI was updated, False if token was invalid.
        """
        if not self._validate_token(token):
            logger.warning(
                "Rejected KPI update for skill_acquisition_rate: invalid token."
            )
            return False
        self._state["skill_acquisition_rate"] = float(value)
        return True

    def update_kpi_2_task_completion(
        self, value: float, token: Optional[str] = None
    ) -> bool:
        """Update the task_completion_rate KPI.

        Args:
            value: New KPI value (0.0 - 100.0 scale).
            token: Valid JWT token; mutation is rejected without it.

        Returns:
            True if the KPI was updated, False if token was invalid.
        """
        if not self._validate_token(token):
            logger.warning(
                "Rejected KPI update for task_completion_rate: invalid token."
            )
            return False
        self._state["task_completion_rate"] = float(value)
        return True

    def get_kpi(self, name: str) -> Optional[float]:
        """Retrieve the current value of a named KPI.

        Args:
            name: KPI name, e.g. 'skill_acquisition_rate'.

        Returns:
            Current float value, or None if the KPI does not exist.
        """
        return self._state.get(name)

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, kpi_name: str, value: float) -> None:
        """Store a baseline value for regression comparison.

        Args:
            kpi_name: Name of the KPI.
            value: Reference (baseline) value.
        """
        self._baseline[kpi_name] = float(value)

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    def check_regression(
        self, kpi_name: str, new_value: float
    ) -> Optional[RegressionAlert]:
        """Compare new_value against the stored baseline for kpi_name.

        Returns a RegressionAlert if the drop exceeds the WARNING threshold,
        or None if no significant regression is detected.

        Args:
            kpi_name: Name of the KPI to check.
            new_value: Newly observed KPI value.

        Returns:
            RegressionAlert if a regression is detected, else None.
        """
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
        """Return True if this instance exposes a method with the given name.

        Args:
            name: Method name to check for.
        """
        return callable(getattr(self, name, None))
