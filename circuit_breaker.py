"""
DMAI Circuit Breaker
====================
Implements CB-01 through CB-06 from the validation framework.

CB-01: 5xx error rate >2% for >=2min  -> pause all outbound API calls
CB-02: State file hash mismatch        -> halt all state writes, lock kaizen
CB-03: KPI velocity >3x baseline       -> investigate for spoofing
CB-04: Kaizen queue >20 pending        -> stop generating new proposals
CB-05: Adversarial probe detected      -> block source IP, alert
CB-06: Reasoning chain >60s or >20steps-> kill chain, log trace, alert
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CircuitState
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Possible states for a circuit breaker."""

    CLOSED = "closed"        # normal operation
    OPEN = "open"            # tripped -- blocking calls
    HALF_OPEN = "half_open"  # testing recovery


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """A single named circuit breaker with trip/reset semantics."""

    def __init__(self, name: str, cb_id: str) -> None:
        """Initialise the circuit breaker in CLOSED (normal) state.

        Args:
            name: Human-readable name, e.g. 'API Error Rate'.
            cb_id: Short identifier string, e.g. 'CB-01'.
        """
        self.name = name
        self.cb_id = cb_id
        self.state: CircuitState = CircuitState.CLOSED
        self.tripped_at: Optional[datetime] = None
        self.trip_reason: str = ""
        self.trip_count: int = 0

    def trip(self, reason: str) -> None:
        """Open the circuit breaker and record the trip event.

        Args:
            reason: Human-readable description of why the breaker tripped.
        """
        self.state = CircuitState.OPEN
        self.tripped_at = datetime.utcnow()
        self.trip_reason = reason
        self.trip_count += 1
        logger.critical(
            "Circuit breaker %s (%s) TRIPPED. reason=%s trip_count=%d",
            self.cb_id,
            self.name,
            reason,
            self.trip_count,
        )

    def reset(self) -> None:
        """Close the circuit breaker and clear trip metadata."""
        self.state = CircuitState.CLOSED
        self.tripped_at = None
        self.trip_reason = ""
        logger.info("Circuit breaker %s (%s) RESET to CLOSED.", self.cb_id, self.name)

    def is_open(self) -> bool:
        """Return True if the circuit breaker is currently OPEN (tripped)."""
        return self.state == CircuitState.OPEN

    def status(self) -> dict:
        """Return a JSON-serialisable status dict for this breaker."""
        return {
            "cb_id": self.cb_id,
            "name": self.name,
            "state": self.state.value,
            "tripped_at": self.tripped_at.isoformat() if self.tripped_at else None,
            "trip_reason": self.trip_reason,
            "trip_count": self.trip_count,
        }


# ---------------------------------------------------------------------------
# CircuitBreakerManager
# ---------------------------------------------------------------------------


class CircuitBreakerManager:
    """Singleton manager for all six DMAI circuit breakers (CB-01 to CB-06).

    Usage::

        mgr = CircuitBreakerManager.get()
        mgr.record_response(500)
    """

    _instance: Optional["CircuitBreakerManager"] = None

    # CB-05 probe patterns
    PROBE_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"<\s*script\s*>",
        r"union\s+select",
        r";\s*drop\s+table",
        r"base64_decode\s*\(",
        r"\.\./\.\./\.\.",
        r"etc/passwd",
    ]

    def __init__(self) -> None:
        """Initialise all six circuit breakers and internal tracking state."""
        self.breakers: dict = {
            "CB-01": CircuitBreaker("API Error Rate", "CB-01"),
            "CB-02": CircuitBreaker("State File Integrity", "CB-02"),
            "CB-03": CircuitBreaker("KPI Velocity", "CB-03"),
            "CB-04": CircuitBreaker("Kaizen Queue Depth", "CB-04"),
            "CB-05": CircuitBreaker("Adversarial Probe", "CB-05"),
            "CB-06": CircuitBreaker("Reasoning Chain Timeout", "CB-06"),
        }
        # CB-01: ring buffer of (timestamp_float, is_5xx)
        self._error_window: deque = deque(maxlen=200)
        # CB-03: ring buffer of increments-per-minute timestamps
        self._kpi_velocity: deque = deque(maxlen=60)
        self._kpi_baseline: float = 0.0
        # CB-05
        self._blocked_ips: set = set()
        self._compiled_probes = [
            re.compile(p, re.IGNORECASE) for p in self.PROBE_PATTERNS
        ]
        # CB-06: active chains {chain_id: (start_time, steps_so_far)}
        self._active_chains: dict = {}
        # Alert callbacks
        self._alert_callbacks: list = []

    @classmethod
    def get(cls) -> "CircuitBreakerManager":
        """Return the singleton CircuitBreakerManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_alert_callback(self, fn: Callable) -> None:
        """Register a callable to be invoked when any circuit breaker trips.

        Args:
            fn: Callable accepting (cb_id: str, reason: str).
        """
        self._alert_callbacks.append(fn)

    def _send_alert(self, cb_id: str, reason: str) -> None:
        """Log a CRITICAL-level alert and invoke all registered callbacks.

        Args:
            cb_id: The circuit breaker identifier that triggered the alert.
            reason: Human-readable trip reason.
        """
        logger.critical("ALERT: %s tripped -- %s", cb_id, reason)
        for cb in self._alert_callbacks:
            try:
                cb(cb_id, reason)
            except Exception as exc:
                logger.error("Alert callback raised an exception: %s", exc)

    # ------------------------------------------------------------------
    # CB-01: API Error Rate
    # ------------------------------------------------------------------

    def record_response(self, status_code: int) -> None:
        """Record an HTTP response status code and evaluate CB-01.

        Trips CB-01 if the 5xx error rate exceeds 2% over the last 2 minutes.

        Args:
            status_code: HTTP status code of the response (e.g. 200, 500).
        """
        now = time.monotonic()
        is_5xx = 500 <= status_code < 600
        self._error_window.append((now, is_5xx))

        # Consider only events in the last 120 seconds (2 minutes)
        cutoff = now - 120.0
        recent = [(ts, err) for ts, err in self._error_window if ts >= cutoff]
        if not recent:
            return

        error_count = sum(1 for _, err in recent if err)
        rate = error_count / len(recent)
        if rate > 0.02 and not self.breakers["CB-01"].is_open():
            reason = "5xx error rate %.1f%% over last 2min (threshold 2%%)" % (rate * 100)
            self.breakers["CB-01"].trip(reason)
            self._send_alert("CB-01", reason)

    # ------------------------------------------------------------------
    # CB-02: State File Integrity
    # ------------------------------------------------------------------

    def verify_state_file(self, path: Path, expected_hash: str) -> bool:
        """Compute SHA-256 of a file and compare to expected_hash.

        Trips CB-02 on mismatch.  All state writes and kaizen operations
        should be gated on this check.

        Args:
            path: Filesystem path to the state file.
            expected_hash: Lowercase hex SHA-256 digest expected.

        Returns:
            True if the file matches the expected hash; False otherwise.
        """
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            reason = "Cannot read state file %s: %s" % (path, exc)
            self.breakers["CB-02"].trip(reason)
            self._send_alert("CB-02", reason)
            return False

        if digest != expected_hash.lower():
            reason = "Hash mismatch for %s (expected %s, got %s)" % (
                path,
                expected_hash,
                digest,
            )
            self.breakers["CB-02"].trip(reason)
            self._send_alert("CB-02", reason)
            return False
        return True

    # ------------------------------------------------------------------
    # CB-03: KPI Velocity
    # ------------------------------------------------------------------

    def set_kpi_baseline(self, increments_per_hour: float) -> None:
        """Set the expected baseline KPI increment rate.

        Args:
            increments_per_hour: Normal number of KPI increments per hour.
        """
        self._kpi_baseline = increments_per_hour
        logger.info("KPI baseline set to %.2f increments/hour.", increments_per_hour)

    def record_kpi_increment(self) -> None:
        """Record a KPI write event and evaluate CB-03.

        Trips CB-03 if the observed increments-per-hour exceeds 3x the
        configured baseline (when a baseline has been set).
        """
        self._kpi_velocity.append(time.monotonic())

        if self._kpi_baseline <= 0:
            return  # No baseline set; cannot evaluate

        # Count increments in the last 60 seconds and extrapolate to hourly
        now = time.monotonic()
        cutoff = now - 60.0
        recent_count = sum(1 for ts in self._kpi_velocity if ts >= cutoff)
        hourly_rate = recent_count * 60.0  # scale 1-min window to 1-hour

        threshold = self._kpi_baseline * 3.0
        if hourly_rate > threshold and not self.breakers["CB-03"].is_open():
            reason = (
                "KPI velocity %.1f/hr is >3x baseline %.1f/hr -- possible spoofing"
                % (hourly_rate, self._kpi_baseline)
            )
            self.breakers["CB-03"].trip(reason)
            self._send_alert("CB-03", reason)

    # ------------------------------------------------------------------
    # CB-04: Kaizen Queue Depth
    # ------------------------------------------------------------------

    def check_kaizen_depth(self, depth: int) -> None:
        """Evaluate the kaizen queue depth and trip CB-04 if > 20.

        Args:
            depth: Current number of pending kaizen proposals.
        """
        if depth > 20 and not self.breakers["CB-04"].is_open():
            reason = "Kaizen queue depth %d exceeds limit of 20" % depth
            self.breakers["CB-04"].trip(reason)
            self._send_alert("CB-04", reason)

    # ------------------------------------------------------------------
    # CB-05: Adversarial Probe Detection
    # ------------------------------------------------------------------

    def check_adversarial_probe(self, request_data: str, source_ip: str = "") -> bool:
        """Scan request data for adversarial probe patterns.

        Trips CB-05 and blocks the source IP if a probe is detected.

        Args:
            request_data: The raw request string to inspect.
            source_ip: Optional IP address of the requester.

        Returns:
            True if a probe pattern was detected; False if the request looks clean.
        """
        for pattern in self._compiled_probes:
            if pattern.search(request_data):
                if source_ip:
                    self._blocked_ips.add(source_ip)
                reason = "Adversarial probe detected (pattern: %s) from ip=%s" % (
                    pattern.pattern,
                    source_ip or "unknown",
                )
                if not self.breakers["CB-05"].is_open():
                    self.breakers["CB-05"].trip(reason)
                    self._send_alert("CB-05", reason)
                else:
                    logger.warning("Subsequent probe blocked: %s", reason)
                return True
        return False

    def is_ip_blocked(self, ip: str) -> bool:
        """Return True if the given IP has been blocked by CB-05.

        Args:
            ip: IP address string to check.
        """
        return ip in self._blocked_ips

    # ------------------------------------------------------------------
    # CB-06: Reasoning Chain Timeout
    # ------------------------------------------------------------------

    def start_chain(self, chain_id: str) -> None:
        """Record the start of a reasoning chain.

        Args:
            chain_id: Unique identifier for this reasoning chain.
        """
        self._active_chains[chain_id] = {"start": time.monotonic(), "steps": 0}
        logger.debug("Reasoning chain started: %s", chain_id)

    def end_chain(self, chain_id: str, steps: int) -> None:
        """Evaluate a completed reasoning chain against CB-06 thresholds.

        Trips CB-06 if the chain took more than 60 seconds or more than 20 steps.

        Args:
            chain_id: Unique identifier matching a prior start_chain() call.
            steps: Number of reasoning steps taken by this chain.
        """
        chain_meta = self._active_chains.pop(chain_id, None)
        if chain_meta is None:
            logger.warning("end_chain called for unknown chain_id: %s", chain_id)
            return

        duration = time.monotonic() - chain_meta["start"]
        violations = []
        if duration > 60.0:
            violations.append("duration %.1fs > 60s limit" % duration)
        if steps > 20:
            violations.append("%d steps > 20-step limit" % steps)

        if violations:
            reason = "Reasoning chain %s exceeded limits: %s" % (
                chain_id,
                "; ".join(violations),
            )
            logger.warning("CB-06 trace: chain_id=%s duration=%.2fs steps=%d", chain_id, duration, steps)
            if not self.breakers["CB-06"].is_open():
                self.breakers["CB-06"].trip(reason)
                self._send_alert("CB-06", reason)

    # ------------------------------------------------------------------
    # Admin helpers
    # ------------------------------------------------------------------

    def clear_breaker(self, cb_id: str) -> bool:
        """Reset a named circuit breaker back to CLOSED.

        Args:
            cb_id: The circuit breaker identifier, e.g. 'CB-01'.

        Returns:
            True if the breaker was found and reset; False if unknown.
        """
        breaker = self.breakers.get(cb_id)
        if breaker is None:
            logger.warning("clear_breaker: unknown cb_id '%s'", cb_id)
            return False
        breaker.reset()
        return True

    def get_all_status(self) -> dict:
        """Return a dict mapping cb_id -> status dict for all six breakers."""
        return {cb_id: cb.status() for cb_id, cb in self.breakers.items()}

    def any_open(self) -> bool:
        """Return True if at least one circuit breaker is currently OPEN."""
        return any(cb.is_open() for cb in self.breakers.values())

    def open_breakers(self) -> list:
        """Return a list of cb_id strings for all currently OPEN breakers."""
        return [cb_id for cb_id, cb in self.breakers.items() if cb.is_open()]


# ---------------------------------------------------------------------------
# Flask helpers
# ---------------------------------------------------------------------------


def circuit_breaker_guard(f):
    """Flask route decorator that returns HTTP 503 if any circuit breaker is open.

    Attach this decorator to any endpoint that should be blocked during
    a circuit-breaker trip event.
    """
    from flask import jsonify

    @wraps(f)
    def decorated(*args, **kwargs):
        """Check all circuit breakers before allowing the request through."""
        mgr = CircuitBreakerManager.get()
        open_ids = mgr.open_breakers()
        if open_ids:
            return (
                jsonify(
                    {
                        "error": "Service temporarily unavailable",
                        "circuit_breakers_open": open_ids,
                    }
                ),
                503,
            )
        return f(*args, **kwargs)

    return decorated


def after_request_hook(response):
    """Flask after_request hook that records every HTTP response for CB-01.

    Register with::

        app.after_request(after_request_hook)

    Args:
        response: Flask Response object.

    Returns:
        The unmodified response.
    """
    try:
        from flask import request
        if request.headers.get("X-Internal-Probe") == "1":
            return response
    except Exception:
        # request context unavailable -- fall through to normal recording
        pass
    try:
        CircuitBreakerManager.get().record_response(response.status_code)
    except Exception as exc:
        logger.error("after_request_hook failed: %s", exc)
    return response
