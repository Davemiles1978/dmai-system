"""Background loop for the purchase-approval gate (PR L).

Runs :meth:`PurchaseGateMonitor.check_once` every 30 minutes. Mirrors the
idempotent-bootstrap + cadence-gate pattern of
:mod:`components.procurement.loop`. :meth:`PurchaseGateMonitorLoop.force_check`
bypasses the cadence for on-demand checks (tests / admin).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from components.purchase_gate import config as cfg
from components.purchase_gate.monitor import PurchaseGateMonitor

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 300  # thread wake cadence
RUN_INTERVAL_SECONDS = cfg.PROPOSAL_POLL_INTERVAL_SECONDS  # 30 min between checks


class PurchaseGateMonitorLoop:
    def __init__(self, *,
                 purchase_db_path: Optional[str] = None,
                 procurement_db_path: Optional[str] = None,
                 treasury_db_path: Optional[str] = None,
                 poll_seconds: int = DEFAULT_POLL_SECONDS,
                 run_interval_seconds: int = RUN_INTERVAL_SECONDS) -> None:
        self._monitor = PurchaseGateMonitor(
            purchase_db_path=purchase_db_path,
            procurement_db_path=procurement_db_path,
            treasury_db_path=treasury_db_path,
        )
        self._poll = int(poll_seconds)
        self._interval = int(run_interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_run_monotonic: Optional[float] = None
        self.last_result: Dict[str, Any] = {}

    @property
    def monitor(self) -> PurchaseGateMonitor:
        return self._monitor

    def _do_check(self) -> Dict[str, Any]:
        result = self._monitor.check_once()
        self.last_result = result
        self._last_run_monotonic = time.monotonic()
        return result

    def _due(self) -> bool:
        if self._last_run_monotonic is None:
            return True
        return (time.monotonic() - self._last_run_monotonic) >= self._interval

    def force_check(self) -> Dict[str, Any]:
        try:
            return self._do_check()
        except Exception as e:
            logger.exception("purchase_gate_loop: force_check failed: %s", e)
            self.last_result = {"triggered": False, "error": str(e)}
            return self.last_result

    def next_check_ts(self) -> Optional[str]:
        if self._monitor.last_check_ts is None:
            return None
        try:
            last = datetime.fromisoformat(self._monitor.last_check_ts)
        except (TypeError, ValueError):
            return None
        return (last + timedelta(seconds=self._interval)).isoformat()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._due():
                    self._do_check()
            except Exception as e:
                logger.exception("purchase_gate_loop: run failed: %s", e)
                self.last_result = {"triggered": False, "error": str(e)}
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="purchase-gate-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[PurchaseGateMonitorLoop] = None


def start_purchase_gate_loop(*,
                             purchase_db_path: Optional[str] = None,
                             procurement_db_path: Optional[str] = None,
                             treasury_db_path: Optional[str] = None,
                             poll_seconds: int = DEFAULT_POLL_SECONDS,
                             run_interval_seconds: int = RUN_INTERVAL_SECONDS,
                             ) -> PurchaseGateMonitorLoop:
    """Idempotent bootstrap."""
    global _LOOP
    if _LOOP is not None and _LOOP.is_running():
        return _LOOP
    loop = PurchaseGateMonitorLoop(
        purchase_db_path=purchase_db_path,
        procurement_db_path=procurement_db_path,
        treasury_db_path=treasury_db_path,
        poll_seconds=poll_seconds,
        run_interval_seconds=run_interval_seconds,
    )
    loop.start()
    _LOOP = loop
    return loop


__all__ = [
    "PurchaseGateMonitorLoop",
    "start_purchase_gate_loop",
    "DEFAULT_POLL_SECONDS",
    "RUN_INTERVAL_SECONDS",
]
