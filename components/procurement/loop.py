"""Background loop for the procurement research skill (PR K).

Runs :func:`components.procurement.researcher.run_research` on a slow
cadence (every 6 hours). Mirrors the idempotent-bootstrap pattern used by
:mod:`components.treasury.treasury_loop` and
:mod:`components.workload.workload_loop`, with an added cadence gate so a
tight ``poll_seconds`` (used in tests) still only actually researches once
per :data:`RUN_INTERVAL_SECONDS`. :meth:`ProcurementLoop.force_run`
bypasses the cadence for on-demand runs (admin endpoint / tests).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from components.procurement import researcher

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600            # how often the thread wakes to check
RUN_INTERVAL_SECONDS = 6 * 60 * 60    # 6 hours between actual research runs


class ProcurementLoop:
    def __init__(self, *,
                 procurement_db_path: Optional[str] = None,
                 workload_db_path: Optional[str] = None,
                 treasury_db_path: Optional[str] = None,
                 poll_seconds: int = DEFAULT_POLL_SECONDS,
                 run_interval_seconds: int = RUN_INTERVAL_SECONDS):
        self._procurement_db_path = procurement_db_path
        self._workload_db_path = workload_db_path
        self._treasury_db_path = treasury_db_path
        self._poll = int(poll_seconds)
        self._interval = int(run_interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_run_monotonic: Optional[float] = None
        self.last_summary: Dict[str, Any] = {}

    # ── the actual research call ─────────────────────────────────────────────

    def _do_run(self) -> Dict[str, Any]:
        summary = researcher.run_research(
            procurement_db_path=self._procurement_db_path,
            workload_db_path=self._workload_db_path,
            treasury_db_path=self._treasury_db_path,
        )
        self.last_summary = summary
        self._last_run_monotonic = time.monotonic()
        return summary

    def _due(self) -> bool:
        if self._last_run_monotonic is None:
            return True
        return (time.monotonic() - self._last_run_monotonic) >= self._interval

    def force_run(self) -> Dict[str, Any]:
        """Run research now, ignoring the cadence gate."""
        try:
            return self._do_run()
        except Exception as e:
            logger.exception("procurement_loop: force_run failed: %s", e)
            self.last_summary = {"ok": False, "error": str(e)}
            return self.last_summary

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._due():
                    self._do_run()
            except Exception as e:
                logger.exception("procurement_loop: run failed: %s", e)
                self.last_summary = {"ok": False, "error": str(e)}
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="procurement-loop",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[ProcurementLoop] = None


def start_procurement_loop(*,
                           procurement_db_path: Optional[str] = None,
                           workload_db_path: Optional[str] = None,
                           treasury_db_path: Optional[str] = None,
                           poll_seconds: int = DEFAULT_POLL_SECONDS,
                           run_interval_seconds: int = RUN_INTERVAL_SECONDS,
                           ) -> ProcurementLoop:
    """Idempotent bootstrap."""
    global _LOOP
    live = (_LOOP is not None
            and getattr(_LOOP, "_thread", None) is not None
            and _LOOP._thread.is_alive())
    if live:
        return _LOOP  # type: ignore[return-value]
    loop = ProcurementLoop(
        procurement_db_path=procurement_db_path,
        workload_db_path=workload_db_path,
        treasury_db_path=treasury_db_path,
        poll_seconds=poll_seconds,
        run_interval_seconds=run_interval_seconds,
    )
    loop.start()
    _LOOP = loop
    return loop


__all__ = [
    "ProcurementLoop",
    "start_procurement_loop",
    "DEFAULT_POLL_SECONDS",
    "RUN_INTERVAL_SECONDS",
]
