"""Background sampler for the workload profiler (PR J).

Every ``poll_seconds`` (600 = 10 minutes in production) it calls
:func:`components.workload.workload_profiler.sample_now`. Idempotent
bootstrap, wired into app startup next to
:mod:`components.treasury.treasury_loop`.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from components.workload import workload_profiler as wp

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600  # 10 minutes


class WorkloadLoop:
    def __init__(self, *,
                 workload_db_path: Optional[str] = None,
                 poll_seconds: int = DEFAULT_POLL_SECONDS):
        self._db_path = workload_db_path
        self._poll = int(poll_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = wp.sample_now(db_path=self._db_path)
                self.last_summary = {
                    "ts":            s.ts,
                    "mem_rss_mb":    s.mem_rss_mb,
                    "cpu_percent":   s.cpu_percent,
                    "thread_count":  s.thread_count,
                    "uptime_hours":  (round(s.uptime_seconds / 3600.0, 2)
                                      if s.uptime_seconds else None),
                }
            except Exception as e:
                logger.exception("workload_loop: sample failed: %s", e)
                self.last_summary = {"error": str(e)}
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="workload-loop",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[WorkloadLoop] = None


def start_workload_loop(*,
                        workload_db_path: Optional[str] = None,
                        poll_seconds: int = DEFAULT_POLL_SECONDS,
                        ) -> WorkloadLoop:
    """Idempotent bootstrap."""
    global _LOOP
    live = (_LOOP is not None
            and getattr(_LOOP, "_thread", None) is not None
            and _LOOP._thread.is_alive())
    if live:
        return _LOOP  # type: ignore[return-value]
    wp.init_workload_db(workload_db_path)
    loop = WorkloadLoop(
        workload_db_path=workload_db_path,
        poll_seconds=poll_seconds,
    )
    loop.start()
    _LOOP = loop
    return loop


__all__ = ["WorkloadLoop", "start_workload_loop", "DEFAULT_POLL_SECONDS"]
