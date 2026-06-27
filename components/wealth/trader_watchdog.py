"""
Watchdog thread for AutonomousTrader.

Runs in the background and ensures the trader's main loop is alive:
  - If the last tick is older than `max_stale_seconds` (default = 2x loop
    interval + 30s grace), force a synchronous tick.
  - If three consecutive forced ticks fail, emit a 'halt' alert and mark
    the trader as unhealthy. /health exposes this so Render can restart.

Adapted from the self-healing cron pattern in the Trader repo, but rebuilt
to run in-process (no external Supabase/pg_cron). Single source of truth.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TraderWatchdog:
    def __init__(self,
                 trader: Any,                       # AutonomousTrader
                 notifier: Optional[Any] = None,
                 check_interval_s: int = 120,
                 max_stale_seconds: Optional[int] = None,
                 failure_threshold: int = 3):
        self.trader = trader
        self.notifier = notifier
        self.check_interval_s = check_interval_s
        self.failure_threshold = failure_threshold
        self.max_stale_seconds = max_stale_seconds or (
            2 * getattr(trader, "loop_interval_s", 300) + 30
        )

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._healthy = True
        self._last_check_ts: Optional[float] = None
        self._last_action: str = "init"

        self._start()

    # ── Public ────────────────────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        return {
            "healthy":             self._healthy,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold":   self.failure_threshold,
            "check_interval_s":    self.check_interval_s,
            "max_stale_seconds":   self.max_stale_seconds,
            "last_check_ts":       self._last_check_ts,
            "last_action":         self._last_action,
        }

    def stop(self) -> None:
        self._stop.set()

    # ── Loop ──────────────────────────────────────────────────────────────────
    def _start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="TraderWatchdog", daemon=True)
        self._thread.start()
        logger.info("TraderWatchdog: started (interval=%ss, stale=%ss)",
                    self.check_interval_s, self.max_stale_seconds)

    def _run(self) -> None:
        time.sleep(30)  # let trader settle
        while not self._stop.is_set():
            try:
                self._check_once()
            except Exception as e:
                logger.exception("TraderWatchdog check failed: %s", e)
                if self.notifier:
                    try:
                        self.notifier.error("watchdog", str(e))
                    except Exception:
                        pass
            slept = 0
            while slept < self.check_interval_s and not self._stop.is_set():
                time.sleep(min(5, self.check_interval_s - slept))
                slept += 5

    def _check_once(self) -> None:
        with self._lock:
            self._last_check_ts = time.time()
            status = self.trader.status()
            enabled = bool(status.get("enabled"))
            last_tick_ts = status.get("last_tick_ts")
            if not enabled:
                self._consecutive_failures = 0
                self._healthy = True
                self._last_action = "skip_disabled"
                return
            stale_s = self._stale_seconds(last_tick_ts)
            if stale_s is None or stale_s <= self.max_stale_seconds:
                self._consecutive_failures = 0
                self._healthy = True
                self._last_action = f"ok ({int(stale_s) if stale_s else 0}s old)"
                return

            # Stale tick — force one synchronously.
            logger.warning("TraderWatchdog: stale tick (%ss old), forcing tick", int(stale_s))
            try:
                self.trader.tick()
                self._consecutive_failures = 0
                self._healthy = True
                self._last_action = f"forced_tick_ok ({int(stale_s)}s stale)"
            except Exception as e:
                self._consecutive_failures += 1
                self._last_action = f"forced_tick_failed: {e}"
                logger.exception("TraderWatchdog: forced tick failed: %s", e)
                if self._consecutive_failures >= self.failure_threshold:
                    self._healthy = False
                    if self.notifier:
                        try:
                            self.notifier.halt(
                                "watchdog_unhealthy",
                                f"{self._consecutive_failures} consecutive forced-tick "
                                f"failures. Last error: {e}",
                            )
                        except Exception:
                            pass

    def _stale_seconds(self, last_tick_ts: Optional[str]) -> Optional[float]:
        if not last_tick_ts:
            return None
        try:
            # AutonomousTrader writes datetime('now') strings: "YYYY-MM-DD HH:MM:SS"
            dt = datetime.strptime(last_tick_ts[:19], "%Y-%m-%d %H:%M:%S")
            return max(0.0, (datetime.utcnow() - dt).total_seconds())
        except Exception:
            return None


def get_watchdog(trader: Any, notifier: Optional[Any] = None) -> TraderWatchdog:
    return TraderWatchdog(trader=trader, notifier=notifier)
