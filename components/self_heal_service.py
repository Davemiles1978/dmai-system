"""Layer 4 chunk L4-6 — SelfHealService scaffold.

In-process replacement for the external ``a5bad513`` watchdog cron. Runs a
background daemon thread that walks a 6-step probe → repair → notify cycle
on a sleep interval.

This chunk ships the **scaffold only**: steps 1 (health probe), 5 (notify
via logging), and 6 (sleep) are implemented; steps 2-4 (gap scan, repair
tick, layer-4 tick) are no-op stubs that L4-9 / L4-10 will fill in.

Cron deletion gate (informational only — not enforced by code):
    1. ``status()["running"] == True`` in production.
    2. ≥7 days of continuous uptime with no halt events.
    3. User explicitly approves the deletion.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger("self_heal_service")


class SelfHealService:
    """In-process watchdog daemon."""

    DEFAULT_INTERVAL_SECONDS = 1800  # 30 minutes
    DEFAULT_HALT_THRESHOLD = 5       # consecutive failures → halt
    DEFAULT_HALT_HOURS = 4

    def __init__(
        self,
        app=None,
        data_path: str = "data",
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        halt_threshold: Optional[int] = None,
        halt_hours: Optional[int] = None,
    ) -> None:
        self.app = app
        self.data_path = data_path
        self.interval_seconds = max(60, int(interval_seconds))
        self.halt_threshold = int(
            halt_threshold
            if halt_threshold is not None
            else os.environ.get("SELF_HEAL_HALT_THRESHOLD", self.DEFAULT_HALT_THRESHOLD)
        )
        self.halt_hours = int(
            halt_hours
            if halt_hours is not None
            else os.environ.get("SELF_HEAL_HALT_HOURS", self.DEFAULT_HALT_HOURS)
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_probe: Optional[str] = None
        self._last_repair: Optional[str] = None
        self._last_repair_summary: Optional[Dict[str, Any]] = None
        self._last_layer4_summary: Optional[Dict[str, Any]] = None
        self._halt_until: Optional[str] = None
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background thread. Idempotent."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="self_heal_service",
            )
            self._thread.start()
            logger.info(
                "SelfHealService started (interval=%ds)", self.interval_seconds
            )

    def stop(self) -> None:
        """Signal the loop to exit. Loop will exit within one interval."""
        self._running = False
        self._stop_event.set()

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "last_probe": self._last_probe,
            "last_repair": self._last_repair,
            "last_repair_summary": self._last_repair_summary,
            "last_layer4_summary": self._last_layer4_summary,
            "halt_until": self._halt_until,
            "consecutive_failures": self._consecutive_failures,
            "halt_threshold": self.halt_threshold,
            "halt_hours": self.halt_hours,
            "interval_seconds": self.interval_seconds,
        }

    # ------------------------------------------------------------------
    # Halt helpers
    # ------------------------------------------------------------------

    def halt_until(self, iso_ts: str) -> None:
        """Set the halt_until barrier; the loop will skip ticks until then."""
        self._halt_until = iso_ts

    def clear_halt(self) -> None:
        self._halt_until = None

    def _should_halt(self) -> bool:
        if not self._halt_until:
            return False
        try:
            return datetime.now(timezone.utc).isoformat() < self._halt_until
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """6-step cycle. Sleeps in 1-second chunks so stop() is responsive."""
        while self._running:
            try:
                if not self._should_halt():
                    self._tick()
            except Exception as e:  # noqa: BLE001
                self._consecutive_failures += 1
                logger.warning("self_heal tick error: %s", e)
                self._log_event("tick_error", {"error": repr(e)})

            # Sleep in 1s slices so stop() reacts within ~1s.
            slept = 0
            while slept < self.interval_seconds and self._running:
                if self._stop_event.wait(timeout=1.0):
                    break
                slept += 1

        logger.info("SelfHealService loop exited cleanly")

    def _tick(self) -> None:
        """One probe → scan → repair → layer4 → notify cycle."""
        ok = self._probe_health()
        now_iso = datetime.now(timezone.utc).isoformat()
        self._last_probe = now_iso

        gap_summary = self._gap_scan_stub()
        repair_summary = self._repair_tick_stub()
        l4_summary = self._layer4_tick_stub()

        if ok:
            self._consecutive_failures = 0
            self._log_event("tick_ok", {
                "ts": now_iso,
                "gap": bool(gap_summary),
                "repair": repair_summary,
                "layer4": l4_summary,
            })
        else:
            self._consecutive_failures += 1
            self._log_event("tick_unhealthy", {
                "ts": now_iso,
                "consecutive_failures": self._consecutive_failures,
            })
            if self._consecutive_failures >= self.halt_threshold:
                halt_iso = (
                    datetime.now(timezone.utc) + timedelta(hours=self.halt_hours)
                ).isoformat()
                self._halt_until = halt_iso
                self._notify_halt(self._consecutive_failures, halt_iso)
                self._log_event("halt_triggered", {
                    "ts": now_iso,
                    "consecutive_failures": self._consecutive_failures,
                    "halt_until": halt_iso,
                })

    def _notify_halt(self, failures: int, halt_iso: str) -> None:
        """Best-effort Slack notification on halt; durable record in JSONL log."""
        msg = (
            f":rotating_light: DMAI in-app SelfHealService HALTED "
            f"after {failures} consecutive unhealthy ticks. "
            f"halt_until={halt_iso}."
        )
        logger.error(msg)
        try:
            import importlib
            slack_mod = importlib.import_module("components.slack_notifier")
            notifier_cls = getattr(slack_mod, "SlackNotifier", None)
            if notifier_cls:
                notifier_cls().post(msg)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Step implementations (1 & 5 real; 2-4 stubs for L4-10)
    # ------------------------------------------------------------------

    def _probe_health(self) -> bool:
        """Step 1 — call /health via Flask test_client when app is available."""
        if self.app is None:
            return False
        try:
            client = self.app.test_client()
            resp = client.get("/health")
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _gap_scan_stub(self) -> Optional[Dict[str, Any]]:
        """Step 2 — SelfScanner.run() (best-effort, never crashes the loop)."""
        try:
            from components.self_scanner import SelfScanner
            scanner = SelfScanner(data_path=self.data_path)
            result = scanner.run()
            return result if isinstance(result, dict) else {"ok": True}
        except Exception as e:  # noqa: BLE001
            self._log_event("gap_scan_error", {"error": repr(e)})
            return None

    def _repair_tick_stub(self) -> Optional[Dict[str, Any]]:
        """Step 3 — SelfRepairOrchestrator.run_once(auto_approve=True).

        Returns a JSON-safe summary on success; None on failure. Updates
        ``self._last_repair`` and ``self._last_repair_summary``.
        """
        try:
            from components.self_repair_orchestrator import SelfRepairOrchestrator
            orch = SelfRepairOrchestrator(repo_root=".")
            summary = orch.run_once(auto_approve=True)
            payload = {
                "matched_patterns": list(getattr(summary, "matched_patterns", []) or []),
                "enqueued": len(getattr(summary, "enqueued_edit_ids", []) or []),
                "auto_approved": len(getattr(summary, "auto_approved_edit_ids", []) or []),
            }
            self._last_repair = datetime.now(timezone.utc).isoformat()
            self._last_repair_summary = payload
            return payload
        except Exception as e:  # noqa: BLE001
            self._log_event("repair_tick_error", {"error": repr(e)})
            return None

    def _layer4_tick_stub(self) -> Optional[Dict[str, Any]]:
        """Step 4 — SelfGenOrchestrator.run_once(), no-op until L4-9 ships.

        Lazy import so a missing SelfGenOrchestrator is a graceful skip,
        not a tick failure. L4-9 will create the module; this hook is ready.
        """
        try:
            from components.self_gen_orchestrator import SelfGenOrchestrator  # noqa: F401
        except Exception:
            return None
        try:
            from components.self_gen_orchestrator import SelfGenOrchestrator
            orch = SelfGenOrchestrator(data_path=self.data_path)
            result = orch.run_once()
            payload = result if isinstance(result, dict) else {"ok": True}
            self._last_layer4_summary = payload
            return payload
        except Exception as e:  # noqa: BLE001
            self._log_event("layer4_tick_error", {"error": repr(e)})
            return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        log_dir = Path(self.data_path) / "self_healing"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "self_heal_service.log.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    **payload,
                }) + "\n")
        except OSError:
            pass
