"""Background poller that keeps the treasury ledger in sync.

Runs :func:`components.treasury.treasury_ledger.sync_from_ledger`
on a slow cadence (every 10 minutes by default) so trade closures
and bet settlements show up in the treasury balance without the
caller having to remember to sync manually.

Idempotent bootstrap - if the loop is already running the second
``start_...`` call is a no-op, matching the pattern used by
:mod:`components.seed_capability_promoter` and
:mod:`components.capability_materialiser`.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from components.treasury import treasury_ledger as tl

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600  # 10 minutes


class TreasuryLoop:
    def __init__(self,
                 *,
                 treasury_db_path: Optional[str] = None,
                 ledger_db_path:   Optional[str] = None,
                 poll_seconds:     int  = DEFAULT_POLL_SECONDS):
        self._treasury_db_path = treasury_db_path
        self._ledger_db_path   = ledger_db_path
        self._poll             = int(poll_seconds)
        self._stop             = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_summary: Dict[str, Any] = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                report = tl.sync_from_ledger(
                    treasury_db_path=self._treasury_db_path,
                    ledger_db_path=self._ledger_db_path,
                )
                self.last_summary = report.as_dict()
            except Exception as e:
                logger.exception("treasury_loop: sync failed: %s", e)
                self.last_summary = {"error": str(e)}
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="treasury-loop",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_LOOP: Optional[TreasuryLoop] = None


def start_treasury_loop(*,
                        treasury_db_path: Optional[str] = None,
                        ledger_db_path:   Optional[str] = None,
                        poll_seconds:     int  = DEFAULT_POLL_SECONDS,
                        ) -> TreasuryLoop:
    """Idempotent bootstrap."""
    global _LOOP
    live = (_LOOP is not None
            and getattr(_LOOP, "_thread", None) is not None
            and _LOOP._thread.is_alive())
    if live:
        return _LOOP  # type: ignore[return-value]
    tl.init_treasury_db(treasury_db_path)  # stamp install_ts + FX
    loop = TreasuryLoop(
        treasury_db_path=treasury_db_path,
        ledger_db_path=ledger_db_path,
        poll_seconds=poll_seconds,
    )
    loop.start()
    _LOOP = loop
    return loop


__all__ = ["TreasuryLoop", "start_treasury_loop", "DEFAULT_POLL_SECONDS"]
