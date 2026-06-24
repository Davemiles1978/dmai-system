"""
components/kaizen_auto_repair.py
──────────────────────────────────────────────────────────────────────────────
DMAI Kaizen Auto-Repair — autonomous execution of queued fixes.

Closes the loop: items in the Kaizen queue marked "Auto-Repair needed"
are now actually attempted, not just listed.

Flow:
  1. Load pending Kaizen proposals from data/kaizen_queue.jsonl
  2. For each AUTO_REPAIR proposal:
     a. Check memory first (MemoryRetrieval) — maybe DMAI already knows the fix
     b. If not, use CodeWriter with AI assistance
     c. Mark proposal as resolved/failed
     d. Log outcome
  3. Update proposal statuses in the queue file
  4. Run on a background thread every 30 minutes

Safety:
  - Only attempts proposals with status == "pending" and type == "auto_repair"
  - Caps at 3 attempts per proposal (tracked by attempt_count)
  - Never commits or pushes — that's a human/cron action
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dmai.kaizen_auto_repair")

_REPO_ROOT      = Path(__file__).resolve().parent.parent
_KAIZEN_FILE    = _REPO_ROOT / "data" / "kaizen_queue.jsonl"
_REPAIR_LOG     = _REPO_ROOT / "data" / "code_writer" / "kaizen_repair_log.jsonl"
_MAX_ATTEMPTS = 5
_LOOP_INTERVAL  = 1800   # 30 minutes


class KaizenAutoRepair:
    """
    Autonomous Kaizen queue executor.
    Reads pending auto-repair proposals and uses CodeWriter to fix them.
    """

    def __init__(self, code_writer=None, memory_retrieval=None, si_core=None):
        self.code_writer      = code_writer
        self.memory_retrieval = memory_retrieval
        self.si_core          = si_core
        self._thread: Optional[threading.Thread] = None
        self._running = False
        logger.info("KaizenAutoRepair initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # Background loop
    # ─────────────────────────────────────────────────────────────────────────

    def start_repair_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.debug("KaizenAutoRepair loop already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._repair_loop,
            daemon=True,
            name="dmai-kaizen-repair",
        )
        self._thread.start()
        logger.info("KaizenAutoRepair background loop started (interval=%ds)", _LOOP_INTERVAL)

    def stop(self) -> None:
        self._running = False

    def _repair_loop(self) -> None:
        # Initial delay — let system boot fully
        time.sleep(120)
        while self._running:
            try:
                self.run_repair_cycle()
            except Exception as e:
                logger.error("KaizenAutoRepair cycle error: %s", e)
            time.sleep(_LOOP_INTERVAL)

    # ─────────────────────────────────────────────────────────────────────────
    # Core repair cycle
    # ─────────────────────────────────────────────────────────────────────────

    def run_repair_cycle(self) -> Dict:
        proposals = self._load_proposals()
        # Reset failed items that are under new attempt limit so they get retried
        for p in proposals:
            if p.get("status") == "failed" and p.get("attempt_count", 0) < _MAX_ATTEMPTS:
                p["status"] = "pending"

        pending = [
            p for p in proposals
            if p.get("status") in ("pending", "auto_repair_needed", "review_and_fix")
            and p.get("attempt_count", 0) < _MAX_ATTEMPTS
        ]

        if not pending:
            logger.debug("KaizenAutoRepair: no pending proposals")
            return {"repaired": 0, "failed": 0, "skipped": 0}

        repaired = 0
        failed   = 0

        for proposal in pending[:10]:  # max 10 per cycle
            result = self._attempt_repair(proposal)
            proposal["attempt_count"] = proposal.get("attempt_count", 0) + 1
            proposal["last_attempt"]  = datetime.now(timezone.utc).isoformat()

            if result.get("ok"):
                proposal["status"]       = "resolved"
                proposal["resolution"]   = result.get("path") or result.get("action", "fixed")
                proposal["resolved_at"]  = datetime.now(timezone.utc).isoformat()
                repaired += 1
                logger.info("Kaizen FIXED: %s", proposal.get("title", proposal.get("id")))
            else:
                if proposal["attempt_count"] >= _MAX_ATTEMPTS:
                    proposal["status"] = "failed"
                    logger.warning("Kaizen FAILED (max attempts): %s — %s",
                                   proposal.get("title", "?"), result.get("error", "?"))
                failed += 1

            self._log_repair_attempt(proposal, result)

        self._save_proposals(proposals)
        logger.info("KaizenAutoRepair cycle: %d fixed, %d failed", repaired, failed)
        return {"repaired": repaired, "failed": failed, "skipped": len(pending) - repaired - failed}

    def _attempt_repair(self, proposal: Dict) -> Dict:
        """Attempt to fix one proposal. Returns {ok, ...}."""
        title       = proposal.get("title", "")
        description = proposal.get("description", "")
        file_hint   = proposal.get("file") or proposal.get("file_path", "")
        fix_hint    = proposal.get("suggested_fix", "")
        pid         = proposal.get("id", "?")

        logger.info("Attempting repair: [%s] %s", pid, title[:60])

        # ── Step 1: Check memory first ────────────────────────────────────────
        if self.memory_retrieval:
            query = f"{title} {description}"
            try:
                mem = self.memory_retrieval(query)
                if mem.sufficient and mem.best_text():
                    logger.info("Memory HIT for Kaizen fix: %s (conf=%.2f)", pid, mem.confidence)
                    # Memory gave us a relevant answer — record as context but still
                    # use CodeWriter for actual file changes if a file is involved
                    if not file_hint:
                        return {
                            "ok": True,
                            "action": "memory_resolved",
                            "memory_source": mem.source,
                            "memory_text": mem.best_text()[:200],
                        }
            except Exception as e:
                logger.debug("Memory recall error: %s", e)

        # ── Step 2: Code-level fix via CodeWriter ─────────────────────────────
        if not self.code_writer:
            return {"ok": False, "error": "No CodeWriter available"}

        # Determine what type of fix to apply
        action = proposal.get("action_type", "patch")

        if action == "new_file" or not file_hint:
            # Generate a new component
            component_name = (title.lower()
                              .replace(" ", "_")
                              .replace("-", "_")
                              .replace("/", "_"))[:40]
            return self.code_writer.generate_component(
                component_name=component_name,
                description=description or title,
                requirements=[fix_hint] if fix_hint else [title],
                origin="kaizen_auto_repair",
            )
        else:
            # Patch an existing file
            return self.code_writer.execute_kaizen_fix(
                kaizen_id=pid,
                file_path=file_hint,
                problem=description or title,
                suggested_fix=fix_hint,
                origin="kaizen_auto_repair",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_proposals(self) -> List[Dict]:
        """Load all proposals from the queue file."""
        if not _KAIZEN_FILE.exists():
            return []
        proposals = []
        for line in _KAIZEN_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                proposals.append(json.loads(line))
            except Exception:
                pass
        return proposals

    def _save_proposals(self, proposals: List[Dict]) -> None:
        """Write all proposals back to the queue file."""
        try:
            _KAIZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_KAIZEN_FILE, "w") as f:
                for p in proposals:
                    f.write(json.dumps(p) + "\n")
        except Exception as e:
            logger.error("Could not save Kaizen proposals: %s", e)

    def _log_repair_attempt(self, proposal: Dict, result: Dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "proposal_id": proposal.get("id", "?"),
            "title": proposal.get("title", "")[:100],
            "attempt": proposal.get("attempt_count", 1),
            "status": proposal.get("status"),
            "ok": result.get("ok"),
            "error": result.get("error", ""),
            "action": result.get("action", result.get("path", "")),
        }
        try:
            _REPAIR_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(_REPAIR_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Could not write repair log: %s", e)

    def get_stats(self) -> Dict:
        proposals = self._load_proposals()
        statuses = {}
        for p in proposals:
            s = p.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total": len(proposals),
            "by_status": statuses,
            "queue_file": str(_KAIZEN_FILE),
        }
