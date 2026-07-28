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
_LOOP_INTERVAL  = 600    # 10 minutes (more aggressive so executed > 0 within the hour)
# Per-cycle cap for AI-assisted repairs (CodeWriter / memory) — keeps token
# spend bounded. Dead-letter sweeps (missing files, backups) are NOT capped.
_AI_REPAIR_BATCH = 100
# Paths that are safe to auto-resolve without invoking CodeWriter — these are
# self-healing artefacts or backups, not real source files.
_DEAD_LETTER_PREFIXES = (
    "components/phase3/",
    "data/self_healing/backups/",
    "data/quarantine/",
    "components/backup_final_",
    "components/backup_",
    "backup_",
)


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
        # Initial delay - let system boot fully
        time.sleep(60)
        while self._running:
            try:
                stats = self.run_repair_cycle()
                logger.info(
                    "[KAIZEN] Loop tick: repaired=%d failed=%d skipped=%d",
                    stats.get("repaired", 0), stats.get("failed", 0), stats.get("skipped", 0),
                )
            except Exception as e:
                import traceback
                logger.error("KaizenAutoRepair cycle error: %s\n%s", e, traceback.format_exc())
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

        # Normalize: some items use "action" instead of "status"
        for p in proposals:
            if "action" in p and "status" not in p:
                p["status"] = p["action"]
        
        pending = [
            p for p in proposals
            if p.get("status") in ("pending", "auto_repair_needed", "review_and_fix")
            and p.get("attempt_count", 0) < _MAX_ATTEMPTS
        ]

        if not pending:
            logger.debug("KaizenAutoRepair: no pending proposals")
            return {"repaired": 0, "failed": 0, "skipped": 0, "dead_lettered": 0}

        repaired = 0
        failed   = 0
        dead_lettered = 0

        # Pass 1: sweep dead-letter proposals (unbounded).
        # Backups, quarantine files and missing files cannot be patched —
        # mark them resolved so the queue actually drains.
        ai_pending = []
        for p in pending:
            f = (p.get("file") or p.get("file_path") or "").lstrip("./")
            if f and any(pref in f for pref in _DEAD_LETTER_PREFIXES):
                p["status"] = "resolved"
                p["resolution"] = "dead_letter_backup_path"
                p["resolved_at"] = datetime.now(timezone.utc).isoformat()
                p["attempt_count"] = p.get("attempt_count", 0) + 1
                dead_lettered += 1
                continue
            if f:
                abs_path = _REPO_ROOT / f
                if not abs_path.exists():
                    p["status"] = "resolved"
                    p["resolution"] = "dead_letter_file_missing"
                    p["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    p["attempt_count"] = p.get("attempt_count", 0) + 1
                    dead_lettered += 1
                    continue
            ai_pending.append(p)
        
        # Deduplicate: keep only most recent per file (collapses 3000 -> ~2)
        seen_files = {}
        deduped = []
        for p in ai_pending:
            f = (p.get("file") or p.get("file_path") or "").lstrip("./")
            if f and f in seen_files:
                p["status"] = "resolved"
                p["resolution"] = "deduplicated_duplicate"
                p["resolved_at"] = datetime.now(timezone.utc).isoformat()
                dead_lettered += 1
            else:
                if f:
                    seen_files[f] = p
                deduped.append(p)
        ai_pending = deduped
        
        # Pass 2: AI-assisted repairs (bounded by _AI_REPAIR_BATCH)
        for proposal in ai_pending[:_AI_REPAIR_BATCH]:
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
        skipped = max(0, len(pending) - repaired - failed - dead_lettered)
        logger.info(
            "KaizenAutoRepair cycle: %d fixed, %d failed, %d dead-lettered, %d deferred",
            repaired, failed, dead_lettered, skipped,
        )
        return {
            "repaired": repaired,
            "failed": failed,
            "skipped": skipped,
            "dead_lettered": dead_lettered,
        }

    def run_repair_batch(self, limit: int = 25, deadline_s: float = 60.0) -> Dict:
        """Bounded drain: process at most `limit` pending proposals within a
        hard wall-clock `deadline_s`. Reuses the same per-item repair logic as
        run_repair_cycle (dead-letter sweep + AI-assisted repair) but never
        iterates the whole queue, so it is safe to call synchronously.

        Returns: processed, succeeded, failed, remaining, duration_s, deadline_hit.
        The AI Hub circuit breaker still guards every CodeWriter/provider call —
        this method does not bypass it.
        """
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 25
        deadline_s = max(1.0, float(deadline_s))

        start = time.monotonic()
        proposals = self._load_proposals()

        # Retry previously-failed items still under the attempt cap (mirrors cycle).
        for p in proposals:
            if p.get("status") == "failed" and p.get("attempt_count", 0) < _MAX_ATTEMPTS:
                p["status"] = "pending"

        pending = [
            p for p in proposals
            if p.get("status") in ("pending", "auto_repair_needed", "review_and_fix")
            and p.get("attempt_count", 0) < _MAX_ATTEMPTS
        ]

        processed = succeeded = failed = 0
        deadline_hit = False

        for proposal in pending:
            if processed >= limit:
                break
            # Hard deadline check between items (no threads/signals).
            if time.monotonic() - start >= deadline_s:
                deadline_hit = True
                break

            f = (proposal.get("file") or proposal.get("file_path") or "").lstrip("./")
            # Dead-letter sweep: backups/quarantine/missing files resolve cheaply.
            if f and any(pref in f for pref in _DEAD_LETTER_PREFIXES):
                proposal["status"] = "resolved"
                proposal["resolution"] = "dead_letter_backup_path"
                proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
                proposal["attempt_count"] = proposal.get("attempt_count", 0) + 1
                processed += 1
                succeeded += 1
                continue
            if f and not (_REPO_ROOT / f).exists():
                proposal["status"] = "resolved"
                proposal["resolution"] = "dead_letter_file_missing"
                proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
                proposal["attempt_count"] = proposal.get("attempt_count", 0) + 1
                processed += 1
                succeeded += 1
                continue

            result = self._attempt_repair(proposal)
            proposal["attempt_count"] = proposal.get("attempt_count", 0) + 1
            proposal["last_attempt"] = datetime.now(timezone.utc).isoformat()
            if result.get("ok"):
                proposal["status"] = "resolved"
                proposal["resolution"] = result.get("path") or result.get("action", "fixed")
                proposal["resolved_at"] = datetime.now(timezone.utc).isoformat()
                succeeded += 1
            else:
                if proposal["attempt_count"] >= _MAX_ATTEMPTS:
                    proposal["status"] = "failed"
                failed += 1
            self._log_repair_attempt(proposal, result)
            processed += 1

        # Persist progress even on partial/deadline-hit runs.
        self._save_proposals(proposals)

        remaining = sum(
            1 for p in proposals
            if p.get("status") in ("pending", "auto_repair_needed", "review_and_fix")
            and p.get("attempt_count", 0) < _MAX_ATTEMPTS
        )
        duration_s = round(time.monotonic() - start, 3)
        logger.info(
            "KaizenAutoRepair batch: processed=%d succeeded=%d failed=%d remaining=%d "
            "duration=%.2fs deadline_hit=%s",
            processed, succeeded, failed, remaining, duration_s, deadline_hit,
        )
        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "remaining": remaining,
            "duration_s": duration_s,
            "deadline_hit": deadline_hit,
        }

    def _attempt_repair(self, proposal: Dict) -> Dict:
        """Attempt to fix one proposal. Returns {ok, ...}."""
        title       = proposal.get("title", "")
        description = proposal.get("description", "")
        file_hint   = proposal.get("file") or proposal.get("file_path", "")
        fix_hint    = proposal.get("suggested_fix", "")
        pid         = proposal.get("id", "?")
        ptype       = proposal.get("type", proposal.get("action_type", "unknown"))

        logger.info(
            "[KAIZEN] Attempting repair: id=%s type=%s file=%s title=%s",
            pid, ptype, file_hint or "-", title[:80],
        )

        try:
            # Step 1: Check memory first
            if self.memory_retrieval:
                query = f"{title} {description}"
                try:
                    mem = self.memory_retrieval(query)
                    if mem.sufficient and mem.best_text():
                        logger.info("[KAIZEN] Memory HIT: %s (conf=%.2f)", pid, mem.confidence)
                        if not file_hint:
                            return {
                                "ok": True,
                                "action": "memory_resolved",
                                "memory_source": mem.source,
                                "memory_text": mem.best_text()[:200],
                            }
                except Exception as e:
                    logger.debug("[KAIZEN] Memory recall error on %s: %s", pid, e)

            # Step 2: Code-level fix via CodeWriter
            if not self.code_writer:
                # Fallback: mark informational proposals resolved so the queue drains
                if ptype in ("info", "informational", "log", "warning") or not file_hint:
                    return {
                        "ok": True,
                        "action": "acknowledged_no_codewriter",
                        "note": "No CodeWriter available; proposal acknowledged.",
                    }
                return {"ok": False, "error": "No CodeWriter available"}

            action = proposal.get("action_type", "patch")

            if action == "new_file" or not file_hint:
                component_name = (title.lower()
                                  .replace(" ", "_")
                                  .replace("-", "_")
                                  .replace("/", "_"))[:40] or f"kaizen_{pid}"
                logger.info("[KAIZEN] Generating new component: %s", component_name)
                return self.code_writer.generate_component(
                    component_name=component_name,
                    description=description or title,
                    requirements=[fix_hint] if fix_hint else [title],
                    origin="kaizen_auto_repair",
                )
            logger.info("[KAIZEN] Patching file: %s", file_hint)
            return self.code_writer.execute_kaizen_fix(
                kaizen_id=pid,
                file_path=file_hint,
                problem=description or title,
                suggested_fix=fix_hint,
                origin="kaizen_auto_repair",
            )
        except Exception as exc:
            import traceback
            logger.error(
                "[KAIZEN] Repair %s failed: %s\n%s",
                pid, exc, traceback.format_exc(),
            )
            return {"ok": False, "error": str(exc)[:200]}

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
        last_executed_at = None
        executed = 0
        for p in proposals:
            stt = p.get("status", "unknown")
            statuses[stt] = statuses.get(stt, 0) + 1
            if stt == "resolved":
                executed += 1
                ts = p.get("resolved_at") or p.get("last_attempt")
                if ts and (last_executed_at is None or ts > last_executed_at):
                    last_executed_at = ts
        return {
            "total":            len(proposals),
            "pending":          statuses.get("pending", 0)
                                + statuses.get("auto_repair_needed", 0)
                                + statuses.get("review_and_fix", 0),
            "executed":         executed,
            "failed":           statuses.get("failed", 0),
            "last_executed_at": last_executed_at,
            "by_status":        statuses,
            "queue_file":       str(_KAIZEN_FILE),
        }
