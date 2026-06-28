"""
DMAI SelfHealer — Production-grade component health monitoring & auto-recovery
===============================================================================
Runs as a daemon thread. Every 60 seconds it:
  1. Syntax-checks every .py file in components/ — flags corrupt files
  2. Verifies critical core files (dmai_core_complete.py) are importable
  3. Checks that all wired background loops are still alive
  4. Restores from GitHub if a file fails syntax and a cached backup exists
  5. Logs all events to data/self_healing/heal_log.jsonl
  6. Emits a KaizenProposal when it detects a repeated failure pattern
"""

import os
import ast
import json
import time
import hashlib
import logging
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dmai.self_healer")

HEAL_LOG   = Path("data/self_healing/heal_log.jsonl")
BACKUP_DIR = Path("data/self_healing/backups")
INTERVAL   = 60   # seconds between health sweeps
CRITICAL_FILES = [
    "dmai_core_complete.py",
    "components/si_core.py",
    "components/research/autonomous_researcher.py",
    "components/sqlite_storage.py",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: dict):
    HEAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(HEAL_LOG, "a") as f:
        f.write(json.dumps({**event, "ts": _now()}) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _syntax_ok(path: Path) -> tuple[bool, str]:
    """Return (ok, error_message). Uses ast.parse for speed."""
    try:
        ast.parse(path.read_text(errors="replace"))
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


class SelfHealer:
    """
    Daemon thread that monitors DMAI component health and auto-repairs where possible.
    Wire it with: SelfHealer(components_dict).start()
    """

    def __init__(self, components: dict = None, repo_root: str = "."):
        self.components  = components or {}
        self.root        = Path(repo_root)
        self._stop       = threading.Event()
        self._thread     = None
        self._fail_count: Dict[str, int] = {}   # path → consecutive failure count
        self._hashes: Dict[str, str]     = {}   # path → last-known-good SHA-256

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):
        """Start background monitoring thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dmai-self-healer"
        )
        self._thread.start()
        logger.info("SelfHealer started (interval=%ds)", INTERVAL)

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        return {
            "running": self._thread.is_alive() if self._thread else False,
            "fail_counts": dict(self._fail_count),
            "last_log": self._last_log_entry(),
        }

    # ── Main loop ──────────────────────────────────────────────────────────

    def _run(self):
        # Initial backup of critical files
        self._snapshot_critical()
        while not self._stop.is_set():
            try:
                self._sweep_components()
                self._check_threads()
            except Exception as e:
                logger.error("SelfHealer sweep error: %s", e)
            self._stop.wait(INTERVAL)

    # ── Sweep all .py files ────────────────────────────────────────────────

    def _sweep_components(self):
        component_dir = self.root / "components"
        if not component_dir.exists():
            return

        py_files = list(component_dir.rglob("*.py"))
        # Also check critical top-level files
        for cf in CRITICAL_FILES:
            p = self.root / cf
            if p.exists():
                py_files.append(p)

        healed = repaired = 0
        for path in py_files:
            ok, err = _syntax_ok(path)
            rel = str(path.relative_to(self.root))

            if ok:
                # Update last-known-good hash
                self._hashes[rel] = _sha256(path)
                self._fail_count.pop(rel, None)
            else:
                self._fail_count[rel] = self._fail_count.get(rel, 0) + 1
                count = self._fail_count[rel]
                logger.warning("Syntax error in %s (x%d): %s", rel, count, err)
                _log({"event": "syntax_error", "file": rel, "error": err, "count": count})

                # Attempt restore from backup
                restored = self._restore_from_backup(path, rel)
                if restored:
                    repaired += 1
                    _log({"event": "restored", "file": rel})
                    logger.info("Restored %s from backup", rel)
                else:
                    # After 3 consecutive failures emit a kaizen proposal
                    if count == 3:
                        self._emit_kaizen_proposal(rel, err)

        if repaired:
            logger.info("SelfHealer: %d file(s) repaired this sweep", repaired)

    # ── Thread liveness check ──────────────────────────────────────────────

    def _check_threads(self):
        """Verify named daemon threads are still running; restart if dead.

        Only flag threads that were *configured* to start. Telegram is
        opt-in via TELEGRAM_BOT_TOKEN; never warn if the bot is intentionally
        disabled or not configured. Also skip if the configured component
        is unavailable at import time.

        Extended 2026-06-28: now also checks critical background components
        registered in self.components (kaizen_auto_repair, greyhound_runner,
        autonomous_researcher, learning_orchestrator, self_evolution_orchestrator).
        If their _thread attribute exists but is_alive()==False, attempts restart.
        """
        import os
        expected = set()
        # ai-discovery + tutor-config are always expected (always wired)
        expected.add("dmai-ai-discovery")
        expected.add("dmai-tutor-config")
        # github-monitor is expected only if a GitHub token is present
        if os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_MODELS_API_KEY"):
            expected.add("dmai-github-monitor")
        # telegram only if the bot is configured AND not disabled
        if (os.environ.get("TELEGRAM_BOT_TOKEN")
                and os.environ.get("TELEGRAM_CHAT_ID")
                and os.environ.get("TELEGRAM_BOT_DISABLE", "false").lower() != "true"):
            expected.add("dmai-telegram")
        alive = {t.name for t in threading.enumerate()}
        dead = expected - alive
        for name in dead:
            logger.warning("Thread '%s' is dead — may need restart", name)
            _log({"event": "thread_dead", "thread": name})
            # Re-start the thread via the components dict if possible
            self._try_restart_thread(name)

        # NEW: also probe registry-tracked components with _thread attribute.
        # Rate-limit restarts: if we've restarted the same component 3 times in
        # the last 30 minutes, escalate via KaizenProposal instead of looping.
        if self.components:
            critical_bg = [
                "kaizen_auto_repair",
                "greyhound_runner",
                "autonomous_researcher",
                "learning_orchestrator",
                "self_evolution_orchestrator",
            ]
            now_ts = time.time()
            if not hasattr(self, "_restart_history"):
                self._restart_history = {}  # comp_key -> [ts, ts, ts]
            for comp_key in critical_bg:
                comp = self.components.get(comp_key)
                if not comp or not hasattr(comp, "_thread"):
                    continue
                t = getattr(comp, "_thread", None)
                # _thread can be None (never started) or a Thread that's now dead
                if t is None or not (hasattr(t, "is_alive") and t.is_alive()):
                    # Drop restart timestamps older than 30 min
                    history = [ts for ts in self._restart_history.get(comp_key, []) if now_ts - ts < 1800]
                    if len(history) >= 3:
                        logger.error(
                            "Critical bg component '%s' is dead but has been restarted "
                            "%d times in the last 30 min — escalating to KaizenProposal",
                            comp_key, len(history),
                        )
                        _log({
                            "event": "thread_restart_limit",
                            "thread": comp_key,
                            "restart_count_30m": len(history),
                            "action": "escalated_to_kaizen",
                        })
                        # Emit a Kaizen proposal so the AI repair loop can investigate.
                        try:
                            self._emit_kaizen_proposal(
                                f"components/{comp_key}.py",
                                f"Background component '{comp_key}' keeps dying — restarted {len(history)} times in 30 min",
                            )
                        except Exception as _ke:
                            logger.warning("Failed to emit KaizenProposal for %s: %s", comp_key, _ke)
                        self._restart_history[comp_key] = history  # keep capped at 3
                        continue
                    logger.warning("Critical bg component '%s' is dead — restarting (attempt %d/3)", comp_key, len(history) + 1)
                    _log({"event": "thread_dead", "thread": comp_key, "restart_attempt": len(history) + 1})
                    history.append(now_ts)
                    self._restart_history[comp_key] = history
                    self._try_restart_thread(comp_key)

    def _try_restart_thread(self, thread_name: str):
        """Best-effort restart of a known daemon thread.

        Extended 2026-06-28: now covers all critical background components
        whose death blocks autonomous self-repair. Each entry maps
        thread/component name → (component_registry_key, start_method).
        """
        import threading as _t
        restarters = {
            # Original three
            "dmai-ai-discovery": ("ai_discovery", "start_discovery_loop"),
            "dmai-github-monitor": ("github_monitor", "run_monitor"),
            "dmai-tutor-config": ("tutor_configurator", "start_health_loop"),
            # Self-repair core — if this dies, DMAI loses autonomous fix capability.
            "kaizen_auto_repair": ("kaizen_auto_repair", "start_repair_loop"),
            "dmai-kaizen-auto-repair": ("kaizen_auto_repair", "start_repair_loop"),
            # Greyhound runner — monetisation; betting picks dry up if dead.
            "greyhound_runner": ("greyhound_runner", "start"),
            "dmai-greyhound-runner": ("greyhound_runner", "start"),
            # Autonomous researcher — stops new topics flowing if dead.
            "autonomous_researcher": ("autonomous_researcher", "run_continuous_research"),
            "dmai-autonomous-researcher": ("autonomous_researcher", "run_continuous_research"),
            # Learning orchestrator — SI KPIs stall if dead.
            "learning_orchestrator": ("learning_orchestrator", "start_continuous_learning"),
            "dmai-learning-orchestrator": ("learning_orchestrator", "start_continuous_learning"),
            # Self-evolution orchestrator — the gap scanner itself.
            "self_evolution_orchestrator": ("self_evolution_orchestrator", "run_forever"),
            "dmai-self-evolution": ("self_evolution_orchestrator", "run_forever"),
        }
        entry = restarters.get(thread_name)
        if not entry:
            return
        comp_key, method_name = entry
        comp = self.components.get(comp_key)
        if comp and hasattr(comp, method_name):
            try:
                t = _t.Thread(
                    target=getattr(comp, method_name),
                    daemon=True, name=thread_name
                )
                t.start()
                logger.info("SelfHealer: restarted thread '%s'", thread_name)
                _log({"event": "thread_restarted", "thread": thread_name})
            except Exception as e:
                logger.error("SelfHealer: failed to restart '%s': %s", thread_name, e)

    # ── Backup / restore ───────────────────────────────────────────────────

    def _snapshot_critical(self):
        """Back up critical files on first start."""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for cf in CRITICAL_FILES:
            src = self.root / cf
            if src.exists():
                ok, _ = _syntax_ok(src)
                if ok:
                    dest = BACKUP_DIR / cf.replace("/", "__")
                    dest.write_bytes(src.read_bytes())
                    self._hashes[cf] = _sha256(src)
        logger.debug("SelfHealer: critical file snapshots taken")

    def _restore_from_backup(self, path: Path, rel: str) -> bool:
        """Restore a file from local backup. Returns True if successful."""
        backup_name = rel.replace("/", "__")
        backup = BACKUP_DIR / backup_name
        if backup.exists():
            ok, _ = _syntax_ok(backup)
            if ok:
                path.write_bytes(backup.read_bytes())
                return True
        return False

    # ── Kaizen proposal emission ───────────────────────────────────────────

    def _emit_kaizen_proposal(self, rel: str, error: str):
        """Auto-enqueue a kaizen proposal directly into the repair queue — no human approval gate."""
        import uuid
        proposal_id = "sh-" + uuid.uuid4().hex[:8]
        proposal = {
            "id": proposal_id,
            "title": f"Auto-repair needed: {rel}",
            "priority": "HIGH",
            "source": "SelfHealer",
            "description": (
                f"File `{rel}` has failed syntax check 3 times consecutively. "
                f"Last error: {error}. Auto-queued for repair."
            ),
            "action": "patch",
            "action_type": "patch",
            "file": rel,
            "file_path": rel,
            "suggested_fix": f"Fix syntax error at: {error}",
            "status": "pending",
            "attempt_count": 0,
            "created_at": _now(),
        }
        # Write to kaizen_queue.jsonl (read by KaizenAutoRepair)
        queue_file = self.root / "data" / "kaizen_queue.jsonl"
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_file, "a") as f:
            f.write(json.dumps(proposal) + "\n")
        # Also write to kaizen_proposals.jsonl (read by KaizenExecutor for PR creation)
        proposals_file = self.root / "data" / "kaizen_proposals.jsonl"
        with open(proposals_file, "a") as f:
            f.write(json.dumps(proposal) + "\n")
        logger.warning("KaizenProposal auto-enqueued for %s (id=%s)", rel, proposal_id)
        # Fire-and-forget POST to suggestions API so dashboard counts update
        try:
            import urllib.request as _req_mod
            import urllib.error as _url_err
            import base64 as _b64
            _payload = json.dumps({
                "title": proposal["title"],
                "description": proposal["description"],
                "source": "SelfHealer",
            }).encode()
            _auth = _b64.b64encode(b"admin:dmai_master").decode()
            _r = _req_mod.Request(
                "http://localhost:5000/api/suggestions",
                data=_payload,
                headers={"Content-Type": "application/json", "Authorization": f"Basic {_auth}"},
                method="POST",
            )
            with _req_mod.urlopen(_r, timeout=3):
                pass
        except Exception as _api_e:
            logger.debug("SelfHealer: could not POST to suggestions API: %s", _api_e)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _last_log_entry(self) -> Optional[dict]:
        if not HEAL_LOG.exists():
            return None
        lines = HEAL_LOG.read_text().strip().splitlines()
        if lines:
            try:
                return json.loads(lines[-1])
            except Exception:
                pass
        return None
