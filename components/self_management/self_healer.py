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

# Rate-limited loops do not keep a thread continuously busy — they sleep for
# long stretches between fires (greyhound fires daily, the trader ticks every
# ~2h) and signal liveness by touching ``data/<component>_heartbeat.txt`` or a
# ``data/<component>_last_*.txt`` marker instead. A fresh heartbeat/marker means
# the component is healthy and MUST NOT be restarted, even if its thread object
# is momentarily absent or sleeping. Map component -> max age (seconds) a
# heartbeat/marker may reach before we treat the component as genuinely dead.
DEFAULT_HEARTBEAT_MAX_AGE = 15 * 60            # 15 min
HEARTBEAT_MAX_AGE = {
    "greyhound_runner":      24 * 3600 + 3600,  # fires daily at 08:00 UK (PR #162)
    "autonomous_trader":     2 * 3600 + 600,    # ticks ~2h, heartbeat every ~5m (PR #163a)
    "autonomous_researcher": 60 * 60,
    "learning_orchestrator": 30 * 60,
}

# Components guarded by an env flag. When the flag is off we skip liveness
# checks and restarts entirely — a stopped component is the intended state, not
# a fault. kaizen_auto_repair is OFF unless KAIZEN_AUTO_REPAIR_ENABLED == "1".
ENV_GATED_COMPONENTS = {
    "kaizen_auto_repair": "KAIZEN_AUTO_REPAIR_ENABLED",
}

CRITICAL_FILES = [
    "dmai_core_complete.py",
    "components/si_core.py",
    "components/research/autonomous_researcher.py",
    "components/sqlite_storage.py",
]

# Directories under components/ that MUST NOT be syntax-swept.
# These are backup snapshots and generated artifacts — they intentionally
# contain stale/broken code and treating them as live source drives the
# Kaizen queue full of "Auto-repair needed" proposals nothing can fix.
EXCLUDED_DIR_PREFIXES = (
    "backup_final_",
    "backup_before_",
)
EXCLUDED_DIR_NAMES = frozenset({
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
})


def _is_excluded_path(path: Path, root: Path) -> bool:
    """True if ``path`` is inside a backup / cache / vendored dir the healer
    must skip. Match is done against every path segment under ``root``.
    """
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    for part in rel_parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        for prefix in EXCLUDED_DIR_PREFIXES:
            if part.startswith(prefix):
                return True
        # Any *.dist-info or *.egg-info directory
        if part.endswith(".dist-info") or part.endswith(".egg-info"):
            return True
    return False


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
        # Grace period — do not restart background components for the first
        # 5 minutes of SelfHealer's life. Most components start their _thread
        # asynchronously after boot, so restarting them mid-init would race
        # the boot sequence.
        self._started_ts = time.time()
        self._grace_seconds = 300

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
        # One-shot: retire stale Kaizen proposals whose target file lives in
        # an excluded dir (backups, __pycache__). Prior versions of this
        # module scanned backup snapshots and filed thousands of
        # "Auto-repair needed:" proposals that nothing could act on.
        try:
            self._retire_excluded_kaizen_entries()
        except Exception as e:
            logger.warning("SelfHealer: kaizen retirement failed: %s", e)
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

        # Skip backup snapshots, __pycache__, and vendored dirs — those
        # contain intentional/legacy syntax errors and treating them as
        # live source spams the Kaizen queue with un-actionable proposals.
        py_files = [
            p for p in component_dir.rglob("*.py")
            if not _is_excluded_path(p, self.root)
        ]
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
        # Grace period: skip during the first 5 minutes after SelfHealer starts
        # so the normal boot sequence has time to start bg threads itself.
        if self.components and (time.time() - self._started_ts) >= self._grace_seconds:
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
                # Env-gated components (e.g. kaizen_auto_repair) are skipped
                # entirely when their flag is off — a stopped component is the
                # intended state, so probing/restarting it just flaps.
                gate = ENV_GATED_COMPONENTS.get(comp_key)
                if gate and os.getenv(gate, "0") != "1":
                    continue
                comp = self.components.get(comp_key)
                if not comp or not hasattr(comp, "_thread"):
                    continue
                t = getattr(comp, "_thread", None)
                # A fresh heartbeat/marker file means a rate-limited loop is
                # healthy but sleeping — never restart it in that case.
                if self._alive_by_heartbeat(comp_key):
                    continue
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

    def _alive_by_heartbeat(self, comp_key: str) -> bool:
        """True if a fresh heartbeat file or last-fire marker proves liveness.

        Rate-limited loops (daily greyhound, hourly trader) sleep for long
        stretches, so thread activity alone is a false "dead" signal. They touch
        ``data/<component>_heartbeat.txt`` while running and/or drop a
        ``data/<component>_last_*.txt`` marker each time they fire. If either is
        newer than the component's allowed max age, the component is healthy and
        must not be restarted.
        """
        max_age = HEARTBEAT_MAX_AGE.get(comp_key, DEFAULT_HEARTBEAT_MAX_AGE)
        now_ts = time.time()
        data_dir = self.root / "data"

        candidates = [data_dir / f"{comp_key}_heartbeat.txt"]
        try:
            candidates.extend(data_dir.glob(f"{comp_key}_last_*.txt"))
        except OSError:
            pass

        for path in candidates:
            try:
                if path.exists() and (now_ts - path.stat().st_mtime) <= max_age:
                    return True
            except OSError:
                continue
        return False

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

    def _retire_excluded_kaizen_entries(self) -> None:
        """Remove any Kaizen entries in both jsonl stores whose ``file`` or
        ``file_path`` points inside an excluded directory. Idempotent.

        Runs once per SelfHealer.start(), which is once per app boot.
        """
        data_dir = self.root / "data"
        removed_total = 0
        for name in ("kaizen_proposals.jsonl", "kaizen_queue.jsonl"):
            f = data_dir / name
            if not f.exists():
                continue
            try:
                kept: list[str] = []
                removed_here = 0
                for line in f.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # Preserve un-parseable lines verbatim.
                        kept.append(line)
                        continue
                    target = obj.get("file") or obj.get("file_path") or ""
                    # Cheap prefix check — target is a repo-relative path str.
                    if isinstance(target, str) and self._path_matches_excluded(target):
                        removed_here += 1
                        continue
                    kept.append(line)
                if removed_here:
                    # Atomic rewrite via temp + rename.
                    tmp = f.with_suffix(f.suffix + ".tmp")
                    tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
                    os.replace(tmp, f)
                    logger.info(
                        "SelfHealer: retired %d stale kaizen entries from %s",
                        removed_here, name,
                    )
                    _log({
                        "event": "kaizen_retired", "file": name, "count": removed_here,
                    })
                removed_total += removed_here
            except Exception as e:
                logger.warning("kaizen retirement pass on %s failed: %s", name, e)
        if removed_total:
            logger.info(
                "SelfHealer: retired %d stale kaizen entries total", removed_total
            )

    @staticmethod
    def _path_matches_excluded(rel: str) -> bool:
        """True if ``rel`` (a repo-relative posix path) sits inside an
        excluded directory. Mirrors _is_excluded_path but works on strings
        so we don't need a real filesystem entry.
        """
        parts = [p for p in rel.replace("\\", "/").split("/") if p]
        for part in parts:
            if part in EXCLUDED_DIR_NAMES:
                return True
            for prefix in EXCLUDED_DIR_PREFIXES:
                if part.startswith(prefix):
                    return True
            if part.endswith(".dist-info") or part.endswith(".egg-info"):
                return True
        return False

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
