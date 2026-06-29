"""
SelfEditQueue \u2014 holds proposed self-edits that exceed the SelfCommitter's
5KB safety ceiling, instead of silently refusing them.

Flow:
  1. SelfCommitter detects a large-file edit and instead of dropping it,
     calls SelfEditQueue.enqueue(...). The diff is stored in SQLite +
     written to data/pending_self_edits/<id>.diff
  2. SlackNotifier (if configured) sends an approval link.
  3. Operator approves via POST /api/self-evolution/approve/<id>
     -> queue applies the diff, AST-validates, commits to git.
  4. Or rejects via POST /api/self-evolution/reject/<id>.

Tables: se_edits
"""
from __future__ import annotations
import ast
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SelfEditQueue:
    def __init__(self, data_path: str | Path = "data", notifier=None):
        self.data_path = str(data_path).rstrip("/")
        self.db_path = os.path.join(self.data_path, "dmai_knowledge.db")
        self.diff_dir = Path(self.data_path) / "pending_self_edits"
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier
        self._lock = threading.RLock()
        self._ensure_tables()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL;")
        return c

    def _conn_safe(self) -> sqlite3.Connection:
        """Return a connection with schema guaranteed.

        Survives external DB rebuilds (watchdog quarantines + recreates the
        file). CREATE TABLE IF NOT EXISTS is idempotent and cheap; calling it
        before each operation is safer than relying on a one-shot __init__.
        """
        c = self._conn()
        try:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS se_edits (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    target_file TEXT NOT NULL,
                    bytes_proposed INTEGER NOT NULL,
                    bytes_existing INTEGER NOT NULL,
                    rationale TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decided_ts TEXT,
                    decided_by TEXT,
                    commit_sha TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_se_edits_status ON se_edits(status, ts);
                """
            )
        except Exception as e:
            logger.warning("SelfEditQueue: lazy ensure-tables failed: %s", e)
        return c

    def _ensure_tables(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS se_edits (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    target_file TEXT NOT NULL,
                    bytes_proposed INTEGER NOT NULL,
                    bytes_existing INTEGER NOT NULL,
                    rationale TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decided_ts TEXT,
                    decided_by TEXT,
                    commit_sha TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_se_edits_status ON se_edits(status, ts);
            """)

    # ── enqueue ────────────────────────────────────────────────────────
    def enqueue(
        self,
        capability: str,
        target_file: str,
        code: str,
        rationale: str = "",
    ) -> str:
        existing = b""
        if os.path.exists(target_file):
            with open(target_file, "rb") as f:
                existing = f.read()
        ts = datetime.now(timezone.utc).isoformat()
        digest = hashlib.sha256(
            (target_file + ts + str(len(code))).encode()
        ).hexdigest()[:12]
        edit_id = f"se_{digest}"
        # Persist proposed file content
        proposed_path = self.diff_dir / f"{edit_id}.py"
        proposed_path.write_text(code)
        meta_path = self.diff_dir / f"{edit_id}.json"
        meta_path.write_text(json.dumps({
            "id": edit_id,
            "ts": ts,
            "capability": capability,
            "target_file": target_file,
            "bytes_proposed": len(code),
            "bytes_existing": len(existing),
            "rationale": rationale,
        }, indent=2))
        with self._lock, self._conn_safe() as conn:
            conn.execute(
                "INSERT INTO se_edits (id, ts, capability, target_file, "
                "bytes_proposed, bytes_existing, rationale, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                (edit_id, ts, capability, target_file, len(code),
                 len(existing), rationale),
            )
        logger.info(
            f"SelfEditQueue: enqueued {edit_id} for {target_file} "
            f"({len(code)} bytes; existing {len(existing)})"
        )
        # Notify if available
        if self.notifier is not None:
            try:
                self.notifier.alert(
                    category="tier",
                    title="Self-edit awaiting approval",
                    text=(f"DMAI proposes editing {target_file} "
                          f"({len(existing)} -> {len(code)} bytes).\n"
                          f"Capability: {capability}\n"
                          f"Review: /api/self-evolution/pending"),
                )
            except Exception as e:
                logger.warning(f"SelfEditQueue: notify failed: {e}")
        return edit_id

    # ── list/read ──────────────────────────────────────────────────────
    def pending(self) -> List[Dict[str, Any]]:
        with self._lock, self._conn_safe() as conn:
            rows = conn.execute(
                "SELECT id, ts, capability, target_file, bytes_proposed, "
                "bytes_existing, rationale FROM se_edits "
                "WHERE status='pending' ORDER BY ts DESC"
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "capability": r[2], "target_file": r[3],
             "bytes_proposed": r[4], "bytes_existing": r[5],
             "rationale": r[6]}
            for r in rows
        ]

    def get_proposed_code(self, edit_id: str) -> Optional[str]:
        p = self.diff_dir / f"{edit_id}.py"
        if p.exists():
            return p.read_text()
        return None

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._conn_safe() as conn:
            rows = conn.execute(
                "SELECT id, ts, capability, target_file, status, "
                "decided_ts, decided_by, commit_sha FROM se_edits "
                "ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "capability": r[2], "target_file": r[3],
             "status": r[4], "decided_ts": r[5], "decided_by": r[6],
             "commit_sha": r[7]}
            for r in rows
        ]

    # ── approve / reject ───────────────────────────────────────────────
    def approve(self, edit_id: str, decided_by: str = "operator") -> Dict[str, Any]:
        with self._lock, self._conn_safe() as conn:
            row = conn.execute(
                "SELECT target_file, status FROM se_edits WHERE id=?",
                (edit_id,),
            ).fetchone()
            if not row:
                return {"ok": False, "error": "not found"}
            if row[1] != "pending":
                return {"ok": False, "error": f"status is {row[1]}"}
            target_file = row[0]
        code = self.get_proposed_code(edit_id)
        if not code:
            return {"ok": False, "error": "proposed code missing"}
        # AST validate
        try:
            ast.parse(code)
        except SyntaxError as e:
            self._mark(edit_id, "rejected_syntax", decided_by, None)
            return {"ok": False, "error": f"syntax: {e}"}
        # Write file
        os.makedirs(os.path.dirname(target_file) or ".", exist_ok=True)
        backup = None
        if os.path.exists(target_file):
            backup = target_file + ".bak"
            with open(target_file, "rb") as f:
                open(backup, "wb").write(f.read())
        with open(target_file, "w") as f:
            f.write(code)
        # Smoke test main file
        try:
            r = subprocess.run(
                ["python3", "-c",
                 "import ast; ast.parse(open('dmai_core_complete.py').read()); print('OK')"],
                capture_output=True, text=True, timeout=15,
            )
            if "OK" not in r.stdout:
                if backup and os.path.exists(backup):
                    os.rename(backup, target_file)
                self._mark(edit_id, "rejected_smoke", decided_by, None)
                return {"ok": False, "error": "smoke test failed"}
        finally:
            if backup and os.path.exists(backup):
                try:
                    os.remove(backup)
                except Exception:
                    pass
        # Git commit
        commit_sha = self._git_commit(target_file, edit_id)
        self._mark(edit_id, "approved", decided_by, commit_sha)
        return {"ok": True, "commit_sha": commit_sha}

    def reject(self, edit_id: str, decided_by: str = "operator") -> Dict[str, Any]:
        self._mark(edit_id, "rejected", decided_by, None)
        return {"ok": True}

    def _mark(self, edit_id: str, status: str, decided_by: str,
              commit_sha: Optional[str]) -> None:
        with self._lock, self._conn_safe() as conn:
            conn.execute(
                "UPDATE se_edits SET status=?, decided_ts=?, decided_by=?, "
                "commit_sha=? WHERE id=?",
                (status, datetime.now(timezone.utc).isoformat(),
                 decided_by, commit_sha, edit_id),
            )

    def _git_commit(self, target_file: str, edit_id: str) -> Optional[str]:
        try:
            subprocess.run(["git", "add", target_file],
                           check=True, timeout=10)
            msg = f"self-edit: approved {edit_id} -> {target_file}"
            subprocess.run(["git", "commit", "-m", msg],
                           check=True, timeout=15)
            r = subprocess.run(["git", "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=5)
            sha = r.stdout.strip()[:12]
            # Push is best-effort; relies on credentials in env
            try:
                subprocess.run(["git", "push"], check=True, timeout=30)
            except Exception as e:
                logger.warning(f"SelfEditQueue: push failed (commit kept locally): {e}")
            return sha
        except Exception as e:
            logger.error(f"SelfEditQueue: git commit failed: {e}")
            return None
