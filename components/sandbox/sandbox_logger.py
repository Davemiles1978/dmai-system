"""
SandboxLogger — append-only JSONL audit trail for every sandbox execution.

Each line is one JSON object. The log self-rotates once it grows past 10MB:
the current file is renamed to ``<path>.1`` (overwriting any previous rotation)
and a fresh file is started.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

try:  # package-relative import
    from .anomaly_detector import highest_severity
except ImportError:  # flat import (gunicorn cwd inside the package)
    from anomaly_detector import highest_severity  # type: ignore

DEFAULT_LOG_PATH = os.environ.get("SANDBOX_LOG_PATH", "/tmp/sandbox_activity.jsonl")
_MAX_BYTES = 10 * 1024 * 1024  # 10MB


class SandboxLogger:
    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = log_path or DEFAULT_LOG_PATH
        self._lock = threading.Lock()

    # ── writing ───────────────────────────────────────────────────────────────
    def log(self, entry: Dict[str, Any]) -> None:
        """Append a single entry as a JSON line. Never raises to the caller."""
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            line = json.dumps({"ts": self.now(), "status": "log_serialise_error"})
        with self._lock:
            try:
                self._rotate_if_needed()
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                # Logging must never break execution.
                pass

    def build_entry(
        self,
        *,
        request_id: str,
        language: str,
        code: str,
        execution_time_ms: int,
        exit_code: int,
        status: str,
        stdout: str,
        stderr: str,
        anomalies: List[Dict],
    ) -> Dict[str, Any]:
        return {
            "ts": self.now(),
            "request_id": request_id,
            "language": language,
            "code_sha256": hashlib.sha256((code or "").encode("utf-8")).hexdigest(),
            "code_length": len(code or ""),
            "execution_time_ms": execution_time_ms,
            "exit_code": exit_code,
            "status": status,
            "stdout_length": len(stdout or ""),
            "stderr_length": len(stderr or ""),
            "anomaly_count": len(anomalies),
            "highest_severity": highest_severity(anomalies),
            "anomalies": anomalies,
        }

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > _MAX_BYTES:
                rotated = self.log_path + ".1"
                try:
                    if os.path.exists(rotated):
                        os.remove(rotated)
                except OSError:
                    pass
                os.replace(self.log_path, rotated)
        except OSError:
            pass

    # ── reading ───────────────────────────────────────────────────────────────
    def _read_lines(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entries.append(json.loads(raw))
                    except (ValueError, TypeError):
                        continue
        except OSError:
            return []
        return entries

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        entries = self._read_lines()
        if n <= 0:
            return []
        return entries[-n:]

    def get_stats(self) -> Dict[str, Any]:
        entries = self._read_lines()
        status_counts: Counter = Counter()
        severity_counts: Counter = Counter()
        language_counts: Counter = Counter()

        for e in entries:
            status_counts[e.get("status", "unknown")] += 1
            language_counts[e.get("language", "unknown")] += 1
            for a in e.get("anomalies", []) or []:
                sev = a.get("severity", "NONE")
                severity_counts[sev] += 1

        return {
            "total_runs": len(entries),
            "ok_count": status_counts.get("ok", 0),
            "error_count": status_counts.get("error", 0),
            "timeout_count": status_counts.get("timeout", 0),
            "blocked_count": status_counts.get("blocked", 0),
            "anomaly_counts_by_severity": dict(severity_counts),
            "top_languages": dict(language_counts.most_common(5)),
        }
