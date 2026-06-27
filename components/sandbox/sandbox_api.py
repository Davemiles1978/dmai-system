"""
sandbox_api.py — Flask REST API that runs untrusted code inside the isolated
``dmai-sandbox`` container.

Hard guarantees:
  * Every code path returns a structured JSON response — an unhandled exception
    must NEVER surface as an HTTP 500 to the caller.
  * Code length is capped at 32KB, timeout clamped to 1-30s.
  * stdout/stderr are truncated to 64KB and stripped of ANSI / non-printable
    characters before being returned.

Entrypoint (see Dockerfile.sandbox)::

    gunicorn sandbox_api:app --bind 0.0.0.0:8765 --workers 1 --timeout 30
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request

try:  # package-relative import (running inside the package)
    from .anomaly_detector import AnomalyDetector, highest_severity
    from .sandbox_logger import SandboxLogger
except ImportError:  # flat import (gunicorn sandbox_api:app with cwd in package)
    from anomaly_detector import AnomalyDetector, highest_severity  # type: ignore
    from sandbox_logger import SandboxLogger  # type: ignore

SANDBOX_VERSION = "1.0.0"
MAX_CODE_BYTES = 32768
MAX_OUTPUT_CHARS = 65536
MIN_TIMEOUT = 1
MAX_TIMEOUT = 30
ALLOWED_LANGUAGES = ("python", "javascript", "bash")

_LANG_EXT = {"python": "code.py", "javascript": "code.js", "bash": "code.sh"}
_ANSI_RE = re.compile(r"\x1B\[[0-9;]*[mK]")

app = Flask(__name__)
_detector = AnomalyDetector()
_logger = SandboxLogger()
_START_TIME = time.time()


# ── helpers ────────────────────────────────────────────────────────────────────
def _sanitise(text: str) -> str:
    """Strip ANSI escapes and non-printable chars (keep \\n and \\t)."""
    if not text:
        return ""
    text = _ANSI_RE.sub("", text)
    return "".join(c for c in text if c in ("\n", "\t") or (32 <= ord(c) < 127) or ord(c) >= 160)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if text and len(text) > limit:
        return text[:limit]
    return text or ""


def _build_command(language: str, file_path: str) -> List[str]:
    if language == "python":
        return ["python3", "-I", "-S", file_path]
    if language == "javascript":
        return ["node", "--no-experimental-fetch",
                "--disallow-code-generation-from-strings", file_path]
    # bash
    return ["bash", "--restricted", file_path]


def _validate(payload: Dict) -> Tuple[bool, str, str, str, int]:
    """Return (ok, error_msg, code, language, timeout)."""
    code = payload.get("code")
    language = payload.get("language", "python")
    timeout = payload.get("timeout", 10)

    if not isinstance(code, str) or not code:
        return False, "Missing or invalid 'code'", "", language, 10
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return False, f"Code exceeds {MAX_CODE_BYTES} byte limit", code, language, 10
    if language not in ALLOWED_LANGUAGES:
        return False, f"Unsupported language '{language}'", code, language, 10
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        return False, "Invalid 'timeout'", code, language, 10
    timeout = max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))
    return True, "", code, language, timeout


def _error_response(request_id: str, message: str, anomalies: List[Dict] | None = None):
    return jsonify({
        "request_id": request_id,
        "status": "error",
        "stdout": "",
        "stderr": message,
        "exit_code": -1,
        "execution_time_ms": 0,
        "anomalies": anomalies or [],
        "sanitised": True,
    })


# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 3),
        "sandbox_version": SANDBOX_VERSION,
    })


@app.route("/execute", methods=["POST"])
def execute():
    request_id = ""
    work_dir = ""
    try:
        payload = request.get_json(silent=True) or {}
        request_id = payload.get("request_id") or str(uuid.uuid4())

        ok, err, code, language, timeout = _validate(payload)
        if not ok:
            return _error_response(request_id, err)

        work_dir = f"/tmp/sandbox_run_{request_id}"
        try:
            os.makedirs(work_dir, exist_ok=True)
            file_path = os.path.join(work_dir, _LANG_EXT[language])
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(code)
        except OSError as exc:
            return _error_response(request_id, f"Failed to stage code: {exc}")

        cmd = _build_command(language, file_path)

        status = "ok"
        stdout = ""
        stderr = ""
        exit_code = -1
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                cwd=work_dir,
                text=True,
                check=False,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
            if exit_code != 0:
                status = "error"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")) \
                or f"Execution exceeded {timeout}s timeout"
            exit_code = -1
        except FileNotFoundError as exc:
            return _error_response(request_id, f"Interpreter not available: {exc}")
        except Exception as exc:  # noqa: BLE001 — must never propagate
            return _error_response(request_id, f"Execution failure: {exc}")
        finally:
            execution_time_ms = int((time.time() - start) * 1000)

        stdout = _truncate(stdout)
        stderr = _truncate(stderr)

        anomalies = _detector.detect(stdout, stderr, code)
        if any(a.get("severity") == "CRITICAL" for a in anomalies):
            status = "blocked"

        stdout = _sanitise(stdout)
        stderr = _sanitise(stderr)

        try:
            entry = _logger.build_entry(
                request_id=request_id,
                language=language,
                code=code,
                execution_time_ms=execution_time_ms,
                exit_code=exit_code,
                status=status,
                stdout=stdout,
                stderr=stderr,
                anomalies=anomalies,
            )
            _logger.log(entry)
        except Exception:  # noqa: BLE001 — logging must not break the response
            pass

        return jsonify({
            "request_id": request_id,
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "execution_time_ms": execution_time_ms,
            "anomalies": anomalies,
            "sanitised": True,
        })
    except Exception as exc:  # noqa: BLE001 — absolute backstop, never return 500
        return _error_response(request_id or str(uuid.uuid4()),
                               f"Internal sandbox error: {exc}")
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765)
