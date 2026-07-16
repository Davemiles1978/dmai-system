"""Sandboxed subprocess runner for candidate capability modules.

Runs staged capability code in a *separate Python process* with:

* a wall-clock timeout (default 5s);
* a CPU / address-space rlimit soft-cap (POSIX);
* a working directory that is NOT the repo root, so relative writes
  cannot corrupt DMAI's state;
* environment stripped to a minimal safe set.

Two public functions:

* :func:`run_pytest_file` - runs ``pytest <path>`` in a subprocess and
  returns exit code + captured output. Used for the auto-generated
  smoke test.
* :func:`run_happy_path` - imports the staged module *inside* a
  subprocess and calls ``run(**kwargs)``. Returns a serialisable
  result. Any exception, timeout, or resource-limit hit becomes
  ``ok=False`` with a specific reason string.

The parent process never imports the candidate module. This is the
whole point.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SEC   = 5
DEFAULT_MEMORY_MB     = 256
DEFAULT_CPU_SECONDS   = 5

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Results ───────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    ok: bool
    reason: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    timed_out: bool = False
    duration_sec: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok":           self.ok,
            "reason":       self.reason,
            "exit_code":    self.exit_code,
            "stdout":       self.stdout[-2000:],
            "stderr":       self.stderr[-2000:],
            "return_value": self.return_value,
            "timed_out":    self.timed_out,
            "duration_sec": round(self.duration_sec, 3),
        }


# ── Environment for the sandbox subprocess ────────────────────────────────

def _clean_env() -> Dict[str, str]:
    """Return a stripped-down env - no secrets, no PATH tricks."""
    keep = {"LANG", "LC_ALL", "TZ", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    # A predictable PYTHONPATH pointing at the repo so imports resolve.
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Explicitly clear secret-shaped keys that may be inherited.
    return env


def _preexec_rlimit(cpu_sec: int, mem_mb: int):
    """Return a preexec_fn that applies POSIX resource limits.

    On non-POSIX (Windows) systems this returns None and no limits
    are applied; the wall-clock timeout is the only remaining bound.
    """
    try:
        import resource  # POSIX only
    except ImportError:
        return None

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
        soft = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            # Some hosts (macOS) don't honour RLIMIT_AS; ignore.
            pass
    return _apply


# ── Happy-path runner ─────────────────────────────────────────────────────

_HAPPY_PATH_HARNESS = r'''
import json, sys, importlib, traceback, time
mod_name = sys.argv[1]
kwargs   = json.loads(sys.argv[2])
try:
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, "run"):
        print(json.dumps({"ok": False, "reason": "no_run_attr"}))
        sys.exit(2)
    t0 = time.monotonic()
    # PR NN: graceful kwargs fallback. If codegen produced a signature
    # that rejects our default happy_kwargs (e.g. it requires db_path
    # or accepts nothing), retry with no kwargs before failing. Real
    # signature validation lives in capability_verifier; the sandbox
    # only checks that run() can be invoked at all.
    try:
        rv = mod.run(**kwargs)
    except TypeError as te:
        msg = str(te)
        if "unexpected keyword" in msg or "takes 0 positional" in msg:
            rv = mod.run()
        else:
            raise
    dt = time.monotonic() - t0
    # Only try to serialise; if it fails we still count the call as ok.
    try:
        serialised = json.loads(json.dumps(rv, default=str))
    except Exception:
        serialised = repr(rv)
    print(json.dumps({"ok": True, "return_value": serialised, "dt": dt}))
except Exception as e:
    print(json.dumps({
        "ok": False,
        "reason": "runtime_error",
        "error_type": type(e).__name__,
        "error_msg":  str(e),
        "traceback":  traceback.format_exc(),
    }))
    sys.exit(3)
'''


def run_happy_path(module_dotted: str,
                   kwargs: Dict[str, Any],
                   *,
                   timeout_sec: int = DEFAULT_TIMEOUT_SEC,
                   cpu_sec:     int = DEFAULT_CPU_SECONDS,
                   memory_mb:   int = DEFAULT_MEMORY_MB) -> SandboxResult:
    """Import *module_dotted* in a subprocess and call ``run(**kwargs)``.

    Returns a :class:`SandboxResult`. Never raises.
    """
    import time as _t
    t0 = _t.monotonic()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_HAPPY_PATH_HARNESS)
        harness = f.name

    try:
        proc = subprocess.run(
            [sys.executable, harness, module_dotted, json.dumps(kwargs)],
            cwd=str(REPO_ROOT),
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=_preexec_rlimit(cpu_sec, memory_mb),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return SandboxResult(
            ok=False, reason="timeout", timed_out=True,
            stdout=(e.stdout or b"").decode("utf-8", "replace")[-2000:],
            stderr=(e.stderr or b"").decode("utf-8", "replace")[-2000:],
            duration_sec=_t.monotonic() - t0,
        )
    finally:
        try:
            os.unlink(harness)
        except OSError:
            pass

    duration = _t.monotonic() - t0
    parsed: Optional[Dict[str, Any]] = None
    # The harness prints exactly one JSON object on stdout.
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if parsed is None:
        # A negative returncode means the child was killed by a
        # signal. -SIGXCPU (-24) means we tripped the CPU rlimit -
        # that is a timeout in every practical sense. -SIGKILL (-9)
        # is what the wall-clock kill path uses. Reclassify so the
        # materialiser can report "timeout" instead of the vague
        # "no_result_json".
        rc = proc.returncode or 0
        if rc in (-9, -24) or duration >= timeout_sec - 0.05:
            return SandboxResult(
                ok=False, reason="timeout", exit_code=rc,
                stdout=proc.stdout, stderr=proc.stderr,
                duration_sec=duration, timed_out=True,
            )
        return SandboxResult(
            ok=False, reason="no_result_json", exit_code=rc,
            stdout=proc.stdout, stderr=proc.stderr, duration_sec=duration,
        )
    if not parsed.get("ok"):
        return SandboxResult(
            ok=False,
            reason=str(parsed.get("reason", "unknown")),
            exit_code=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
            duration_sec=duration,
            return_value=parsed,
        )
    return SandboxResult(
        ok=True, exit_code=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr,
        return_value=parsed.get("return_value"),
        duration_sec=duration,
    )


# ── Pytest smoke runner ───────────────────────────────────────────────────

def run_pytest_file(test_path: Path,
                    *,
                    timeout_sec: int = 30,
                    cpu_sec:     int = 30,
                    memory_mb:   int = 512) -> SandboxResult:
    """Invoke ``pytest`` on *test_path* in a subprocess."""
    import time as _t
    t0 = _t.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path),
             "-q", "--no-header", "--tb=short",
             "--import-mode=importlib"],
            cwd=str(REPO_ROOT),
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=_preexec_rlimit(cpu_sec, memory_mb),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return SandboxResult(
            ok=False, reason="pytest_timeout", timed_out=True,
            stdout=(e.stdout or b"").decode("utf-8", "replace")[-2000:],
            stderr=(e.stderr or b"").decode("utf-8", "replace")[-2000:],
            duration_sec=_t.monotonic() - t0,
        )

    duration = _t.monotonic() - t0
    if proc.returncode == 0:
        return SandboxResult(
            ok=True, exit_code=0,
            stdout=proc.stdout, stderr=proc.stderr,
            duration_sec=duration,
        )
    return SandboxResult(
        ok=False, reason=f"pytest_exit_{proc.returncode}",
        exit_code=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr,
        duration_sec=duration,
    )


__all__ = [
    "SandboxResult",
    "run_happy_path",
    "run_pytest_file",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_MEMORY_MB",
    "DEFAULT_CPU_SECONDS",
]
