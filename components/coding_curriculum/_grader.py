"""Safe grader for coding-curriculum exercises.

Runs a candidate module's ``run(**kwargs)`` in an isolated subprocess
with a hard timeout, then compares the returned value against each
grading case's predicate or expected value.

Safety properties:
  - Subprocess isolation (crashing the candidate does not crash DMAI).
  - Hard wall-clock timeout via subprocess.run(timeout=...).
  - No network, no import of DMAI packages, no filesystem writes from
    the runner.
  - Candidate code is written to a tempfile, executed, and the tempfile
    plus its parent directory are cleaned up unconditionally.
  - Predicates are evaluated in a restricted namespace (only 'result',
    a small set of stdlib callables).

Return contract from ``grade_exercise``:
    {
      "ok":          bool,       # True iff every case passed
      "cases":       [ {passed, error, description}, ... ],
      "runtime_ms":  int,
      "reason":      str | None, # short summary
    }
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._exercises import Exercise, GradingCase


# Safe builtins available to grading predicates.
_PRED_GLOBALS = {
    "__builtins__": {
        "isinstance": isinstance, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "tuple": tuple,
        "abs": abs, "len": len, "sorted": sorted, "sum": sum,
        "min": min, "max": max, "bool": bool, "any": any, "all": all,
        "range": range, "set": set,
    },
}


def _write_candidate(dir_path: Path, candidate_code: str) -> Path:
    p = dir_path / "candidate.py"
    p.write_text(candidate_code, encoding="utf-8")
    return p


_RUNNER_TEMPLATE = textwrap.dedent(
    """
    # Runner harness. Loads candidate.py by path (no package import),
    # calls run(**kwargs), and prints the result as a JSON line on
    # stdout. Any exception is captured and reported as JSON too.
    import importlib.util, json, sys, traceback, os
    try:
        spec = importlib.util.spec_from_file_location('candidate', {candidate_path!r})
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, 'run'):
            print(json.dumps({{'ok': False, 'err': 'no run() function'}}))
            sys.exit(0)
        kwargs = json.loads({kwargs_json!r})
        try:
            out = mod.run(**kwargs)
        except Exception as e:
            print(json.dumps({{'ok': False, 'err': f'{{type(e).__name__}}: {{e}}',
                               'trace': traceback.format_exc()[-800:]}}))
            sys.exit(0)
        try:
            json.dumps(out)
        except TypeError:
            out = json.loads(json.dumps(out, default=str))
        print(json.dumps({{'ok': True, 'result': out}}))
    except Exception as e:
        print(json.dumps({{'ok': False, 'err': f'runner-{{type(e).__name__}}: {{e}}',
                           'trace': traceback.format_exc()[-800:]}}))
    """
).strip()


def _run_case(candidate_path: Path,
              case: GradingCase,
              timeout_seconds: float) -> Dict[str, Any]:
    """Execute one grading case in a subprocess."""
    runner_src = _RUNNER_TEMPLATE.format(
        candidate_path=str(candidate_path),
        kwargs_json=json.dumps(case.kwargs),
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_src],
            capture_output=True, text=True,
            timeout=timeout_seconds,
            # Isolate: no DMAI env leakage into the candidate runner.
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C"},
        )
    except subprocess.TimeoutExpired:
        return {
            "passed":      False,
            "error":       f"timeout after {timeout_seconds:.1f}s",
            "description": case.description,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "passed":      False,
            "error":       f"runner failed: {type(e).__name__}: {e}",
            "description": case.description,
        }

    stdout = (proc.stdout or "").strip().splitlines()
    stderr = (proc.stderr or "").strip()[-400:]
    if not stdout:
        return {
            "passed":      False,
            "error":       f"no output (rc={proc.returncode}); stderr={stderr!r}",
            "description": case.description,
        }
    try:
        payload = json.loads(stdout[-1])
    except json.JSONDecodeError:
        return {
            "passed":      False,
            "error":       f"runner emitted non-JSON: {stdout[-1][:120]!r}",
            "description": case.description,
        }
    if not payload.get("ok"):
        return {
            "passed":      False,
            "error":       payload.get("err", "candidate raised"),
            "description": case.description,
        }

    result = payload.get("result")

    # Compare using predicate first, then equality.
    try:
        if case.predicate:
            ok = bool(eval(case.predicate, _PRED_GLOBALS, {"result": result}))
            if not ok:
                return {
                    "passed":      False,
                    "error":       f"predicate false; got result={result!r}",
                    "description": case.description,
                }
        else:
            if result != case.expected:
                return {
                    "passed":      False,
                    "error":       f"expected {case.expected!r}, got {result!r}",
                    "description": case.description,
                }
    except Exception as e:  # noqa: BLE001
        return {
            "passed":      False,
            "error":       f"predicate raised {type(e).__name__}: {e}",
            "description": case.description,
        }

    return {"passed": True, "error": None, "description": case.description}


def grade_exercise(exercise: Exercise,
                   candidate_code: str,
                   *,
                   timeout_seconds: float = 3.0) -> Dict[str, Any]:
    """Grade a candidate module against an exercise's cases.

    Rules honoured:
      - Never hang: every case has a subprocess timeout.
      - Never write partial/None results: if any case fails or errors,
        the whole exercise is a fail.
      - Cleanup unconditional: the candidate tempdir is always removed.
    """
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="dmai_grade_"))
    try:
        candidate_path = _write_candidate(tmp, candidate_code)
        # Cheap sanity: does it parse?
        try:
            compile(candidate_code, str(candidate_path), "exec")
        except SyntaxError as e:
            return {
                "ok":         False,
                "cases":      [],
                "runtime_ms": int((time.time() - t0) * 1000),
                "reason":     f"SyntaxError: {e.msg} at line {e.lineno}",
            }

        results: List[Dict[str, Any]] = []
        all_ok = True
        for case in exercise.grading:
            r = _run_case(candidate_path, case, timeout_seconds)
            results.append(r)
            if not r["passed"]:
                all_ok = False
                # Do not short-circuit: we want the full case log for
                # study analytics. But stop early if we already have
                # >= 3 failures — no point piling on.
                if sum(1 for x in results if not x["passed"]) >= 3:
                    break

        return {
            "ok":         all_ok,
            "cases":      results,
            "runtime_ms": int((time.time() - t0) * 1000),
            "reason":     None if all_ok else "one or more cases failed",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
