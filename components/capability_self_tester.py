"""Capability Self-Tester — Layer 4 chunk L4-4.

Runs the ``test_cases`` declared on a ``ModuleSpec`` against the implemented
module in an isolated subprocess. On failure, marks the corresponding
SelfEditQueue record as ``rejected_test`` and (up to MAX_RETRIES) re-queues a
new spec-gen + impl cycle with the failure diffs threaded back through the
gap entry's ``extra.failure_context``.

Adapted to the **actual** SelfEditQueue interface (``data_path=`` ctor,
``reject(edit_id, decided_by=...)``), since the spec snippet was written
against an idealised signature.
"""

from __future__ import annotations

import json
import os
import sqlite3  # noqa: F401  # used only for sqlite3.Error type ref
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    test_case: Dict[str, Any]
    passed: bool
    stdout: str
    stderr: str
    diff: str = ""


@dataclass
class TestResult:
    passed: bool
    failures: List[CaseResult]
    edit_id: str
    status: str  # "test_passed" | "rejected_test"
    retry_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Self-tester
# ---------------------------------------------------------------------------


class CapabilitySelfTester:
    """Execute ModuleSpec.test_cases against an implemented module."""

    MAX_RETRIES = 3
    SUBPROCESS_TIMEOUT_SEC = 30

    def __init__(self, repo_root: str = ".", data_path: str = "data") -> None:
        self.repo_root = repo_root
        self.data_path = data_path
        self.db_path = str(Path(data_path) / "dmai_knowledge.db")

    # ---- public API ----------------------------------------------------

    def run_tests(
        self,
        spec,  # ModuleSpec (avoid hard import to keep module light)
        edit_id: str,
        retry_count: int = 0,
    ) -> TestResult:
        """Execute every test_case for spec; return a TestResult.

        On any failure: marks SelfEditQueue record status=rejected_test, and
        (if retry_count < MAX_RETRIES) re-queues a fresh spec-gen + impl
        cycle with the failure diffs in the gap entry's extra.failure_context.

        On success: marks status=test_passed.
        """
        failures: List[CaseResult] = []
        for tc in (spec.test_cases or []):
            result = self._run_one(spec.target_path, tc)
            if not result.passed:
                failures.append(result)

        if failures:
            return self._handle_failure(spec, edit_id, failures, retry_count)
        return self._handle_success(spec, edit_id, retry_count)

    # ---- private helpers ----------------------------------------------

    def _run_one(self, target_path: str, tc: Dict[str, Any]) -> CaseResult:
        """Run one test case in a subprocess.

        Test case shape (additive — unknown keys are ignored):
            {
              "name": str,                # optional
              "call": "module.func(*args, **kwargs)",  # optional override
              "expected_output": Any,     # JSON-comparable
              "script": str,              # full override; bypass auto-build
            }
        """
        script = tc.get("script") or self._build_test_script(target_path, tc)
        try:
            proc = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=self.SUBPROCESS_TIMEOUT_SEC,
                cwd=self.repo_root,
            )
        except subprocess.TimeoutExpired as e:
            return CaseResult(
                test_case=tc, passed=False, stdout="",
                stderr=f"TIMEOUT after {self.SUBPROCESS_TIMEOUT_SEC}s: {e}",
                diff="timeout",
            )
        except Exception as e:  # noqa: BLE001
            return CaseResult(
                test_case=tc, passed=False, stdout="",
                stderr=f"subprocess exception: {e!r}", diff="exception",
            )

        passed = proc.returncode == 0
        diff_str = ""
        if not passed:
            diff_str = self._diff(tc.get("expected_output"), proc.stdout)
        return CaseResult(
            test_case=tc, passed=passed,
            stdout=proc.stdout, stderr=proc.stderr, diff=diff_str,
        )

    def _build_test_script(self, target_path: str, tc: Dict[str, Any]) -> str:
        """Build a default Python script that imports the module and runs `call`.

        If `expected_output` is provided, the script exits non-zero on mismatch.
        If `call` is omitted, the script just imports the module (smoke test).
        """
        module = target_path.replace("/", ".")
        if module.endswith(".py"):
            module = module[:-3]
        call_expr = tc.get("call")
        expected = tc.get("expected_output")
        lines = [
            "import json, sys",
            f"import {module} as _m",
        ]
        if call_expr:
            lines.append(f"_result = _m.{call_expr}")
            lines.append("print(json.dumps(_result, default=str))")
            if expected is not None:
                expected_json = json.dumps(expected, default=str)
                lines.append(f"_expected = json.loads({expected_json!r})")
                lines.append("if _result != _expected: sys.exit(1)")
        else:
            # Pure import smoke test
            lines.append("print('imported ok')")
        return "\n".join(lines)

    def _diff(self, expected: Any, actual_stdout: str) -> str:
        actual = (actual_stdout or "").strip()
        try:
            expected_str = json.dumps(expected, default=str, sort_keys=True)
        except Exception:  # noqa: BLE001
            expected_str = repr(expected)
        return f"expected={expected_str} | actual={actual[:500]}"

    def _handle_failure(
        self,
        spec,
        edit_id: str,
        failures: List[CaseResult],
        retry_count: int,
    ) -> TestResult:
        # Mark the SelfEditQueue record rejected — use direct SQL because
        # the public reject() signature is reject(id, decided_by) and we want
        # a distinct status ("rejected_test") for observability.
        self._update_edit_status(edit_id, "rejected_test", failures)

        # Re-queue (best effort) if within retry budget.
        if retry_count < self.MAX_RETRIES:
            try:
                self._requeue_with_context(spec, failures, retry_count + 1)
            except Exception as e:  # noqa: BLE001
                # Re-queue is best-effort; record but don't crash the tester.
                self._log_event(
                    "requeue_failed",
                    {"edit_id": edit_id, "error": repr(e)},
                )

        return TestResult(
            passed=False, failures=failures, edit_id=edit_id,
            status="rejected_test", retry_count=retry_count,
        )

    def _handle_success(self, spec, edit_id: str, retry_count: int) -> TestResult:
        self._update_edit_status(edit_id, "test_passed", [])
        return TestResult(
            passed=True, failures=[], edit_id=edit_id,
            status="test_passed", retry_count=retry_count,
        )

    def _update_edit_status(
        self,
        edit_id: str,
        status: str,
        failures: List[CaseResult],
    ) -> None:
        """Set se_edits.status for edit_id. Tolerates missing DB / row.

        Uses ``safe_open_kdb`` (the project-standard SQLite opener) to comply
        with preflight check 0 (no bare-sqlite usage inside ``components/``).
        """
        try:
            from components.db import safe_open_kdb
            conn = safe_open_kdb(self.db_path, timeout=5)
            try:
                conn.execute(
                    "UPDATE se_edits SET status=?, decided_ts=?, decided_by=? "
                    "WHERE id=?",
                    (
                        status,
                        datetime.now(timezone.utc).isoformat(),
                        "capability_self_tester",
                        edit_id,
                    ),
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except sqlite3.Error:
            # Non-fatal — caller still gets a TestResult.
            return
        except Exception:  # noqa: BLE001
            # safe_open_kdb may raise non-sqlite errors on disk corruption.
            return
        # Append a log line for observability.
        self._log_event(
            "status_update",
            {
                "edit_id": edit_id,
                "status": status,
                "failure_count": len(failures),
            },
        )

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        log_dir = Path(self.data_path) / "self_healing"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "capability_self_tester.log.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    **payload,
                }) + "\n")
        except OSError:
            pass

    def _requeue_with_context(
        self,
        spec,
        failures: List[CaseResult],
        retry_count: int,
    ) -> None:
        """Feed failure diffs back through spec-gen + impl for another try."""
        from components.capability_gap_entry import CapabilityGapEntry
        from components.capability_spec_generator import CapabilitySpecGenerator
        from components.capability_implementer import CapabilityImplementer

        gap = CapabilityGapEntry(
            name=getattr(spec, "source_gap", "") or spec.target_path,
            description=getattr(spec, "description", "") or "",
            priority=1,
            evidence_source="self_tester:retry",
            target_kpi="",
            retry_count=retry_count,
            extra={"failure_context": [f.diff for f in failures]},
        )
        gen = CapabilitySpecGenerator()
        new_spec = gen.generate(gap)
        impl = CapabilityImplementer(data_path=self.data_path)
        impl.implement(new_spec)
