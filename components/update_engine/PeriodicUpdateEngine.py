"""
DMAI PeriodicUpdateEngine — Patched v2
=======================================
Fixes applied vs original:
  - _benchmark(): 15% regression threshold, CRITICAL/HIGH severity, requires_human_review flag
  - _save_baseline(): atomic write (temp+rename) + SHA-256 hash guard
  - _load_baseline(): hash verification, trips CB-02 on mismatch
  - _job_kaizen(): queue depth check — refuses to submit when queue >= 20
"""

import os
import json
import hashlib
import logging
import asyncio
import tempfile
import httpx

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using temp file + os.replace() pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, suffix='.tmp',
        delete=False, encoding='utf-8'
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _compute_hash(data: dict) -> str:
    """Compute SHA-256 of canonicalised JSON dict."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# PerformanceBenchmark (patched)
# ---------------------------------------------------------------------------

class PerformanceBenchmark:
    """
    Runs reference prompts through DMAI and compares keyword coverage.
    Flags regressions (>= 15% drop) with severity labels.
    Uses atomic writes + SHA-256 guard on benchmark_baseline.json.
    NEVER triggers auto-retraining.
    """

    REGRESSION_THRESHOLD = 0.15   # 15% drop triggers HIGH alert
    CRITICAL_THRESHOLD   = 0.30   # 30% drop triggers CRITICAL alert

    def __init__(self, config: Dict, data_path: Path, ai_hub=None, si_core=None):
        """Initialise benchmark with config, paths, and optional AI hub."""
        self.config        = config
        self.ai_hub        = ai_hub
        self.si_core       = si_core
        self.baseline_file = data_path / "benchmark_baseline.json"
        self.hash_file     = data_path / "benchmark_baseline.hash"
        self.baseline: Dict = self._load_baseline()

    def _load_baseline(self) -> Dict:
        """Load baseline with SHA-256 integrity check. Trips CB-02 on mismatch."""
        if not self.baseline_file.exists():
            return {}
        try:
            data = json.loads(self.baseline_file.read_text())
        except Exception as e:
            logger.error("Failed to read baseline file: %s", e)
            return {}

        if self.hash_file.exists():
            expected = self.hash_file.read_text().strip()
            actual   = _compute_hash(data)
            if actual != expected:
                logger.critical(
                    "INTEGRITY ALERT: benchmark_baseline.json hash mismatch! "
                    "Expected %s... got %s... Possible tampering. Refusing to load.",
                    expected[:16], actual[:16]
                )
                try:
                    from circuit_breaker import CircuitBreakerManager
                    CircuitBreakerManager.get().breakers["CB-02"].trip(
                        "benchmark_baseline.json hash mismatch"
                    )
                except Exception:
                    pass
                return {}   # refuse tampered data
        return data

    def _save_baseline(self, results: Dict) -> None:
        """Save baseline atomically + write SHA-256 hash file."""
        _atomic_write_json(self.baseline_file, results)
        h = _compute_hash(results)
        self.hash_file.write_text(h)
        logger.info("Benchmark baseline saved. Hash prefix: %s", h[:16])

    async def run(self) -> Dict:
        """Run all benchmark prompts and return regression report."""
        prompts    = self.config.get("benchmark_prompts", [])
        scores     = []
        regressions = []

        for item in prompts:
            prompt   = item["prompt"]
            expected = item.get("expected_keywords", [])
            response = await self._query(prompt)

            if response is None:
                logger.info("[BENCHMARK] Skipped prompt (no ai_hub): '%s'", prompt[:50])
                continue

            rl    = response.lower()
            score = (
                sum(1 for kw in expected if kw.lower() in rl) / len(expected)
                if expected else 0.5
            )

            prev_score = self.baseline.get(prompt, {}).get("score", score)
            drop = (prev_score - score) / prev_score if prev_score > 0 else 0.0

            if drop >= self.REGRESSION_THRESHOLD:
                severity = "CRITICAL" if drop >= self.CRITICAL_THRESHOLD else "HIGH"
                regression = {
                    "prompt":                   prompt,
                    "prev":                     round(prev_score, 4),
                    "current":                  round(score, 4),
                    "drop_pct":                 round(drop * 100, 1),
                    "severity":                 severity,
                    "auto_retraining_triggered": False,
                    "requires_human_review":     True,
                    "timestamp":                datetime.now(timezone.utc).isoformat(),
                }
                regressions.append(regression)
                logger.warning(
                    "REGRESSION [%s]: '%s' dropped %.1f%% (%.3f -> %.3f). "
                    "Auto-retraining NOT triggered. Human review required.",
                    severity, prompt[:50], drop * 100, prev_score, score
                )

            scores.append(score)
            self.baseline[prompt] = {
                "score":     score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if not scores:
            logger.info("[BENCHMARK] No prompts scored this run (no ai_hub connected)")
            return {
                "avg_score":    None,
                "regressions":  [],
                "prompts_run":  0,
                "status":       "skipped",
                "reason":       "no_ai_provider",
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }

        avg_score = sum(scores) / len(scores)
        self._save_baseline(self.baseline)

        if self.si_core and regressions:
            # Do NOT auto-retrain. Log regressions only.
            for r in regressions:
                logger.warning(
                    "KPI review needed [%s]: %s", r["severity"], r["prompt"][:60]
                )

        return {
            "avg_score":    round(avg_score, 3),
            "regressions":  regressions,
            "prompts_run":  len(prompts),
            "status":       "completed",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    async def _query(self, prompt: str) -> Optional[str]:
        """Query ai_hub. Returns None if no provider available."""
        if not self.ai_hub or not hasattr(self.ai_hub, "chat"):
            return None
        try:
            return await self.ai_hub.chat(prompt)
        except Exception as e:
            logger.warning("[BENCHMARK SKIP] ai_hub.chat failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# KaizenIntegrator (patched) — queue depth guard
# ---------------------------------------------------------------------------

class KaizenIntegrator:
    """
    Submits improvement proposals to /api/kaizen.
    Refuses to submit when the queue already has >= 20 pending proposals.
    """

    MAX_QUEUE_DEPTH = 20

    def __init__(self, config: Dict):
        """Initialise with config dict containing kaizen_endpoint."""
        self.endpoint    = config.get("kaizen_endpoint", "http://localhost:5000/api/kaizen")
        self.kaizen_file = Path(config.get("data_path", "data")) / "kaizen_proposals.jsonl"

    def _current_queue_depth(self) -> int:
        """Count unprocessed proposals in the kaizen JSONL file."""
        if not self.kaizen_file.exists():
            return 0
        try:
            lines = [l for l in self.kaizen_file.read_text().strip().split("\n") if l.strip()]
            return len(lines)
        except Exception:
            return 0

    async def submit_proposal(self, proposal: Dict) -> bool:
        """Submit one proposal, respecting the 20-item queue cap."""
        depth = self._current_queue_depth()
        if depth >= self.MAX_QUEUE_DEPTH:
            logger.warning(
                "Kaizen queue at capacity (%d >= %d). Skipping proposal: %s",
                depth, self.MAX_QUEUE_DEPTH, proposal.get("title", "")[:60]
            )
            try:
                from circuit_breaker import CircuitBreakerManager
                CircuitBreakerManager.get().check_kaizen_depth(depth)
            except Exception:
                pass
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp    = await client.post(self.endpoint, json=proposal)
                success = resp.status_code in (200, 201, 202)
                if success:
                    logger.info(
                        "KaizenIntegrator: submitted — %s", proposal.get("title", "")
                    )
                return success
        except Exception as e:
            logger.warning("KaizenIntegrator: submit failed — %s", e)
            return False

    async def propose_from_results(
        self, benchmark_result: Dict, feedback_result: Dict
    ) -> Dict:
        """Generate and submit kaizen proposals from benchmark regressions."""
        submitted = 0
        proposals = []

        for regression in benchmark_result.get("regressions", []):
            severity = regression.get("severity", "HIGH")
            proposal = {
                "title":       f"Fix regression: {regression['prompt'][:40]}",
                "description": (
                    f"[{severity}] Performance dropped {regression.get('drop_pct', '?')}% "
                    f"({regression['prev']:.2f} -> {regression['current']:.2f}). "
                    f"Human review required. Auto-retraining NOT triggered."
                ),
                "priority":              "high" if severity == "HIGH" else "critical",
                "type":                  "regression_fix",
                "severity":              severity,
                "requires_human_review": True,
                "auto_retraining_triggered": False,
                "data":                  regression,
            }
            if await self.submit_proposal(proposal):
                submitted += 1
            proposals.append(proposal)

        return {
            "proposals_submitted": submitted,
            "proposals_total":     len(proposals),
            "queue_depth":         self._current_queue_depth(),
        }


# ---------------------------------------------------------------------------
# PeriodicUpdateEngine — top-level convenience wrapper expected by orchestrator
# ---------------------------------------------------------------------------

class PeriodicUpdateEngine:
    """Lightweight runner that combines PerformanceBenchmark + KaizenIntegrator
    into a single periodic update thread.

    The orchestrator imports this class and starts it as a background thread.
    Each tick runs a benchmark (if configured), and forwards any detected
    regressions to the Kaizen queue.
    """

    def __init__(self, config=None, data_path=None, ai_hub=None, si_core=None, knowledge_graph=None, **_extra_kwargs):
        from pathlib import Path as _Path
        self.config = config or {}
        self.data_path = _Path(data_path) if data_path else _Path(self.config.get("data_path", "data"))
        self.data_path.mkdir(parents=True, exist_ok=True)
        try:
            self.benchmark = PerformanceBenchmark(self.config, self.data_path, ai_hub=ai_hub, si_core=si_core)
        except Exception as e:
            logging.getLogger(__name__).warning(f"PerformanceBenchmark init failed: {e}")
            self.benchmark = None
        try:
            self.kaizen = KaizenIntegrator({**self.config, "data_path": str(self.data_path)})
        except Exception as e:
            logging.getLogger(__name__).warning(f"KaizenIntegrator init failed: {e}")
            self.kaizen = None
        self._stop = False

    def start(self):
        """Run a single benchmark/kaizen cycle synchronously (best effort).

        Background-thread loop is owned by the orchestrator; this method is
        the per-tick entry-point.
        """
        if not self.benchmark and not self.kaizen:
            return {"ok": False, "reason": "benchmark_and_kaizen_unavailable"}
        try:
            import asyncio
            results = {}
            if self.benchmark and hasattr(self.benchmark, "run"):
                results["benchmark"] = self.benchmark.run() if not asyncio.iscoroutinefunction(self.benchmark.run) else asyncio.run(self.benchmark.run())
            return {"ok": True, **results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop(self):
        self._stop = True

    # ---- Compatibility shims used by DMAITrainingOrchestrator ----------------
    async def run_once(self):
        """Async wrapper for orchestrator.run_update_only().

        Runs one synchronous tick in a thread to avoid blocking the event loop.
        """
        import asyncio
        return await asyncio.to_thread(self.start)

    def get_status(self) -> dict:
        """Lightweight status snapshot for /api/training/status."""
        return {
            "ok": True,
            "benchmark_loaded": bool(self.benchmark),
            "kaizen_loaded": bool(self.kaizen),
            "stopped": bool(getattr(self, "_stop", False)),
            "data_path": str(getattr(self, "data_path", "")),
        }

    async def _arun_start_loop(self):
        """Async wrapper called by orchestrator's background thread.

        Loops calling start() at a low cadence; this is what loop.run_until_complete
        expects. Without it, run_until_complete(self.update_engine.start()) crashes
        because start() returns a dict, not an awaitable.
        """
        import asyncio
        while not getattr(self, "_stop", False):
            try:
                await asyncio.to_thread(self.start)
            except Exception as _e:
                logging.getLogger(__name__).warning("PeriodicUpdateEngine tick failed: %s", _e)
            # 20-minute cadence by default; configurable via config['update_interval_sec'].
            await asyncio.sleep(int((self.config or {}).get("update_interval_sec", 1200)))
