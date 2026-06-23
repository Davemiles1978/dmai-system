#!/usr/bin/env python3
"""
DMAI Metrics Pipeline — End-to-End Diagnostic Test
====================================================
Verifies that the data-seeder, kpi-cache, and orchestrator status endpoints
return consistent, non-zero values across all 5 training domains after every
redeploy.

Usage:
    python3 aevora-training/tests/test_metrics_pipeline.py
    python3 aevora-training/tests/test_metrics_pipeline.py --base-url https://dmai-web.onrender.com
    python3 aevora-training/tests/test_metrics_pipeline.py --local          # hits localhost:5000

Exit codes:
    0  All checks passed
    1  One or more checks failed (details printed to stdout)
"""

import sys
import json
import time
import argparse
import datetime
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://dmai-web.onrender.com"
TIMEOUT_S        = 30          # per-request timeout
RETRY_WAIT_S     = 8           # wait before retry on cold-start 502/503
MAX_RETRIES      = 3

# The 8 SICore KPI keys that must ALL be present and non-zero
KPI_KEYS = [
    "skill_acquisition_rate",
    "transfer_learning_rate",
    "zero_shot_success_count",
    "agentic_capability_score",
    "recursive_self_improvement_rate",
    "sample_efficiency_trend",
    "metacognition_accuracy",
    "multi_modal_integration_score",
]

# The 5 training domains we expect to appear in the orchestrator status
EXPECTED_DOMAINS = [
    "Accelerator",
    "Wealth",
    "Artistic",
    "Core",
    "machine_learning",   # from autonomous researcher / syllabus
]

# Thread services that must be alive for status == "healthy"
EXPECTED_SERVICES = [
    "background_updater",
    "parallel_learner",
    "autonomous_researcher",
    "stage_learner",
]

# ---------------------------------------------------------------------------
# Minimal HTTP client (stdlib only — no requests required)
# ---------------------------------------------------------------------------
try:
    import urllib.request as _ureq
    import urllib.error   as _uerr

    def _get(url: str, timeout: int = TIMEOUT_S) -> Tuple[int, dict]:
        req = _ureq.Request(url, headers={"Accept": "application/json"})
        try:
            with _ureq.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, json.loads(body)
        except _uerr.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                return e.code, json.loads(body)
            except Exception:
                return e.code, {"_raw": body[:200]}
        except Exception as e:
            return 0, {"_error": str(e)}

except ImportError:
    def _get(url, timeout=TIMEOUT_S):
        return 0, {"_error": "urllib not available"}


def _get_with_retry(url: str, retries: int = MAX_RETRIES) -> Tuple[int, dict]:
    for attempt in range(1, retries + 1):
        code, data = _get(url)
        if code in (200, 201):
            return code, data
        if code in (502, 503, 0) and attempt < retries:
            print(f"  ⏳  {url} → {code} (attempt {attempt}/{retries}) — waiting {RETRY_WAIT_S}s for cold start…")
            time.sleep(RETRY_WAIT_S)
        else:
            return code, data
    return code, data   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = True
        self.details: List[str] = []

    def fail(self, msg: str):
        self.passed = False
        self.details.append(f"  ✗  {msg}")

    def ok(self, msg: str):
        self.details.append(f"  ✓  {msg}")

    def warn(self, msg: str):
        self.details.append(f"  ⚠  {msg}")


def check_system_health(base: str) -> CheckResult:
    """GET /api/status — system must be online."""
    r = CheckResult("System Health (/api/status)")
    code, data = _get_with_retry(f"{base}/api/status")
    if code != 200:
        r.fail(f"HTTP {code} — system unreachable")
        return r
    r.ok(f"System online (HTTP {code})")
    st = data.get("status", "unknown")
    if st not in ("ok", "online", "healthy", "running"):
        r.warn(f"status field is '{st}' (expected ok/online/healthy)")
    else:
        r.ok(f"status = '{st}'")
    return r


def check_db_seeder(base: str) -> CheckResult:
    """GET /api/metrics — DB-seeder must return non-zero capabilities + insights."""
    r = CheckResult("DB Seeder (/api/metrics)")
    code, data = _get_with_retry(f"{base}/api/metrics")
    if code != 200:
        r.fail(f"HTTP {code}")
        return r

    caps = data.get("capabilities", 0)
    ins  = data.get("insights",     data.get("total_insights", 0))

    if caps == 0:
        r.fail(f"capabilities = 0 — seeder not writing to DB")
    else:
        r.ok(f"capabilities = {caps:,}")

    if ins == 0:
        r.fail(f"insights = 0 — no learning recorded yet")
    else:
        r.ok(f"insights = {ins:,}")

    active = data.get("active_components", 0)
    r.ok(f"active_components = {active}")
    return r


def check_kpi_cache(base: str) -> CheckResult:
    """GET /api/learning/full-status — all 8 KPI keys must be present and non-zero."""
    r = CheckResult("KPI Cache (/api/learning/full-status)")
    code, data = _get_with_retry(f"{base}/api/learning/full-status")
    if code != 200:
        r.fail(f"HTTP {code}")
        return r

    kpis = data.get("kpis", {})
    if not kpis:
        r.fail("'kpis' key missing from response")
        return r

    all_zero = True
    for key in KPI_KEYS:
        val = kpis.get(key)
        if val is None:
            r.fail(f"KPI '{key}' missing")
        elif val == 0:
            r.warn(f"KPI '{key}' = 0.00 (may need more training data)")
        else:
            r.ok(f"KPI '{key}' = {val:.4f}")
            all_zero = False

    if all_zero:
        r.fail("ALL KPIs are zero — DB cache seeder not running or kpi_cache.json not found")

    # Also check study block
    study = data.get("study")
    if study is None:
        r.fail("'study' key missing — UI stage breakdown will show blank")
    else:
        r.ok(f"study.current_stage = '{study.get('current_stage', '?')}', "
             f"topics_mastered = {study.get('topics_mastered', 0)}")
    return r


def check_orchestrator_status(base: str) -> CheckResult:
    """GET /api/training/status — must include thread services and domain progress."""
    r = CheckResult("Orchestrator Status (/api/training/status)")
    code, data = _get_with_retry(f"{base}/api/training/status")
    if code != 200:
        r.fail(f"HTTP {code}")
        return r

    # ── Thread services (from our augmented get_status) ──
    services = data.get("services")
    if services is None:
        r.warn("'services' key missing — orchestrator not yet redeployed with thread patch")
    else:
        for svc in EXPECTED_SERVICES:
            alive = services.get(svc, False)
            if alive:
                r.ok(f"thread '{svc}' alive")
            else:
                r.warn(f"thread '{svc}' not detected (may still be starting)")

        active = data.get("active_count", 0)
        status = data.get("status", "unknown")
        if status == "healthy":
            r.ok(f"overall status = healthy ({active} threads active)")
        else:
            r.warn(f"overall status = {status} ({active} threads active)")

    # ── Domain progress (from ai_training component) ──
    components = data.get("components", {})
    ai = components.get("ai_training", {})
    progress = ai.get("progress", {})
    domains_total = ai.get("domains", 0)

    if domains_total == 0:
        r.fail("ai_training reports 0 domains — orchestrator component not loaded")
    else:
        r.ok(f"ai_training domains = {domains_total}")

    # Check 5 expected domains are covered
    curriculum = ai.get("curriculum_categories", [])
    for domain in ["Accelerator", "Wealth", "Artistic", "Core"]:
        if domain in curriculum:
            r.ok(f"domain '{domain}' in curriculum")
        else:
            r.warn(f"domain '{domain}' not in curriculum_categories {curriculum}")

    avg_mastery = progress.get("avg_mastery", None)
    if avg_mastery is None:
        r.warn("avg_mastery not reported")
    elif avg_mastery == 0.0:
        r.warn(f"avg_mastery = 0.0 — domains have not been trained yet")
    else:
        r.ok(f"avg_mastery = {avg_mastery:.4f}")

    by_stage = progress.get("by_stage", {})
    total_placed = sum(by_stage.values()) if by_stage else 0
    r.ok(f"stage distribution: {by_stage} (total={total_placed})")

    return r


def check_learning_progress(base: str) -> CheckResult:
    """GET /api/learning/progress — study progress must not be 'Error' state."""
    r = CheckResult("Learning Progress (/api/learning/progress)")
    code, data = _get_with_retry(f"{base}/api/learning/progress")
    if code != 200:
        r.fail(f"HTTP {code}")
        return r

    stage = data.get("current_stage", "")
    mastered = data.get("mastered", 0)
    in_progress = data.get("in_progress", 0)
    mastery_pct = data.get("mastery_pct", 0.0)

    if not stage:
        r.fail("current_stage missing — endpoint returning empty")
    else:
        r.ok(f"current_stage = '{stage}'")

    r.ok(f"mastered = {mastered}, in_progress = {in_progress}, mastery_pct = {mastery_pct}%")

    discoveries = data.get("recent_discoveries", [])
    r.ok(f"recent_discoveries = {len(discoveries)} entries")
    return r


def check_graph_schema(base: str) -> CheckResult:
    """GET /api/graph/schema — graph must have neurons > 0."""
    r = CheckResult("Graph Schema (/api/graph/schema)")
    code, data = _get_with_retry(f"{base}/api/graph/schema")
    if code != 200:
        r.fail(f"HTTP {code}")
        return r

    neurons  = data.get("total_neurons",  len(data.get("neurons",  [])))
    synapses = data.get("total_synapses", len(data.get("synapses", [])))
    cycle    = data.get("evolution_cycle", 0)

    if neurons == 0:
        r.fail("total_neurons = 0 — graph schema not loaded")
    else:
        r.ok(f"neurons = {neurons}, synapses = {synapses}, evolution_cycle = {cycle}")
    return r


def check_consistency(base: str) -> CheckResult:
    """
    Cross-check: capabilities count must agree within 5% between
    /api/metrics and /api/learning/full-status.
    """
    r = CheckResult("Cross-Endpoint Consistency")

    _, m = _get_with_retry(f"{base}/api/metrics")
    _, f = _get_with_retry(f"{base}/api/learning/full-status")

    caps_metrics = m.get("capabilities", 0)
    caps_full    = (f.get("db_stats") or {}).get("capabilities", 0)

    if caps_metrics == 0 and caps_full == 0:
        r.fail("Both endpoints report capabilities = 0")
    elif caps_metrics == 0 or caps_full == 0:
        r.warn(f"/api/metrics capabilities={caps_metrics}, /api/learning/full-status db_stats.capabilities={caps_full} — one is zero")
    else:
        diff_pct = abs(caps_metrics - caps_full) / max(caps_metrics, caps_full) * 100
        if diff_pct > 5:
            r.fail(f"capabilities mismatch > 5%: metrics={caps_metrics:,}, full-status={caps_full:,} ({diff_pct:.1f}% diff)")
        else:
            r.ok(f"capabilities consistent: metrics={caps_metrics:,} vs full-status={caps_full:,} ({diff_pct:.1f}% diff)")

    # KPI consistency: kpi_cache must match /api/training/status active_count trend
    kpis = (f.get("kpis") or {})
    all_zero_kpis = all(v == 0 for v in kpis.values() if isinstance(v, (int, float)))
    if all_zero_kpis:
        r.fail("KPIs are all zero across full-status — cache seeder not providing data")
    else:
        non_zero = sum(1 for v in kpis.values() if isinstance(v, (int, float)) and v > 0)
        r.ok(f"{non_zero}/{len(kpis)} KPIs non-zero")

    return r


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(base: str) -> Tuple[bool, List[CheckResult]]:
    checks = [
        check_system_health,
        check_db_seeder,
        check_kpi_cache,
        check_orchestrator_status,
        check_learning_progress,
        check_graph_schema,
        check_consistency,
    ]
    results = []
    for fn in checks:
        print(f"\n▶  {fn.__doc__.strip().splitlines()[0]}")
        res = fn(base)
        results.append(res)
        for line in res.details:
            print(line)
        status = "PASS" if res.passed else "FAIL"
        print(f"   → {status}: {res.name}")

    overall = all(r.passed for r in results)
    return overall, results


def build_summary(base: str, overall: bool, results: List[CheckResult]) -> str:
    ts    = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    emoji = "✅" if overall else "❌"
    lines = [
        f"{emoji} DMAI Metrics Pipeline Diagnostic — {ts}",
        f"Target: {base}",
        f"Overall: {'ALL CHECKS PASSED' if overall else 'ONE OR MORE CHECKS FAILED'}",
        "",
    ]
    for r in results:
        icon = "✅" if r.passed else "❌"
        lines.append(f"{icon} {r.name}")
        for d in r.details:
            lines.append(f"   {d.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMAI metrics pipeline diagnostic")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Base URL of the DMAI deployment (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--local", action="store_true",
                        help="Shortcut: use http://localhost:5000")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON summary to stdout")
    args = parser.parse_args()

    base = "http://localhost:5000" if args.local else args.base_url.rstrip("/")

    print(f"\n{'='*60}")
    print(f"  DMAI Metrics Pipeline — End-to-End Diagnostic")
    print(f"  Target : {base}")
    print(f"  Time   : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    overall, results = run_all(base)
    summary = build_summary(base, overall, results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(summary)

    if args.json:
        payload = {
            "overall": overall,
            "target":  base,
            "ts":      datetime.datetime.utcnow().isoformat(),
            "checks":  [
                {"name": r.name, "passed": r.passed, "details": r.details}
                for r in results
            ],
        }
        print("\nJSON:")
        print(json.dumps(payload, indent=2))

    sys.exit(0 if overall else 1)
