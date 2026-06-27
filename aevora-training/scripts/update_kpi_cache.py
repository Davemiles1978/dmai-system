#!/usr/bin/env python3
"""
update_kpi_cache.py
===================
Fetches live KPI values from a deployed DMAI instance and writes a
kpi_cache.json file that the aevora-training dashboard reads.

This script is called by the GitHub Actions CI workflow ONLY after
all diagnostic tests pass — it is the production gate between a new
dataset version and the KPI baseline.

Usage:
    python3 aevora-training/scripts/update_kpi_cache.py \
        --base-url https://dmai-web.onrender.com \
        --output aevora-training/dashboard/data/kpi_cache.json

Exit codes:
    0  Cache written successfully
    1  Could not fetch valid KPIs from the deployment
"""

import sys
import json
import argparse
import datetime
import urllib.request as _req
import urllib.error   as _err

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://dmai-web.onrender.com"
DEFAULT_OUTPUT   = "aevora-training/dashboard/data/kpi_cache.json"
TIMEOUT          = 30
MAX_RETRIES      = 3
RETRY_WAIT       = 10

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


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────
def _get(url: str) -> tuple[int, dict]:
    try:
        with _req.urlopen(_req.Request(url, headers={"Accept": "application/json"}),
                          timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read().decode())
    except _err.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as exc:
        return 0, {"_error": str(exc)}


def _get_retry(url: str) -> tuple[int, dict]:
    import time
    for attempt in range(1, MAX_RETRIES + 1):
        code, data = _get(url)
        if code == 200:
            return code, data
        if attempt < MAX_RETRIES:
            print(f"  ⏳  {url} → HTTP {code} (attempt {attempt}) — retrying in {RETRY_WAIT}s")
            time.sleep(RETRY_WAIT)
    return code, data  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# KPI fetching — tries multiple endpoints, takes best available data
# ─────────────────────────────────────────────────────────────────────────────
def fetch_kpis(base: str) -> dict:
    """
    Fetch KPI values from the live deployment.
    Priority:
      1. /api/learning/full-status  → kpis (DB-seeded, most accurate)
      2. /api/metrics               → raw DB counts → compute KPI proxies
    Returns a dict of KPI key → float, or raises RuntimeError.
    """
    print(f"  Fetching KPIs from {base}/api/learning/full-status ...")
    code, data = _get_retry(f"{base}/api/learning/full-status")

    kpis: dict = {}

    if code == 200:
        raw = data.get("kpis", {})
        non_zero = {k: v for k, v in raw.items()
                    if k in KPI_KEYS and isinstance(v, (int, float)) and v > 0}
        if len(non_zero) >= 4:
            print(f"  ✅  Got {len(non_zero)}/8 non-zero KPIs from full-status")
            kpis = {k: float(raw.get(k, 0.0)) for k in KPI_KEYS}
        else:
            print(f"  ⚠️  Only {len(non_zero)}/8 KPIs non-zero from full-status — trying /api/metrics fallback")

    if not kpis or all(v == 0 for v in kpis.values()):
        print(f"  Fetching raw counts from {base}/api/metrics ...")
        code2, metrics = _get_retry(f"{base}/api/metrics")
        if code2 != 200:
            raise RuntimeError(f"Both /api/learning/full-status and /api/metrics failed "
                               f"(HTTP {code} / {code2})")

        caps     = metrics.get("capabilities", 0)
        insights = metrics.get("insights", metrics.get("total_insights", 0))
        active   = metrics.get("active_components", 0)

        if caps == 0 and insights == 0:
            raise RuntimeError("Raw DB counts are also zero — deployment may not have trained data yet")

        # Derive KPI proxies from raw counts (same formulas as _seed_kpis_from_db)
        kpis = {
            "skill_acquisition_rate":          min(caps      / 50_000, 1.0),
            "transfer_learning_rate":           0.0,   # can't derive stage from here
            "zero_shot_success_count":          min(insights  / 300_000, 1.0),
            "agentic_capability_score":         min(caps      / 20_000, 1.0),
            "recursive_self_improvement_rate":  0.0,   # needs stage_within_pct
            "sample_efficiency_trend":          0.0,   # needs 7-day avg
            "metacognition_accuracy":           0.0,   # needs vocab count
            "multi_modal_integration_score":    min(active    / 56, 1.0),
        }
        print(f"  ✅  Derived KPI proxies from raw counts "
              f"(caps={caps:,}, insights={insights:,})")

    return kpis


def fetch_metadata(base: str) -> dict:
    """Fetch supplementary metadata to store alongside KPIs."""
    meta: dict = {}

    # Graph stats
    code, g = _get_retry(f"{base}/api/graph/schema")
    if code == 200:
        meta["graph"] = {
            "total_neurons":  g.get("total_neurons",  len(g.get("neurons",  []))),
            "total_synapses": g.get("total_synapses", len(g.get("synapses", []))),
            "evolution_cycle": g.get("evolution_cycle", 0),
            "last_updated":    g.get("last_updated", ""),
        }

    # Learning progress
    code, lp = _get_retry(f"{base}/api/learning/progress")
    if code == 200:
        meta["learning"] = {
            "current_stage":    lp.get("current_stage", ""),
            "mastery_pct":      lp.get("mastery_pct", 0.0),
            "mastered":         lp.get("mastered", 0),
            "in_progress":      lp.get("in_progress", 0),
            "last_cycle":       lp.get("last_learning_cycle", ""),
        }

    # Raw DB counts
    code, m = _get_retry(f"{base}/api/metrics")
    if code == 200:
        meta["db"] = {
            "capabilities":      m.get("capabilities", 0),
            "insights":          m.get("insights", m.get("total_insights", 0)),
            "active_components": m.get("active_components", 0),
        }

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch live KPIs and write kpi_cache.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output",   default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print the cache content without writing to disk")
    args = parser.parse_args()

    base   = args.base_url.rstrip("/")
    output = args.output

    print(f"\n{'='*60}")
    print(f"  DMAI KPI Cache Updater")
    print(f"  Source : {base}")
    print(f"  Output : {output}")
    print(f"  Time   : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # ── Fetch ──────────────────────────────────────────────────────────────
    try:
        kpis = fetch_kpis(base)
    except RuntimeError as e:
        print(f"\n❌  KPI fetch failed: {e}")
        sys.exit(1)

    print("\nFetching supplementary metadata...")
    meta = fetch_metadata(base)

    # ── Build cache payload ────────────────────────────────────────────────
    now = datetime.datetime.utcnow()
    cache = {
        "kpis": kpis,
        "ts":   now.isoformat(),
        "source": base,
        "generated_by": "aevora-training/scripts/update_kpi_cache.py",
        "ci_context": {
            "github_sha":    "",   # populated below if env var present
            "github_run_id": "",
        },
        "metadata": meta,
    }

    import os
    cache["ci_context"]["github_sha"]    = os.environ.get("GITHUB_SHA", "")[:8]
    cache["ci_context"]["github_run_id"] = os.environ.get("GITHUB_RUN_ID", "")

    # ── Print / write ──────────────────────────────────────────────────────
    payload_str = json.dumps(cache, indent=2)

    print("\n=== KPI Cache Payload ===")
    print(payload_str)

    if args.dry_run:
        print("\n⚠️  Dry-run mode — file not written")
        sys.exit(0)

    import pathlib
    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload_str)

    print(f"\n✅  kpi_cache.json written to {output}")
    print(f"    KPIs: {', '.join(f'{k}={v:.4f}' for k, v in kpis.items())}")


if __name__ == "__main__":
    main()
