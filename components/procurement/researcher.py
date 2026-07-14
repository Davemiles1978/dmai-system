"""Procurement research orchestrator (PR K).

One research run:

1.  Read DMAI's own workload footprint (PR J workload DB) and treasury
    balance (PR I treasury DB).
2.  For every configured source, load candidate rows via its parser stub
    (materialised ``parse`` if available, else the hand-written
    ``seed_fallback``).
3.  Normalise each row into a ``hardware_catalog`` row (USD → GBP via the
    configured FX rate; idle wattage defaults to TDP when idle isn't
    published).
4.  Compute 3-year TCO, apply the RAM + CPU headroom gates, rank the
    survivors by lowest TCO, tag verdicts against the treasury balance,
    and write the ``procurement_shortlist``.

Deterministic: given the same catalog + workload + balance the ranking
and verdicts are stable (ties broken by capex then name).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from components.procurement import config as cfg
from components.procurement.parsers import load_source_rows
from components.procurement.store import (
    ProcurementStore,
    STATE_LAST_RUN_TS,
    STATE_LAST_SUMMARY,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── workload + treasury reads ────────────────────────────────────────────────

def read_workload(workload_db_path: Optional[str] = None,
                  ) -> Optional[Dict[str, Any]]:
    """Return a workload snapshot the sizing math consumes.

    Reads the PR J workload DB directly. Returns None when there are no
    samples yet (the caller then skips the run — we won't size a box off
    an empty profile).
    """
    try:
        from components.workload import workload_profiler as wp
    except Exception as e:  # pragma: no cover - workload pkg should exist
        logger.warning("procurement: workload profiler unavailable: %s", e)
        return None

    rollup = wp.get_daily_rollup(days=7, db_path=workload_db_path)
    if not rollup:
        return None

    peak_rss_mb = 0.0
    cpu_seconds_delta = 0.0
    for day in rollup:
        prss = day.get("peak_rss_mb")
        if prss is not None:
            peak_rss_mb = max(peak_rss_mb, float(prss))
        csd = day.get("cpu_seconds_delta")
        if csd is not None:
            cpu_seconds_delta = max(cpu_seconds_delta, float(csd))

    if peak_rss_mb <= 0.0:
        # Fall back to the latest sample's RSS if the rollup lacked peaks.
        latest = wp.get_latest(db_path=workload_db_path)
        if latest and latest.get("mem_rss_mb"):
            peak_rss_mb = float(latest["mem_rss_mb"])

    if peak_rss_mb <= 0.0:
        return None

    return {
        "peak_rss_mb":        round(peak_rss_mb, 3),
        "cpu_seconds_delta":  round(cpu_seconds_delta, 3),
        "days": len(rollup),
    }


def read_treasury_balance(treasury_db_path: Optional[str] = None) -> float:
    try:
        from components.treasury import treasury_ledger as tl
        return float(tl.get_balance(db_path=treasury_db_path))
    except Exception as e:
        logger.warning("procurement: treasury balance unavailable: %s", e)
        return 0.0


# ── sizing helpers ────────────────────────────────────────────────────────────

def required_ram_gb(peak_rss_mb: float) -> float:
    """RAM headroom gate threshold: 2x peak RSS, in GB."""
    return cfg.HEADROOM_MULTIPLIER * float(peak_rss_mb) / 1024.0


def inferred_cpu_score(cpu_seconds_delta: float) -> float:
    """Turn measured CPU-seconds/day into a PassMark-equivalent score.

    current_utilisation = cpu_seconds_delta / 86400 (1.0 == one core
    saturated all day). Scale by the baseline PassMark of the shared
    Render CPU to estimate the PassMark DMAI currently uses.
    """
    util = float(cpu_seconds_delta) / float(cfg.CPU_SECONDS_PER_CORE_DAY)
    return util * float(cfg.BASELINE_RENDER_PASSMARK)


def required_passmark(cpu_seconds_delta: float) -> float:
    """CPU headroom gate threshold: 2x the inferred current PassMark."""
    return cfg.HEADROOM_MULTIPLIER * inferred_cpu_score(cpu_seconds_delta)


# ── normalisation ─────────────────────────────────────────────────────────────

def normalise_row(raw: Dict[str, Any], source_key: str,
                  fx: float) -> Optional[Dict[str, Any]]:
    """Turn a parser row into a hardware_catalog row (priced in GBP).

    Returns None if the row has no usable price after normalisation and
    isn't a pure spec row (spec rows carry price_gbp=None deliberately and
    are still stored for CPU enrichment, but they never enter the ranking).
    """
    name = raw.get("name")
    cpu = raw.get("cpu")
    if not name or not cpu:
        return None

    currency_orig = raw.get("currency_orig") or "GBP"
    price_orig = raw.get("price_orig")
    price_gbp = raw.get("price_gbp")

    if price_gbp is None and price_orig is not None:
        if currency_orig == "USD":
            price_gbp = round(float(price_orig) * fx, 2)
        else:
            price_gbp = round(float(price_orig), 2)

    idle_w = raw.get("idle_w")
    tdp_w = raw.get("tdp_w")
    # Idle dominates for an always-on box; fall back to TDP when idle
    # isn't published.
    effective_idle = idle_w if idle_w is not None else tdp_w

    return {
        "source":        source_key,
        "url":           raw.get("url"),
        "name":          name,
        "cpu":           cpu,
        "cpu_passmark":  raw.get("cpu_passmark"),
        "tdp_w":         tdp_w,
        "idle_w":        effective_idle,
        "ram_gb":        raw.get("ram_gb"),
        "storage_gb":    raw.get("storage_gb"),
        "price_gbp":     price_gbp,
        "currency_orig": currency_orig,
        "price_orig":    price_orig,
        "fx_used":       fx if currency_orig == "USD" else 1.0,
        "fetched_ts":    _now_iso(),
        "raw_json":      raw,
    }


# ── ranking ─────────────────────────────────────────────────────────────────

def _passes_headroom(row: Dict[str, Any], req_ram: float,
                     req_pm: float) -> Tuple[bool, Optional[float],
                                             Optional[float], str]:
    ram_gb = row.get("ram_gb")
    passmark = row.get("cpu_passmark")
    if not ram_gb or not passmark:
        return False, None, None, "missing ram/passmark"
    headroom_ram = round(float(ram_gb) / req_ram, 3) if req_ram > 0 else None
    headroom_cpu = (round(float(passmark) / (req_pm), 3)
                    if req_pm > 0 else None)
    if float(ram_gb) < req_ram:
        return False, headroom_ram, headroom_cpu, "ram below 2x peak RSS"
    if float(passmark) < req_pm:
        return False, headroom_ram, headroom_cpu, "passmark below 2x inferred"
    return True, headroom_ram, headroom_cpu, "ok"


def _verdict_for(rank: int, capex: float, top3_capex: float,
                 balance: float) -> str:
    """Verdict against the treasury balance.

    - affordable  : combined top-3 capex <= AFFORDABILITY_MULTIPLIER x balance
    - stretch     : this single item's capex <= balance (reachable soon)
    - aspirational: otherwise
    """
    if balance > 0 and top3_capex <= cfg.AFFORDABILITY_MULTIPLIER * balance:
        return "affordable"
    if balance > 0 and capex <= balance:
        return "stretch"
    return "aspirational"


# ── main entrypoint ─────────────────────────────────────────────────────────

def run_research(*,
                 procurement_db_path: Optional[str] = None,
                 workload_db_path: Optional[str] = None,
                 treasury_db_path: Optional[str] = None,
                 html_by_source: Optional[Dict[str, str]] = None,
                 ) -> Dict[str, Any]:
    """Execute one full research run. Returns a summary dict.

    ``html_by_source`` lets a caller (or test) inject page HTML keyed by
    source ``key``; absent that, parsers fall back to their seed rows.
    """
    store = ProcurementStore(procurement_db_path)
    store.init_db()
    run_ts = _now_iso()
    html_by_source = html_by_source or {}

    workload = read_workload(workload_db_path)
    if workload is None:
        summary = {
            "ok":        False,
            "run_ts":    run_ts,
            "skipped":   "no_workload_data",
            "shortlist": [],
        }
        store.set_state(STATE_LAST_RUN_TS, run_ts)
        store.set_state(STATE_LAST_SUMMARY, _json(summary))
        return summary

    balance = read_treasury_balance(treasury_db_path)
    fx = cfg.fx_usd_gbp()

    req_ram = required_ram_gb(workload["peak_rss_mb"])
    req_pm = required_passmark(workload["cpu_seconds_delta"])

    # 1. gather + normalise + persist catalog. Each run rebuilds the
    # catalog + shortlist from scratch (we only keep the latest run's
    # rows). Clear the shortlist first — it FKs to the catalog.
    store.clear_shortlist()
    store.clear_catalog()
    catalog: List[Dict[str, Any]] = []
    parser_errors: Dict[str, str] = {}
    for src in cfg.SOURCES:
        try:
            rows = load_source_rows(src["module"],
                                    html_by_source.get(src["key"], ""))
        except Exception as e:  # graceful degradation: skip a broken source
            logger.warning("procurement: source %s failed: %s",
                           src["key"], e)
            parser_errors[src["key"]] = str(e)
            continue
        for raw in rows:
            norm = normalise_row(raw, src["key"], fx)
            if norm is None:
                continue
            norm["id"] = store.insert_catalog(norm)
            catalog.append(norm)

    # 2. candidates = priced rows that pass headroom
    candidates: List[Dict[str, Any]] = []
    for row in catalog:
        if row.get("price_gbp") is None or row.get("idle_w") is None:
            continue
        ok, hr_ram, hr_cpu, reason = _passes_headroom(row, req_ram, req_pm)
        if not ok:
            continue
        capex = float(row["price_gbp"])
        idle_w = float(row["idle_w"])
        tco = cfg.tco_gbp_3yr(capex, idle_w)
        candidates.append({
            **row,
            "capex_gbp":      capex,
            "tco_gbp_3yr":    tco,
            "opex_3yr_gbp":   round(tco - capex, 2),
            "headroom_ram_x": hr_ram,
            "headroom_cpu_x": hr_cpu,
        })

    if not candidates:
        summary = {
            "ok":            True,
            "run_ts":        run_ts,
            "skipped":       "no_candidates_pass_headroom",
            "req_ram_gb":    round(req_ram, 3),
            "req_passmark":  round(req_pm, 1),
            "catalog_size":  len(catalog),
            "parser_errors": parser_errors,
            "shortlist":     [],
        }
        store.set_state(STATE_LAST_RUN_TS, run_ts)
        store.set_state(STATE_LAST_SUMMARY, _json(summary))
        return summary

    # 3. deterministic ranking: lowest TCO, then capex, then name
    candidates.sort(key=lambda r: (r["tco_gbp_3yr"], r["capex_gbp"],
                                   r["name"]))
    top3_capex = sum(c["capex_gbp"] for c in candidates[:3])

    # 4. write shortlist
    shortlist_out: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        verdict = _verdict_for(i, cand["capex_gbp"], top3_capex, balance)
        notes = (f"{cand['source']} | {cand['cpu']} | "
                 f"idle {cand['idle_w']}W | ram {cand['ram_gb']}GB")
        store.insert_shortlist_row({
            "run_ts":         run_ts,
            "rank":           i,
            "hardware_id":    cand["id"],
            "tco_gbp_3yr":    cand["tco_gbp_3yr"],
            "capex_gbp":      cand["capex_gbp"],
            "opex_3yr_gbp":   cand["opex_3yr_gbp"],
            "headroom_ram_x": cand["headroom_ram_x"],
            "headroom_cpu_x": cand["headroom_cpu_x"],
            "verdict":        verdict,
            "notes":          notes,
        })
        shortlist_out.append({
            "rank":         i,
            "name":         cand["name"],
            "source":       cand["source"],
            "capex_gbp":    cand["capex_gbp"],
            "tco_gbp_3yr":  cand["tco_gbp_3yr"],
            "verdict":      verdict,
        })

    summary = {
        "ok":             True,
        "run_ts":         run_ts,
        "catalog_size":   len(catalog),
        "candidate_count": len(candidates),
        "req_ram_gb":     round(req_ram, 3),
        "req_passmark":   round(req_pm, 1),
        "treasury_gbp":   round(balance, 2),
        "fx_usd_gbp":     fx,
        "workload":       workload,
        "parser_errors":  parser_errors,
        "shortlist":      shortlist_out[:3],
    }
    store.set_state(STATE_LAST_RUN_TS, run_ts)
    store.set_state(STATE_LAST_SUMMARY, _json(summary))
    return summary


def _json(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"


def get_last_summary(procurement_db_path: Optional[str] = None,
                     ) -> Dict[str, Any]:
    store = ProcurementStore(procurement_db_path)
    store.init_db()
    raw = store.get_state(STATE_LAST_SUMMARY)
    if not raw:
        return {}
    import json
    try:
        return json.loads(raw)
    except Exception:
        return {}


__all__ = [
    "run_research",
    "read_workload",
    "read_treasury_balance",
    "required_ram_gb",
    "required_passmark",
    "inferred_cpu_score",
    "normalise_row",
    "get_last_summary",
]
