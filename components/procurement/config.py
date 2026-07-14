"""Configuration constants for the procurement research skill (PR K).

All the sizing / pricing knobs live here as module-level constants so a
future operator can bump them in one place without touching the
researcher logic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

# ── DB path ─────────────────────────────────────────────────────────────────

PROCUREMENT_DB_FILENAME = "dmai_procurement.db"


def default_procurement_path() -> str:
    """Return the default procurement DB path (respects DATA_PATH)."""
    base = os.environ.get("DATA_PATH", "data")
    p = Path(base) / PROCUREMENT_DB_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ── Pricing / TCO knobs ──────────────────────────────────────────────────────

# UK domestic electricity unit rate. Jul 2026 average for a single-rate
# residential tariff is ~£0.27/kWh (Ofgem price cap). Bump this one line
# when the cap changes.
ELEC_RATE_GBP_PER_KWH = 0.27

# TCO is amortised over a 3-year ownership horizon (typical refresh cycle
# for a small always-on home-lab box).
TCO_HORIZON_YEARS = 3

# Size for 2x the current measured workload profile (RSS + CPU-seconds/day)
# so the box has room to grow before it needs replacing.
HEADROOM_MULTIPLIER = 2.0

# Baseline PassMark of the shared vCPU DMAI currently runs on (Render
# standard instance ~= a fraction of a modern desktop CPU). Used to turn
# measured CPU-seconds/day into a target PassMark. ~8000 is a conservative
# single-thread-ish estimate for the shared render CPU slice.
BASELINE_RENDER_PASSMARK = 8000

# Seconds of CPU time that equals one core saturated for a full day.
CPU_SECONDS_PER_CORE_DAY = 24 * 3600  # 86400

# Fallback USD -> GBP rate if the treasury config can't be read. Kept in
# step with components.treasury.treasury_ledger.DEFAULT_USD_TO_GBP.
DEFAULT_FX_USD_GBP = 0.77

# A shortlist run tags its top-3 as 'affordable' when their combined capex
# is within this multiple of the current treasury balance.
AFFORDABILITY_MULTIPLIER = 1.5


def fx_usd_gbp() -> float:
    """USD -> GBP rate. Prefer the treasury's live rate; fall back to the
    module default if the treasury DB isn't available (e.g. under test)."""
    try:
        from components.treasury import treasury_ledger as _tl
        rate = _tl.get_fx_usd_gbp()
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    return DEFAULT_FX_USD_GBP


# ── Data sources (v1) ────────────────────────────────────────────────────────
#
# One parser stub per source. ``module`` is the dotted path of the parser
# module under components.procurement.parsers; ``currency`` is the native
# currency of the prices that source publishes.

SOURCES: List[Dict[str, Any]] = [
    {
        "key":      "serve_the_home",
        "name":     "ServeTheHome + STH forum",
        "module":   "components.procurement.parsers.serve_the_home",
        "currency": "GBP",   # editorial/spec source; prices normalised to GBP
        "region":   "review",
    },
    {
        "key":      "techpowerup",
        "name":     "TechPowerUp CPU DB",
        "module":   "components.procurement.parsers.techpowerup",
        "currency": "GBP",   # spec/PassMark source; no retail price
        "region":   "spec",
    },
    {
        "key":      "newegg_us",
        "name":     "Newegg (US)",
        "module":   "components.procurement.parsers.newegg_us",
        "currency": "USD",
        "region":   "US",
    },
    {
        "key":      "amazon_uk",
        "name":     "Amazon UK",
        "module":   "components.procurement.parsers.amazon_uk",
        "currency": "GBP",
        "region":   "UK",
    },
]


def tco_gbp_3yr(capex_gbp: float, idle_w: float) -> float:
    """Total cost of ownership over :data:`TCO_HORIZON_YEARS` years.

    capex plus electricity for an always-on box at idle wattage::

        tco = capex + (idle_w * 24 * 365 * horizon / 1000) * rate

    Idle wattage dominates for a 24/7 box; the caller should pass load
    wattage only when idle isn't published.
    """
    kwh = (float(idle_w) * 24.0 * 365.0 * TCO_HORIZON_YEARS) / 1000.0
    return round(float(capex_gbp) + kwh * ELEC_RATE_GBP_PER_KWH, 2)


__all__ = [
    "PROCUREMENT_DB_FILENAME",
    "default_procurement_path",
    "ELEC_RATE_GBP_PER_KWH",
    "TCO_HORIZON_YEARS",
    "HEADROOM_MULTIPLIER",
    "BASELINE_RENDER_PASSMARK",
    "CPU_SECONDS_PER_CORE_DAY",
    "DEFAULT_FX_USD_GBP",
    "AFFORDABILITY_MULTIPLIER",
    "fx_usd_gbp",
    "SOURCES",
    "tco_gbp_3yr",
]
