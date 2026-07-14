"""Tests for components.procurement.config (PR K)."""
from __future__ import annotations

from components.procurement import config as cfg


def test_constants_exist():
    assert cfg.ELEC_RATE_GBP_PER_KWH == 0.27
    assert cfg.TCO_HORIZON_YEARS == 3
    assert cfg.HEADROOM_MULTIPLIER == 2.0
    assert cfg.CPU_SECONDS_PER_CORE_DAY == 86400
    # Four v1 sources, one parser stub each.
    assert len(cfg.SOURCES) == 4
    keys = {s["key"] for s in cfg.SOURCES}
    assert keys == {"serve_the_home", "techpowerup",
                    "newegg_us", "amazon_uk"}


def test_tco_math_known_input():
    # capex £500, idle 10W over 3 years:
    #   kwh = 10 * 24 * 365 * 3 / 1000 = 262.8
    #   opex = 262.8 * 0.27 = 70.956
    #   tco  = 500 + 70.956 = 570.96 (rounded)
    assert cfg.tco_gbp_3yr(500.0, 10.0) == 570.96
    # Zero idle -> pure capex.
    assert cfg.tco_gbp_3yr(300.0, 0.0) == 300.0


def test_headroom_multiplier_applied():
    from components.procurement import researcher as r
    # RAM: 2x peak RSS in GB. 1024 MB peak -> 2 GB required.
    assert r.required_ram_gb(1024.0) == 2.0
    # CPU: one core saturated (86400 CPU-s/day) -> util 1.0 ->
    # inferred passmark == baseline; required == 2x baseline.
    assert r.inferred_cpu_score(cfg.CPU_SECONDS_PER_CORE_DAY) == \
        cfg.BASELINE_RENDER_PASSMARK
    assert r.required_passmark(cfg.CPU_SECONDS_PER_CORE_DAY) == \
        cfg.HEADROOM_MULTIPLIER * cfg.BASELINE_RENDER_PASSMARK
