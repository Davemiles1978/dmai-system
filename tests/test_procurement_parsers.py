"""Tests for the procurement parser stubs (PR K).

Each stub must (a) raise NotImplementedError from parse() until the
materialiser generates a body, (b) expose SEED_CAPABILITY_METADATA shaped
for the seed promoter, and (c) return >= 2 seed rows with the required
fields from seed_fallback().
"""
from __future__ import annotations

import importlib

import pytest

from components.procurement.parsers import (
    serve_the_home, techpowerup, newegg_us, amazon_uk,
)

MODULES = [serve_the_home, techpowerup, newegg_us, amazon_uk]


@pytest.mark.parametrize("mod", MODULES)
def test_seed_fallback_returns_rows(mod):
    rows = mod.seed_fallback()
    assert len(rows) >= 2
    for r in rows:
        assert r.get("name")
        assert r.get("cpu")
        # Every row must carry a PassMark so the headroom gate can run.
        assert r.get("cpu_passmark")


@pytest.mark.parametrize("mod", MODULES)
def test_parse_is_stub(mod):
    with pytest.raises(NotImplementedError):
        mod.parse("<html></html>")


@pytest.mark.parametrize("mod", MODULES)
def test_seed_metadata_shape(mod):
    md = mod.SEED_CAPABILITY_METADATA
    # Fields the seed_capability_promoter / materialiser recognise.
    assert md["runtime_mode"] == "stub"
    assert md["provenance"] == "fresh_blood_seed"
    assert md["capability_type"] == "html_parser"
    assert "parse" in md["methods"]
    assert md["seed_hash"]


def test_newegg_prices_in_usd_amazon_in_gbp():
    for r in newegg_us.seed_fallback():
        assert r["currency_orig"] == "USD"
        assert r.get("price_orig") is not None
    for r in amazon_uk.seed_fallback():
        assert r["currency_orig"] == "GBP"
        assert r.get("price_gbp") is not None
