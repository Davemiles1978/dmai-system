"""ServeTheHome + STH forum parser stub (PR K).

STUB: the LLM-driven capability materialiser (PR H) will generate the
body of :func:`parse`. Until then :func:`seed_fallback` supplies a handful
of known-good boxes so the procurement pipeline produces a shortlist from
day 1.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Metadata the seed_capability_promoter / capability_materialiser recognise.
# runtime_mode='stub' + provenance='fresh_blood_seed' marks this as a
# judge-acceptable seed; self_judge acceptance promotes provenance to
# 'fresh_blood_seed+self_judge', which is the materialiser's pick filter
# (components/capability_materialiser.py:_pick_candidates).
SEED_CAPABILITY_METADATA: Dict[str, Any] = {
    "name":            "procurement_parser_serve_the_home",
    "type":            "concept",
    "capability_type": "html_parser",
    "description": (
        "Parse a ServeTheHome review / STH forum thread HTML page into a "
        "list of hardware dicts. Extract, per candidate box: name (str), "
        "cpu (str), cpu_passmark (int, look up if absent), tdp_w (float), "
        "idle_w (float, use measured idle from the review if published), "
        "ram_gb (int), storage_gb (int), price_gbp (float), url (str). "
        "Return list[dict]; skip rows missing name or cpu."
    ),
    "source_url":    "https://www.servethehome.com/",
    "runtime_mode":  "stub",
    "language":      "python",
    "methods":       ["parse"],
    "is_async":      False,
    "args":          ["html"],
    "provenance":    "fresh_blood_seed",
    "seed_hash":     "procurement:serve_the_home:v1",
}


def parse(html: str) -> List[Dict[str, Any]]:
    """Parse a ServeTheHome page into hardware rows.

    :param html: raw HTML of a STH review or forum thread.
    :returns: list of dicts with keys name, cpu, cpu_passmark, tdp_w,
        idle_w, ram_gb, storage_gb, price_gbp, url.

    Not yet materialised — the PR H capability materialiser generates this
    body. Until then the researcher uses :func:`seed_fallback`.
    """
    raise NotImplementedError(
        "serve_the_home.parse is a materialiser stub; see "
        "SEED_CAPABILITY_METADATA for the generation brief"
    )


def seed_fallback() -> List[Dict[str, Any]]:
    """Hand-written known-good STH-reviewed mini PCs (GBP)."""
    return [
        {
            "name": "Minisforum MS-01",
            "url": "https://www.servethehome.com/minisforum-ms-01-review/",
            "cpu": "Intel Core i9-13900H",
            "cpu_passmark": 28000,
            "tdp_w": 45.0,
            "idle_w": 15.0,
            "ram_gb": 32,
            "storage_gb": 1000,
            "price_gbp": 649.0,
            "currency_orig": "GBP",
            "price_orig": 649.0,
        },
        {
            "name": "Beelink SER8",
            "url": "https://www.servethehome.com/beelink-ser8-review/",
            "cpu": "AMD Ryzen 7 8845HS",
            "cpu_passmark": 28500,
            "tdp_w": 54.0,
            "idle_w": 11.0,
            "ram_gb": 32,
            "storage_gb": 1000,
            "price_gbp": 599.0,
            "currency_orig": "GBP",
            "price_orig": 599.0,
        },
        {
            "name": "ASRock DeskMini X600",
            "url": "https://www.servethehome.com/asrock-deskmini-x600-review/",
            "cpu": "AMD Ryzen 5 8600G",
            "cpu_passmark": 24000,
            "tdp_w": 65.0,
            "idle_w": 12.0,
            "ram_gb": 32,
            "storage_gb": 1000,
            "price_gbp": 449.0,
            "currency_orig": "GBP",
            "price_orig": 449.0,
        },
    ]


__all__ = ["SEED_CAPABILITY_METADATA", "parse", "seed_fallback"]
