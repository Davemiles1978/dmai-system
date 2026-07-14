"""Infrastructure procurement research skill (PR K).

Shortlists candidate home-lab hardware for DMAI's eventual migration
off Render, priced on a 3-year total-cost-of-ownership basis using
DMAI's own workload footprint (PR J) and treasury balance (PR I).

Hybrid delivery:
* Deterministic orchestrator + TCO calc ship as regular Python
  (:mod:`components.procurement.researcher`).
* Vendor-specific HTML parsers ship as judge-accepted *stubs*
  (:mod:`components.procurement.parsers`) so the PR H capability
  materialiser can generate their bodies. Each stub carries a
  hand-written seed fallback so the pipeline works from day 1.
"""
from __future__ import annotations

__all__ = ["config", "schema", "store", "researcher", "loop", "parsers"]
