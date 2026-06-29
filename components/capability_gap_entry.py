"""Layer 4: CapabilityGapEntry dataclass.

Shared shape for the Layer 4 self-generation pipeline. Produced by
SelfScanner.audit_capability_gaps_typed() and consumed by the spec
generator, implementer, self-tester, and autonomy-score tracker.

Importing this module must have zero side effects (no DB opens, no I/O).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CapabilityGapEntry:
    name: str                          # machine-slug, e.g. "skill_acquisition_engine"
    description: str                   # human-readable intent
    priority: int                      # 1 (highest) – 5 (lowest)
    evidence_source: str               # "kpi:<key>" or "registry:missing"
    target_kpi: str                    # the KPI this capability is expected to move
    current_value: float = 0.0         # latest observed KPI value
    target_value: float = 0.5          # goal (all 6 underperforming KPIs share this)
    retry_count: int = 0               # generation attempts so far
    extra: Dict[str, Any] = field(default_factory=dict)
