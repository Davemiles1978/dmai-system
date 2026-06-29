"""Layer 4: CapabilitySpecGenerator (chunk L4-2).

Takes a CapabilityGapEntry, returns a ModuleSpec describing what
CapabilityImplementer needs to produce. Always returns a structurally valid
ModuleSpec — never raises. The LLM path is a stub here (chunk L4-7) and
defaults to None, so behaviour in L4-2 is deterministic template-driven.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from components.capability_gap_entry import CapabilityGapEntry


@dataclass
class ModuleSpec:
    target_path: str
    public_api: List[str]
    test_cases: List[Dict[str, Any]]
    dependencies: List[str]
    smoke_test_cmd: str
    description: str = ""
    source_gap: str = ""


def _to_classname(slug: str) -> str:
    """skill_acquisition_engine -> SkillAcquisitionEngine"""
    parts = re.split(r"[_\-]+", slug.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "Capability"


def _generic_scaffold(gap: CapabilityGapEntry) -> ModuleSpec:
    cls = _to_classname(gap.name)
    return ModuleSpec(
        target_path=f"components/{gap.name}.py",
        public_api=[
            f"class {cls}:",
            f"    def __init__(self) -> None: ...",
            f"    def run(self) -> dict: ...",
        ],
        test_cases=[{"input": {}, "expected_output": {"status": "ok"}}],
        dependencies=[],
        smoke_test_cmd=(
            f"python3 -c 'from components.{gap.name} import {cls}; "
            f"assert {cls}().run().get(\"status\") == \"ok\"'"
        ),
        description=gap.description or f"Auto-generated baseline for {gap.name}",
        source_gap=gap.name,
    )


class CapabilitySpecGenerator:
    """Deterministic spec generator with LLM stub (LLM path lands in L4-7)."""

    # The fallback templates use the same generic scaffold for now; future
    # chunks can specialise per-capability (e.g. transfer_learning_adapter
    # gets a specific public_api shape).
    FALLBACK_TEMPLATES: Dict[str, Callable[[CapabilityGapEntry], ModuleSpec]] = {}

    def __init__(self, model_surface: Optional[Callable[[str], str]] = None) -> None:
        """model_surface: optional LLM callable str -> str. None => template path."""
        self._model = model_surface

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, gap: CapabilityGapEntry) -> ModuleSpec:
        """Always returns a ModuleSpec. Never raises.

        - If an LLM surface is configured, try it; on any failure, fall
          back to the deterministic template.
        - LLM path is stubbed here and shipped in chunk L4-7.
        """
        if self._model is not None:
            try:
                spec = self._llm_generate(gap)
                if spec is not None:
                    return spec
            except Exception:
                pass
        return self._template_generate(gap)

    # ------------------------------------------------------------------
    # LLM path (stub for L4-2; full implementation in L4-7)
    # ------------------------------------------------------------------
    def _llm_generate(self, gap: CapabilityGapEntry) -> Optional[ModuleSpec]:
        prompt = self._build_prompt(gap)
        raw = self._model(prompt) if self._model else ""
        if not raw:
            return None
        return self._parse_llm_response(raw)

    def _build_prompt(self, gap: CapabilityGapEntry) -> str:
        return (
            "You are a senior Python engineer working inside DMAI. "
            "Produce a ModuleSpec JSON object (no prose) for the capability "
            f"named '{gap.name}'. Description: {gap.description!r}. "
            f"Target KPI: {gap.target_kpi!r}. "
            "Required fields: target_path, public_api (list of signature strings), "
            "test_cases (list of {input, expected_output}), dependencies (list of "
            "modules), smoke_test_cmd (single shell command), description, source_gap. "
            "Wrap the JSON in a ```json fenced block."
        )

    def _parse_llm_response(self, raw: str) -> ModuleSpec:
        """Extract JSON block from LLM output. Raises on malformed input
        so generate() can fall back to the template path."""
        import json
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        payload = m.group(1) if m else raw
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("LLM response was not a JSON object")
        # Keep only known fields; the dataclass __init__ will catch missing required ones.
        known = {k: data[k] for k in ModuleSpec.__dataclass_fields__ if k in data}
        return ModuleSpec(**known)

    # ------------------------------------------------------------------
    # Template fallback path
    # ------------------------------------------------------------------
    def _template_generate(self, gap: CapabilityGapEntry) -> ModuleSpec:
        builder = self.FALLBACK_TEMPLATES.get(gap.name)
        if builder is not None:
            try:
                spec = builder(gap)
                if isinstance(spec, ModuleSpec):
                    return spec
            except Exception:
                pass
        return _generic_scaffold(gap)


__all__ = ["ModuleSpec", "CapabilitySpecGenerator", "_to_classname"]
