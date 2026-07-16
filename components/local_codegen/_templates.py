"""Template synthesis for well-understood capability shapes.

Design principle: every template must produce a module whose `run()`
succeeds for every valid `happy_kwargs` the materialiser might send
(see `_happy_kwargs_for` in capability_materialiser.py).

Never hallucinate: if we don't recognise the capability_type we
return `can_template=False` and let the external LLM cascade take
over.
"""
from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional


LOCAL_CAPABILITY_TYPES = frozenset({
    "utility",
    "configuration",
    "data_structure",
    "trading",
    "blockchain",
    "interface",
    "research",
    "integration",
    "composite",
    "frontier",
    "diversity_nudge",
    "ai_provider_update",
    "concept",
})


@dataclass
class TemplateResult:
    ok: bool
    source: str
    reason: str = ""
    template_id: str = ""


def can_template(capability_type: str) -> bool:
    return str(capability_type or "").lower() in LOCAL_CAPABILITY_TYPES


# ── Slug helpers ──────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slugify(text: str, max_len: int = 40) -> str:
    s = _SLUG_RE.sub("_", (text or "").strip().lower()).strip("_")
    if not s:
        s = "capability"
    return s[:max_len]


def _module_docstring(concept: str, insight: str, cap_type: str) -> str:
    concept_line = (concept or "").strip().replace("\n", " ")[:200]
    insight_line = (insight or "").strip().replace("\n", " ")[:300]
    return textwrap.dedent(f'''\
        """
        DMAI locally-authored capability (template synthesis).

        Type    : {cap_type}
        Concept : {concept_line}
        Insight : {insight_line}

        This module was written by DMAI's local code-authoring path
        (components/local_codegen), not by an external LLM. It is
        deliberately conservative: pure functions, no I/O, safe to
        smoke-test with any happy_kwargs.
        """
        ''')


# ── Per-shape body generators ────────────────────────────────────────────
# Each returns just the body of run(**kwargs); the wrapper below adds
# imports, module docstring and def run().

def _body_utility(concept: str) -> str:
    return textwrap.dedent('''\
        values = kwargs.get("values") or []
        try:
            nums = [float(v) for v in values if isinstance(v, (int, float))]
        except (TypeError, ValueError):
            nums = []
        n = len(nums)
        total = sum(nums)
        mean = total / n if n else 0.0
        return {
            "ok":     True,
            "count":  n,
            "sum":    total,
            "mean":   mean,
            "min":    min(nums) if nums else None,
            "max":    max(nums) if nums else None,
            "source": "local_template.utility",
        }
        ''')


def _body_configuration() -> str:
    return textwrap.dedent('''\
        config = kwargs.get("config") or {}
        if not isinstance(config, dict):
            return {"ok": False, "reason": "config must be a dict",
                    "source": "local_template.configuration"}
        normalised = {str(k): v for k, v in config.items()}
        return {
            "ok":         True,
            "size":       len(normalised),
            "keys":       sorted(normalised.keys()),
            "normalised": normalised,
            "source":     "local_template.configuration",
        }
        ''')


def _body_data_structure() -> str:
    return textwrap.dedent('''\
        items = kwargs.get("items") or []
        if not isinstance(items, list):
            return {"ok": False, "reason": "items must be a list",
                    "source": "local_template.data_structure"}
        counts = {}
        for it in items:
            k = type(it).__name__
            counts[k] = counts.get(k, 0) + 1
        return {
            "ok":            True,
            "length":        len(items),
            "type_counts":   counts,
            "first":         items[0] if items else None,
            "last":          items[-1] if items else None,
            "source":        "local_template.data_structure",
        }
        ''')


def _body_trading() -> str:
    return textwrap.dedent('''\
        prices = kwargs.get("prices") or []
        try:
            series = [float(p) for p in prices]
        except (TypeError, ValueError):
            return {"ok": False, "reason": "prices must be numeric",
                    "source": "local_template.trading"}
        if not series:
            return {"ok": True, "count": 0, "return_pct": 0.0,
                    "source": "local_template.trading"}
        first, last = series[0], series[-1]
        ret_pct = ((last - first) / first * 100.0) if first else 0.0
        if len(series) > 1:
            m = sum(series) / len(series)
            var = sum((x - m) ** 2 for x in series) / len(series)
            vol = var ** 0.5
        else:
            vol = 0.0
        return {
            "ok":         True,
            "count":      len(series),
            "first":      first,
            "last":       last,
            "return_pct": ret_pct,
            "volatility": vol,
            "source":     "local_template.trading",
        }
        ''')


def _body_blockchain() -> str:
    return textwrap.dedent('''\
        payload = kwargs.get("payload") or {}
        try:
            blob = repr(sorted(payload.items()) if isinstance(payload, dict)
                        else payload).encode("utf-8", "replace")
        except Exception:
            blob = repr(payload).encode("utf-8", "replace")
        digest = hashlib.sha256(blob).hexdigest()
        return {
            "ok":       True,
            "digest":   digest,
            "size":     len(blob),
            "kind":     type(payload).__name__,
            "source":   "local_template.blockchain",
        }
        ''')


def _body_interface() -> str:
    return textwrap.dedent('''\
        request = kwargs.get("request") or {}
        if not isinstance(request, dict):
            return {"ok": False, "reason": "request must be a dict",
                    "source": "local_template.interface"}
        method = str(request.get("method", "GET")).upper()
        path   = str(request.get("path", "/"))
        return {
            "ok":       True,
            "method":   method,
            "path":     path,
            "echo":     request,
            "source":   "local_template.interface",
        }
        ''')


def _body_research() -> str:
    return textwrap.dedent('''\
        query = str(kwargs.get("query", "")).strip()
        tokens = [t for t in query.split() if t]
        return {
            "ok":            True,
            "query":         query,
            "token_count":   len(tokens),
            "tokens":        tokens[:32],
            "empty":         not tokens,
            "source":        "local_template.research",
        }
        ''')


def _body_integration() -> str:
    return textwrap.dedent('''\
        payload = kwargs.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return {
            "ok":       True,
            "size":     len(payload),
            "keys":     sorted(str(k) for k in payload.keys()),
            "passthrough": payload,
            "source":   "local_template.integration",
        }
        ''')


def _body_composite() -> str:
    return textwrap.dedent('''\
        a = kwargs.get("a") or {}
        b = kwargs.get("b") or {}
        if not isinstance(a, dict): a = {"value": a}
        if not isinstance(b, dict): b = {"value": b}
        merged = {**a, **b}
        return {
            "ok":       True,
            "a_size":   len(a),
            "b_size":   len(b),
            "merged_size": len(merged),
            "merged":   merged,
            "source":   "local_template.composite",
        }
        ''')


def _body_frontier() -> str:
    return textwrap.dedent('''\
        seed = int(kwargs.get("seed", 0) or 0)
        rng = random.Random(seed)
        return {
            "ok":       True,
            "seed":     seed,
            "value":    rng.random(),
            "picks":    [rng.randint(0, 99) for _ in range(5)],
            "source":   "local_template.frontier",
        }
        ''')


def _body_diversity_nudge() -> str:
    return textwrap.dedent('''\
        seed = int(kwargs.get("seed", 0) or 0)
        rng = random.Random(seed ^ 0x5EED)
        return {
            "ok":         True,
            "seed":       seed,
            "nudge":      rng.gauss(0.0, 1.0),
            "direction":  "up" if rng.random() > 0.5 else "down",
            "source":     "local_template.diversity_nudge",
        }
        ''')


def _body_ai_provider_update() -> str:
    return textwrap.dedent('''\
        release = kwargs.get("release") or {}
        if not isinstance(release, dict):
            release = {"tag": str(release)}
        tag = str(release.get("tag", "unknown"))
        return {
            "ok":         True,
            "tag":        tag,
            "has_notes":  bool(release.get("notes")),
            "release":    release,
            "source":     "local_template.ai_provider_update",
        }
        ''')


def _body_concept() -> str:
    return textwrap.dedent('''\
        return {
            "ok":       True,
            "input":    kwargs.get("input"),
            "kind":     type(kwargs.get("input")).__name__,
            "source":   "local_template.concept",
        }
        ''')


_BODIES = {
    "utility":            _body_utility,
    "configuration":      _body_configuration,
    "data_structure":     _body_data_structure,
    "trading":            _body_trading,
    "blockchain":         _body_blockchain,
    "interface":          _body_interface,
    "research":           _body_research,
    "integration":        _body_integration,
    "composite":          _body_composite,
    "frontier":           _body_frontier,
    "diversity_nudge":    _body_diversity_nudge,
    "ai_provider_update": _body_ai_provider_update,
    "concept":            _body_concept,
}


# Extra imports each body needs (keep minimal).
_IMPORTS = {
    "blockchain":       "import hashlib",
    "frontier":         "import random",
    "diversity_nudge":  "import random",
}


# ── Wrapper + validator ──────────────────────────────────────────────────

def _wrap_module(cap_type: str, concept: str, insight: str, body: str) -> str:
    doc = _module_docstring(concept, insight, cap_type)
    imports = _IMPORTS.get(cap_type, "")
    import_block = imports + "\n" if imports else ""
    # Indent body inside run().
    indented = textwrap.indent(body.rstrip(), "    ")
    return (
        doc
        + import_block
        + "\n"
        + "def run(**kwargs):\n"
        + '    """Locally-authored capability entrypoint."""\n'
        + indented
        + "\n"
    )


def generate_from_template(*,
                           concept: str,
                           insight: str,
                           capability_type: str,
                           happy_kwargs: Optional[Dict[str, Any]] = None,
                           ) -> TemplateResult:
    """Synthesise a capability module locally.

    Returns TemplateResult with ok=True on success. On unrecognised
    capability_type or a validator failure, returns ok=False so the
    caller can fall through to the LLM cascade.
    """
    ct = str(capability_type or "").lower()
    if ct not in LOCAL_CAPABILITY_TYPES:
        return TemplateResult(
            ok=False, source="",
            reason=f"no_template_for_capability_type:{ct}",
            template_id="",
        )
    body_fn = _BODIES[ct]
    body = body_fn(concept) if body_fn is _body_utility else body_fn()
    source = _wrap_module(ct, concept, insight, body)

    # Local validator: syntax must parse and `run` must be defined.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return TemplateResult(
            ok=False, source=source,
            reason=f"template_syntax_error:{e}",
            template_id=ct,
        )
    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run"
        for n in ast.walk(tree)
    )
    if not has_run:
        return TemplateResult(
            ok=False, source=source,
            reason="template_missing_run_function",
            template_id=ct,
        )
    return TemplateResult(
        ok=True, source=source, reason="",
        template_id=f"local_template.{ct}",
    )
