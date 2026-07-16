"""Exercise generator for coding curriculum topics.

For a given topic slug, produces a concrete coding exercise DMAI can
attempt to solve. Each exercise has:

    - a stable exercise_id (deterministic per topic + variant)
    - a natural-language brief
    - a target module skeleton (function signature + docstring)
    - one or more grading test cases: (kwargs_dict, expected_predicate)

The grader (in ``_grader.py``) runs the candidate code in a subprocess
with the given kwargs and checks the predicate.

This module is intentionally pure (no I/O, no network, no LLM). Exercise
templates are hand-written per capability shape so grading is
deterministic and we never store zero/None rows.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._taxonomy import CURRICULUM_TOPICS


# ── Exercise data model ──────────────────────────────────────────────────

@dataclass
class GradingCase:
    """A single grading test case for an exercise.

    ``kwargs`` are passed to the generated ``run(**kwargs)``. The
    grader compares the returned value against ``expected``:
        - if ``predicate`` is set, it's called as
          ``predicate(result_value)`` and must return True
        - otherwise ``expected`` is compared for equality
    """
    kwargs:     Dict[str, Any]
    expected:   Any = None
    predicate:  Optional[str] = None   # eval'd expression on 'result'
    description: str = ""


@dataclass
class Exercise:
    exercise_id: str
    topic_slug:  str
    brief:       str
    signature:   str      # e.g. "def run(**kwargs) -> dict:"
    docstring:   str
    hint:        str      # short hint the candidate solver sees
    grading:     List[GradingCase]
    capability_shape: str  # maps to local_codegen template shape

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exercise_id":      self.exercise_id,
            "topic_slug":       self.topic_slug,
            "brief":            self.brief,
            "signature":        self.signature,
            "docstring":        self.docstring,
            "hint":             self.hint,
            "capability_shape": self.capability_shape,
            "grading":          [
                {
                    "kwargs":      g.kwargs,
                    "expected":    g.expected,
                    "predicate":   g.predicate,
                    "description": g.description,
                }
                for g in self.grading
            ],
        }


# ── Exercise families ────────────────────────────────────────────────────
#
# Each family targets one topic characteristic. We route topics to
# families by inspecting the topic dict (language, tier, depth,
# keywords). We prefer generic-but-verifiable exercises over ones
# that require deep language knowledge — the goal is to build muscle
# memory, not trick the solver.

def _exercise_id(slug: str, variant: str) -> str:
    return hashlib.sha256(
        f"{slug}::{variant}".encode("utf-8"),
    ).hexdigest()[:16]


def _family_list_transform(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "list_transform_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(values=[...]) that "
            f"returns {{'ok': True, 'result': sum(values), "
            f"'count': len(values)}} when values is a list of numbers."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: sum a numeric list.\n"
            f"Topic: {topic['title']} ({slug}).\n"
            f"Return {{'ok': True, 'result': <sum>, 'count': <len>}}."
        ),
        hint=(
            "Read kwargs.get('values', []), coerce to list, sum floats, "
            "return dict with ok/result/count."
        ),
        capability_shape="data_structure",
        grading=[
            GradingCase(
                kwargs={"values": [1, 2, 3, 4]},
                predicate=(
                    "isinstance(result, dict) and result.get('ok') is True "
                    "and abs(float(result.get('result', 0)) - 10.0) < 1e-6 "
                    "and int(result.get('count', 0)) == 4"
                ),
                description="basic 4-element sum",
            ),
            GradingCase(
                kwargs={"values": []},
                predicate=(
                    "isinstance(result, dict) and result.get('ok') is True "
                    "and float(result.get('result', 0)) == 0.0 "
                    "and int(result.get('count', -1)) == 0"
                ),
                description="empty list is not a hang",
            ),
        ],
    )


def _family_utility_fn(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "utility_fn_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(values=[...]) that "
            f"returns {{'ok': True, 'result': [v*2 for v in values]}}."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: double every element.\n"
            f"Topic: {topic['title']} ({slug})."
        ),
        hint="Iterate values, multiply by 2, return dict with ok+result list.",
        capability_shape="utility",
        grading=[
            GradingCase(
                kwargs={"values": [1, 2, 3]},
                predicate=(
                    "result.get('ok') is True and "
                    "list(result.get('result', [])) == [2, 4, 6]"
                ),
                description="doubling 3 items",
            ),
            GradingCase(
                kwargs={"values": [0]},
                predicate=(
                    "result.get('ok') is True and "
                    "list(result.get('result', [])) == [0]"
                ),
                description="zero survives",
            ),
        ],
    )


def _family_config_merge(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "config_merge_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(base={{}}, override={{}}) "
            f"returning {{'ok': True, 'result': merged}} where override "
            f"wins for overlapping keys."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: shallow dict merge, override wins.\n"
            f"Topic: {topic['title']} ({slug})."
        ),
        hint="Copy base, update with override, wrap in dict.",
        capability_shape="configuration",
        grading=[
            GradingCase(
                kwargs={"base": {"a": 1, "b": 2}, "override": {"b": 20}},
                predicate=(
                    "result.get('ok') is True and "
                    "dict(result.get('result', {})) == {'a': 1, 'b': 20}"
                ),
                description="override wins",
            ),
            GradingCase(
                kwargs={"base": {}, "override": {"x": 1}},
                predicate=(
                    "result.get('ok') is True and "
                    "dict(result.get('result', {})) == {'x': 1}"
                ),
                description="empty base",
            ),
        ],
    )


def _family_price_series(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "price_series_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(prices=[...]) that returns "
            f"{{'ok': True, 'mean': avg_price, 'n': len(prices)}}."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: mean of a price series (safe on empty).\n"
            f"Topic: {topic['title']} ({slug})."
        ),
        hint="Guard empty list -> mean=0.0. Coerce to float.",
        capability_shape="trading",
        grading=[
            GradingCase(
                kwargs={"prices": [10.0, 20.0, 30.0]},
                predicate=(
                    "result.get('ok') is True and "
                    "abs(float(result.get('mean', 0)) - 20.0) < 1e-6 and "
                    "int(result.get('n', 0)) == 3"
                ),
                description="mean of three prices",
            ),
            GradingCase(
                kwargs={"prices": []},
                predicate=(
                    "result.get('ok') is True and "
                    "float(result.get('mean', -1)) == 0.0"
                ),
                description="empty series does not hang or divide by zero",
            ),
        ],
    )


def _family_query_echo(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "query_echo_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(query='hello') returning "
            f"{{'ok': True, 'query': query, 'length': len(query)}}."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: research-echo bundler.\n"
            f"Topic: {topic['title']} ({slug})."
        ),
        hint="Read query string, return dict with ok/query/length.",
        capability_shape="research",
        grading=[
            GradingCase(
                kwargs={"query": "abc"},
                predicate=(
                    "result.get('ok') is True and "
                    "result.get('query') == 'abc' and "
                    "int(result.get('length', 0)) == 3"
                ),
                description="basic echo",
            ),
            GradingCase(
                kwargs={"query": ""},
                predicate=(
                    "result.get('ok') is True and "
                    "int(result.get('length', -1)) == 0"
                ),
                description="empty query is not a hang",
            ),
        ],
    )


def _family_composite(topic: dict) -> Exercise:
    slug = topic["slug"]
    return Exercise(
        exercise_id=_exercise_id(slug, "composite_v1"),
        topic_slug=slug,
        brief=(
            f"Topic: {topic['title']}. Write run(a={{}}, b={{}}) that returns "
            f"{{'ok': True, 'merged': {{**a, **b}}, 'size': ...}}."
        ),
        signature="def run(**kwargs) -> dict:",
        docstring=(
            f"Exercise: composite merge of two dicts.\n"
            f"Topic: {topic['title']} ({slug})."
        ),
        hint="Merge a and b (b wins), also report resulting size.",
        capability_shape="composite",
        grading=[
            GradingCase(
                kwargs={"a": {"x": 1}, "b": {"y": 2}},
                predicate=(
                    "result.get('ok') is True and "
                    "dict(result.get('merged', {})) == {'x': 1, 'y': 2} and "
                    "int(result.get('size', 0)) == 2"
                ),
                description="two non-overlapping dicts",
            ),
        ],
    )


# ── Routing: topic -> exercise ───────────────────────────────────────────

# Keyword-based family routing. First match wins; falls back to the
# safest generic family (list transform).
_ROUTES: List[Tuple[Tuple[str, ...], Callable[[dict], Exercise]]] = [
    (("trading", "market", "backtest", "kelly", "portfolio", "risk"),
     _family_price_series),
    (("dict", "config", "settings", "merge", "environment"),
     _family_config_merge),
    (("research", "search", "query", "knowledge", "insight", "ingest"),
     _family_query_echo),
    (("compose", "composite", "pipeline", "orchestrat", "aggregate"),
     _family_composite),
    (("list", "array", "iter", "sequence", "map", "filter", "reduce",
      "comprehension", "generator"),
     _family_utility_fn),
]


def _route_topic(topic: dict) -> Callable[[dict], Exercise]:
    haystack = (
        topic["slug"] + " "
        + topic["title"].lower() + " "
        + " ".join(topic.get("keywords") or []).lower()
    )
    for keywords, fn in _ROUTES:
        if any(k in haystack for k in keywords):
            return fn
    return _family_list_transform


def exercise_for_topic(slug: str) -> Optional[Exercise]:
    """Build an exercise for a topic slug, or None if unknown."""
    topic = CURRICULUM_TOPICS.get(slug)
    if not topic:
        return None
    return _route_topic(topic)(topic)


def all_supported_shapes() -> List[str]:
    """Return the distinct capability_shape strings this module emits."""
    return sorted({
        "data_structure", "utility", "configuration",
        "trading", "research", "composite",
    })
