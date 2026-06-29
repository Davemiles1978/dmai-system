"""Utilities to fetch /api/self-evolution/gaps in-process.

Layer 3 intentionally avoids HTTP loopback calls. We import the gap builder
directly (where available), and return a normalized list of gap entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class GapEntry:
    category: str
    payload: Dict[str, Any]


def _normalize_gaps(gaps: Dict[str, Any]) -> List[GapEntry]:
    out: List[GapEntry] = []
    if not isinstance(gaps, dict):
        return out

    for category in ("broken_routes", "dead_threads", "empty_tables"):
        entries = gaps.get(category) or []
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict):
                out.append(GapEntry(category=category, payload=item))
            else:
                out.append(GapEntry(category=category, payload={"value": item}))
    return out


def fetch_gaps(fresh: bool = True) -> Tuple[List[GapEntry], Dict[str, Any]]:
    """Return normalized gap entries plus the raw response.

    Tries (in order):
      1) Import a callable builder from components/self_evolution_gaps.py
      2) Import the Flask route function (if exposed) and call it in a safe way

    Returns: (entries, raw)
    """

    raw: Dict[str, Any] = {}

    # Preferred: a pure function that builds the gaps dict.
    try:
        from components.self_evolution_gaps import build_gaps  # type: ignore

        raw = build_gaps(fresh=fresh)  # type: ignore
        return _normalize_gaps(raw), raw
    except Exception:
        pass

    # Fallback: if the route is implemented as a function returning dict.
    try:
        from components.self_evolution_gaps import get_self_evolution_gaps  # type: ignore

        raw = get_self_evolution_gaps(fresh=fresh)  # type: ignore
        return _normalize_gaps(raw), raw
    except Exception:
        pass

    return [], raw


def iter_gaps(fresh: bool = True) -> Iterable[GapEntry]:
    entries, _raw = fetch_gaps(fresh=fresh)
    for entry in entries:
        yield entry


def iter_gap_entries(fresh: bool = True) -> Iterable[Dict[str, Any]]:
    """Yield gap entries as plain dicts shaped for RepairPattern detectors.

    Chunk 10: this is the contract the orchestrator already imports. Each
    yielded dict carries the original payload plus an injected ``kind`` field
    (the category) so detectors can branch on category strings cleanly.

    String-style payloads (e.g. ``empty_tables`` returns bare table names
    like ``"capabilities"``) are coerced into ``{"table": <name>}`` so the
    table-shaped detectors hit.
    """

    for entry in iter_gaps(fresh=fresh):
        payload = dict(entry.payload or {})
        payload.setdefault("kind", entry.category)
        # Coerce common string-shaped payloads into table-keyed dicts so
        # downstream detectors (which look at ``table``) can match.
        if entry.category == "empty_tables" and "value" in payload and "table" not in payload:
            payload["table"] = payload["value"]
        yield payload
