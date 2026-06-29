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


def fetch_gaps(fresh: bool = True, app=None, data_path: str = "data") -> Tuple[List[GapEntry], Dict[str, Any]]:
    """Return normalized gap entries plus the raw response.

    Tries (in order):
      1) SelfScanner(app, data_path).run() — the actual builder used by
         /api/self-evolution/gaps in dmai_core_complete.py
      2) Cached gap_report.json on disk

    Returns: (entries, raw)
    """
    import os
    import json
    import logging
    log = logging.getLogger(__name__)

    raw: Dict[str, Any] = {}

    if fresh:
        try:
            from components.self_scanner import SelfScanner
            raw = SelfScanner(app=app, data_path=data_path).run()
            return _normalize_gaps(raw), raw
        except Exception as e:
            log.warning("gap_fetcher: fresh SelfScanner failed: %s", e)

    # Fallback to cached report
    try:
        p = os.path.join(str(data_path).rstrip("/"), "gap_report.json")
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            return _normalize_gaps(raw), raw
    except Exception as e:
        log.warning("gap_fetcher: cached gap_report.json read failed: %s", e)

    # Last-resort fresh attempt even if fresh=False
    if not fresh:
        try:
            from components.self_scanner import SelfScanner
            raw = SelfScanner(app=app, data_path=data_path).run()
            return _normalize_gaps(raw), raw
        except Exception as e:
            log.warning("gap_fetcher: last-resort SelfScanner failed: %s", e)

    return [], raw


def iter_gaps(fresh: bool = True) -> Iterable[GapEntry]:
    entries, _raw = fetch_gaps(fresh=fresh)
    for entry in entries:
        yield entry
