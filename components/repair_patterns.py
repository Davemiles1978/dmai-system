"""Layer 3 repair pattern registry.

This module is intentionally standalone and safe to import during startup.
It defines a small library of "repair patterns" that can detect specific gap
entries (from /api/self-evolution/gaps) and propose small code edits.

Chunk 1 only scaffolds the library + detection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class FixProposal:
    """A concrete, human-readable patch proposal derived from a single gap entry."""

    file: str
    original_snippet: str
    new_snippet: str
    description: str
    confidence: float = 0.5
    meta: Optional[Dict[str, Any]] = None

from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class RepairPattern:
    """A known repair pattern.

    detect: returns True if this pattern is applicable to the provided gap entry.
    propose: in later chunks, should return a FixProposal for the change.
    """

    name: str
    detect: Callable[[Dict[str, Any]], bool]
    propose: Callable[[Dict[str, Any], str], Optional[Any]]  # FixProposal later


def _gap_text(gap_entry: Dict[str, Any]) -> str:
    """Best-effort text representation of a gap entry for detection."""

    if gap_entry is None:
        return ""
    parts: List[str] = []
    for k in ("route", "thread", "table", "error", "detail", "message", "reason"):
        v = gap_entry.get(k)
        if v:
            parts.append(str(v))
    return " | ".join(parts).lower()


def _detect_startup_errors_swallowed(gap_entry: Dict[str, Any]) -> bool:
    t = _gap_text(gap_entry)
    return (
        "broken_routes" in str(gap_entry.get("kind", "")).lower()
        and ("503" in t or "service unavailable" in t)
        and ("auth" in t or "master" in t)
    )


def _detect_safe_open_kdb_check_same_thread_kwarg(gap_entry: Dict[str, Any]) -> bool:
    t = _gap_text(gap_entry)
    return "check_same_thread" in t and ("unexpected" in t or "typeerror" in t)


def _detect_dead_thread_false_positive(gap_entry: Dict[str, Any]) -> bool:
    t = _gap_text(gap_entry)
    return (
        "dead_threads" in str(gap_entry.get("kind", "")).lower()
        and ("false" in t or "alive" in t or "still running" in t)
    )


def _detect_bytes_affinity_keyerror(gap_entry: Dict[str, Any]) -> bool:
    t = _gap_text(gap_entry)
    return "keyerror" in t and ("bytes" in t or "affinity" in t or "text" in t)


def _detect_bytes_json_serialization_typeerror(gap_entry: Dict[str, Any]) -> bool:
    t = _gap_text(gap_entry)
    return "typeerror" in t and "bytes" in t and ("json" in t or "serialize" in t)


def _noop_propose(_: Dict[str, Any], __: str) -> Optional[Any]:
    return None


def _detect_empty_sqlite_tables(gap_entry: Dict[str, Any]) -> bool:
    """Detect a gap indicating core SQLite tables exist but are empty.

    Intended for production live-state gaps like empty `capabilities`, `insights`,
    `suggestions` tables.
    """

    t = _gap_text(gap_entry)
    # Match either explicit table key or textual mention.
    table = str(gap_entry.get("table", "")).lower()
    if table in {"capabilities", "insights", "suggestions"}:
        return True
    return (
        ("capabilities" in t or "insights" in t or "suggestions" in t)
        and ("empty" in t or "0 rows" in t or "no rows" in t)
    )


PATTERNS: List[RepairPattern] = [
    RepairPattern(
        name="startup_errors_swallowed",
        detect=_detect_startup_errors_swallowed,
        propose=_noop_propose,
    ),
    RepairPattern(
        name="safe_open_kdb_check_same_thread_kwarg",
        detect=_detect_safe_open_kdb_check_same_thread_kwarg,
        propose=_noop_propose,
    ),
    RepairPattern(
        name="dead_thread_false_positive",
        detect=_detect_dead_thread_false_positive,
        propose=_noop_propose,
    ),
    RepairPattern(
        name="bytes_affinity_keyerror",
        detect=_detect_bytes_affinity_keyerror,
        propose=_noop_propose,
    ),
    RepairPattern(
        name="bytes_json_serialization_typeerror",
        detect=_detect_bytes_json_serialization_typeerror,
        propose=_noop_propose,
    ),
    RepairPattern(
        name="empty_sqlite_tables",
        detect=_detect_empty_sqlite_tables,
        propose=_noop_propose,
    ),
]
