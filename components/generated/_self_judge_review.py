"""Post-generation review: run DMAI's own self_judge against the
docstring of the generated module.

The idea: a module that passed syntactic + runtime gates is not
enough. DMAI must also read what she just wrote and confirm it
still matches the concept she accepted. If the docstring drifts
(the LLM invented a different capability), the judge will reject
or defer and the materialiser scraps the module.

This function is intentionally *strict* on rejects and *tolerant*
on defers: a re-eval defer means "docstring is ambiguous, try
again" - not a hard failure. The materialiser will treat defers as
retries with a hint pointing at the concept mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from components.db import safe_open_kdb
from components.self_judge import judge_seed


@dataclass
class ReviewResult:
    ok: bool                 # True if the judge accepted the docstring
    verdict: str             # accept / reject / defer
    confidence: float
    reason: str
    gap_summary: str = ""


def review_generated_module(*,
                            concept: str,
                            channel: str,
                            docstring: str,
                            db_path: Optional[str] = None,
                            accept_threshold: float = 0.55,
                            ) -> ReviewResult:
    """Ask self_judge whether *docstring* still stands for *concept*.

    Uses a slightly *lower* accept threshold than the seed-time
    judge (default 0.55 vs 0.65) because the docstring is short and
    the vocabulary coverage signal is naturally lower - we still
    want a positive verdict for anything that looks aligned. Reject
    threshold is left at the module default (0.30).
    """
    # PR PP: mark gap-driven / self-scanner seeds so self_judge applies
    # the relaxed vocab floor. Any capability_type starting with 'gap_'
    # or channel already tagged gap_driven/self_scanner/backlog_seed
    # gets the relaxed floor.
    normalised_channel = str(channel or "").lower()
    if normalised_channel.startswith("gap_") or normalised_channel in (
        "self_scanner", "backlog_seed", "self_gen", "gap_driven",
    ):
        normalised_channel = "gap_driven"

    seed = {
        "channel":      normalised_channel,
        "concept":      concept,
        "insight_text": docstring or "",
    }
    # PR MM/PR PP: use safe_open_kdb so the reviewer participates in
    # the shared write lock instead of being a hidden lock competitor.
    conn = safe_open_kdb(db_path) if db_path else None
    try:
        verdict = judge_seed(seed, conn, accept_threshold=accept_threshold)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return ReviewResult(
        ok=(verdict.verdict == "accept"),
        verdict=verdict.verdict,
        confidence=float(verdict.confidence),
        reason=verdict.reason,
        gap_summary=verdict.knowledge_gap or "",
    )


__all__ = ["ReviewResult", "review_generated_module"]
