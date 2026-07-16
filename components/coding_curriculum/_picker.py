"""What should DMAI study next?

Given the mastery store + taxonomy, return the topic that:
  1. Has all its prerequisites at >= 0.5 mastery (or has no prereqs)
  2. Has the lowest mastery score itself
  3. Tie-break by tier (foundations first), then depth, then slug

If nothing is prerequisite-ready but there are unstudied topics with
no prereqs, we always have something to pick (tier 1 topics have no
prereqs). Function only returns None if the taxonomy is empty.
"""
from __future__ import annotations

from typing import Optional

from ._store import all_mastery
from ._taxonomy import CURRICULUM_TOPICS, all_topic_slugs


def _prereq_mastery_ok(slug: str, mastery: dict, threshold: float = 0.5) -> bool:
    topic = CURRICULUM_TOPICS.get(slug)
    if not topic:
        return False
    for p in topic["prerequisites"]:
        row = mastery.get(p)
        if not row or row["mastery_score"] < threshold:
            return False
    return True


def next_topic_to_study(*,
                        language: Optional[str] = None,
                        db_path: str = "data/dmai_knowledge.db",
                        ) -> Optional[dict]:
    """Pick the single most valuable topic to study right now."""
    if not CURRICULUM_TOPICS:
        return None
    mastery = all_mastery(db_path=db_path)

    candidates = []
    for slug in all_topic_slugs():
        t = CURRICULUM_TOPICS[slug]
        if language is not None and t["language"] != language:
            continue
        if not _prereq_mastery_ok(slug, mastery):
            continue
        row = mastery.get(slug)
        score = row["mastery_score"] if row else 0.0
        # Fully mastered topics don't need more study right now.
        if score >= 0.9:
            continue
        candidates.append((score, t["tier"], t["depth"], slug, t))

    if not candidates:
        # Everything is mastered or blocked. Return the lowest-mastery
        # blocked topic anyway - the system will make progress on
        # prereqs via the same picker over subsequent nights.
        for slug in all_topic_slugs():
            if language and CURRICULUM_TOPICS[slug]["language"] != language:
                continue
            row = mastery.get(slug)
            score = row["mastery_score"] if row else 0.0
            if score < 0.9:
                t = CURRICULUM_TOPICS[slug]
                candidates.append((score, t["tier"], t["depth"], slug, t))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    _, _, _, slug, topic = candidates[0]
    row = mastery.get(slug)
    return {
        **topic,
        "mastery_score": row["mastery_score"] if row else 0.0,
        "exposures":     row["exposures"] if row else 0,
        "status":        "unstudied" if row is None else "studied",
    }
