"""fresh_blood_injector channel: coding_curriculum.

Every round DMAI's fresh_blood_injector runs, this channel emits one
seed insight targeting the lowest-mastery topic whose prerequisites
are satisfied. The seed goes through the normal insight -> capability
pipeline and (crucially) records an 'exposure' event in the mastery
store so the picker moves on next round.

Contract with fresh_blood_injector:
    Returns a list of seed dicts with the same shape as other channels:
        {
            "channel":      "coding_curriculum",
            "concept":      "coding_curriculum:<slug>",
            "insight_text": "...",
            "source_url":   "curriculum://<slug>",
            "seed_hash":    "<16 hex chars>",
        }
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from ._picker import next_topic_to_study
from ._store import record_exposure, initialise

logger = logging.getLogger(__name__)


def _seed_hash(slug: str) -> str:
    return hashlib.sha256(
        f"coding_curriculum::{slug}".encode("utf-8"),
    ).hexdigest()[:16]


def _insight_text(topic: dict) -> str:
    """Compose the learning insight for a topic.

    This is the prompt DMAI sees. It nudges the downstream reasoning
    pipeline (autonomous_researcher / capability_promoter) toward
    treating this as a coding-education seed rather than a generic
    research topic.
    """
    slug = topic["slug"]
    keywords = topic.get("keywords") or []
    queries = topic.get("search_queries") or []
    kw_line = ", ".join(keywords[:6]) if keywords else ""
    query_hint = queries[0] if queries else topic["title"]
    return (
        f"CODING CURRICULUM TOPIC: {topic['title']} "
        f"(language={topic['language']}, tier={topic['tier']}, "
        f"depth={topic['depth']}).\n"
        f"Slug: {slug}. Keywords: {kw_line}.\n"
        f"Research goal: learn this topic to expert level. Study "
        f"official docs, canonical examples, common mistakes, and "
        f"design patterns. Produce a compact reference you can retrieve "
        f"later when writing code that uses this concept.\n"
        f"Suggested search: '{query_hint}'."
    )


def inject_coding_curriculum_seeds(*,
                                   seen: Optional[set] = None,
                                   limit: int = 1,
                                   db_path: str = "data/dmai_knowledge.db",
                                   ) -> List[Dict[str, Any]]:
    """Emit seed(s) for the coding_curriculum fresh_blood channel.

    Called by fresh_blood_injector during a round. Records an
    'exposure' event in the mastery store for each emitted seed so
    the next call picks a different topic.

    Args:
        seen:    Set of recent seed_hash values from fresh_blood's log
                 (for dedup).
        limit:   Max seeds to emit per round. Default 1 - we don't
                 flood a single round with curriculum topics.
        db_path: SQLite path.
    """
    # Ensure the mastery table exists (idempotent).
    try:
        initialise(db_path=db_path)
    except Exception as e:  # noqa: BLE001
        logger.info("coding_curriculum: initialise failed: %s", e)
        return []

    seen = seen or set()
    seeds: List[Dict[str, Any]] = []
    tried_slugs: set = set()

    for _ in range(limit):
        topic = next_topic_to_study(db_path=db_path)
        if topic is None:
            break
        slug = topic["slug"]
        if slug in tried_slugs:
            break  # picker returned the same slug twice → nothing else
        tried_slugs.add(slug)

        seed_hash = _seed_hash(slug)
        if seed_hash in seen:
            # This exact topic was seeded recently; bump exposure
            # (moves mastery slightly) so the picker rotates, and
            # skip this round rather than emit a duplicate.
            record_exposure(
                slug, source="coding_curriculum_dedup",
                summary="dedup-skip", db_path=db_path,
            )
            continue

        seeds.append({
            "channel":      "coding_curriculum",
            "concept":      f"coding_curriculum:{slug}",
            "insight_text": _insight_text(topic),
            "source_url":   f"curriculum://{slug}",
            "seed_hash":    seed_hash,
        })

        # Record the exposure - satisfies user rule 'never insert None
        # values': every emitted seed persists a real mastery row with
        # non-zero score (0.3 for first exposure).
        record_exposure(
            slug,
            source="fresh_blood.coding_curriculum",
            summary=(
                f"seeded via fresh_blood_injector round; "
                f"tier={topic['tier']} depth={topic['depth']}"
            ),
            db_path=db_path,
        )

    return seeds
