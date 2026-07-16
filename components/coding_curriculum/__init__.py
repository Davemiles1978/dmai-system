"""DMAI Coding Curriculum (PR YY-1).

Gives DMAI a structured curriculum she can study through her existing
LearningPipeline / AutonomousResearcher / KnowledgeAcquirer stack.

Architecture:
    _taxonomy.py     - the topic graph (~800 slugs, generated from a
                       compact spec) with prerequisites, tiers, and
                       depth levels.
    _store.py        - SQLite-backed mastery store: (topic_slug,
                       mastery_score, exposures, last_studied,
                       provenance).
    _picker.py       - "what should DMAI study next" - picks the
                       lowest-mastery topic whose prerequisites are
                       satisfied. Used by fresh_blood + code_study.
    _sources.py      - registers coding knowledge sources (Python
                       docs, PEPs, Flask, SQLite, MDN, TC39, Bash
                       manual) into KnowledgeAcquirer so
                       AutonomousResearcher can pull them on demand.
    _channel.py      - fresh_blood_injector channel:
                       'coding_curriculum'. Emits ONE insight per
                       round targeting the lowest-mastery topic.

Public API is the four functions re-exported below. Everything is
side-effect free until `initialise()` is called.
"""
from __future__ import annotations

from ._taxonomy import (
    CURRICULUM_TOPICS,
    TOPIC_TIERS,
    get_topic,
    all_topic_slugs,
    prerequisites_of,
    tier_of,
)
from ._store import (
    initialise,
    record_exposure,
    mastery_of,
    all_mastery,
    lowest_mastery_topics,
    coverage_summary,
)
from ._picker import next_topic_to_study
from ._channel import inject_coding_curriculum_seeds

__all__ = [
    # Taxonomy
    "CURRICULUM_TOPICS",
    "TOPIC_TIERS",
    "get_topic",
    "all_topic_slugs",
    "prerequisites_of",
    "tier_of",
    # Mastery store
    "initialise",
    "record_exposure",
    "mastery_of",
    "all_mastery",
    "lowest_mastery_topics",
    "coverage_summary",
    # Study
    "next_topic_to_study",
    # fresh_blood channel
    "inject_coding_curriculum_seeds",
]
