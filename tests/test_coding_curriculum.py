"""Tests for the coding-curriculum system (PR YY-1)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from components.coding_curriculum import (
    CURRICULUM_TOPICS,
    all_mastery,
    coverage_summary,
    initialise,
    inject_coding_curriculum_seeds,
    lowest_mastery_topics,
    mastery_of,
    next_topic_to_study,
    record_exposure,
    tier_of,
)
from components.coding_curriculum._taxonomy import DANGLING_PREREQUISITES


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "curriculum_test.db")


# ── Taxonomy sanity ──────────────────────────────────────────────────────

class TestTaxonomyIntegrity:
    def test_at_least_200_topics(self):
        assert len(CURRICULUM_TOPICS) >= 200

    def test_no_dangling_prerequisites(self):
        assert DANGLING_PREREQUISITES == (), (
            f"Dangling prereqs: {DANGLING_PREREQUISITES[:5]}"
        )

    def test_every_topic_has_required_fields(self):
        needed = {"slug", "title", "language", "tier", "depth",
                  "prerequisites", "keywords", "search_queries"}
        for slug, t in CURRICULUM_TOPICS.items():
            assert needed.issubset(t.keys()), f"{slug} missing fields"
            assert 1 <= t["tier"] <= 4
            assert 1 <= t["depth"] <= 5

    def test_tier_1_topics_have_no_prereqs_within_language(self):
        for slug, t in CURRICULUM_TOPICS.items():
            if t["tier"] != 1:
                continue
            for p in t["prerequisites"]:
                # Cross-language prereqs are allowed for tier 1
                # (e.g. python.dsa.list -> cs.dsa.array_list).
                if p.startswith(f"{t['language']}."):
                    prereq = CURRICULUM_TOPICS[p]
                    assert prereq["tier"] <= t["tier"], (
                        f"{slug} (tier {t['tier']}) has same-language "
                        f"prereq {p} of tier {prereq['tier']}"
                    )

    def test_covers_five_languages(self):
        langs = {t["language"] for t in CURRICULUM_TOPICS.values()}
        assert langs >= {"python", "js", "bash", "sql", "cs"}


# ── Store ────────────────────────────────────────────────────────────────

class TestMasteryStore:
    def test_initialise_is_idempotent(self, tmp_db):
        r1 = initialise(db_path=tmp_db)
        r2 = initialise(db_path=tmp_db)
        assert r1["ok"] and r2["ok"]
        assert r2["rows_present"] == 0

    def test_record_exposure_inserts_row(self, tmp_db):
        r = record_exposure(
            "python.core.variables",
            source="test",
            summary="learned var",
            db_path=tmp_db,
        )
        assert r["ok"]
        assert r["mastery_score"] == pytest.approx(0.3)
        row = mastery_of("python.core.variables", db_path=tmp_db)
        assert row["exposures"] == 1
        assert row["last_source"] == "test"

    def test_record_exposure_rejects_unknown_slug(self, tmp_db):
        r = record_exposure(
            "not.a.real.topic", source="test", db_path=tmp_db,
        )
        assert r.get("skipped") is True

    def test_record_exposure_rejects_empty_source(self, tmp_db):
        r = record_exposure(
            "python.core.variables", source="", db_path=tmp_db,
        )
        assert r.get("skipped") is True

    def test_exercise_pass_raises_score(self, tmp_db):
        record_exposure(
            "python.core.variables", source="test", db_path=tmp_db,
        )
        for _ in range(5):
            record_exposure(
                "python.core.variables",
                source="test",
                kind="exercise_pass",
                db_path=tmp_db,
            )
        row = mastery_of("python.core.variables", db_path=tmp_db)
        assert row["mastery_score"] > 0.7
        assert row["exercises_passed"] == 5

    def test_exercise_fail_lowers_score_but_not_below_floor(self, tmp_db):
        record_exposure(
            "python.core.variables", source="test", db_path=tmp_db,
        )
        # Baseline is 0.3; many fails floor at 0.2.
        for _ in range(20):
            record_exposure(
                "python.core.variables",
                source="test",
                kind="exercise_fail",
                db_path=tmp_db,
            )
        row = mastery_of("python.core.variables", db_path=tmp_db)
        assert row["mastery_score"] >= 0.2

    def test_coverage_summary_reflects_progress(self, tmp_db):
        cov0 = coverage_summary(db_path=tmp_db)
        assert cov0["seen"] == 0

        for slug in [
            "python.core.variables",
            "python.core.types_primitives",
            "sql.core.select_where",
        ]:
            record_exposure(slug, source="test", db_path=tmp_db)

        cov1 = coverage_summary(db_path=tmp_db)
        assert cov1["seen"] == 3
        assert cov1["by_language"]["python"]["seen"] == 2
        assert cov1["by_language"]["sql"]["seen"] == 1


# ── Picker ───────────────────────────────────────────────────────────────

class TestPicker:
    def test_picker_returns_a_tier1_topic_first(self, tmp_db):
        initialise(db_path=tmp_db)
        topic = next_topic_to_study(db_path=tmp_db)
        assert topic is not None
        assert topic["tier"] == 1

    def test_picker_respects_language_filter(self, tmp_db):
        initialise(db_path=tmp_db)
        topic = next_topic_to_study(language="sql", db_path=tmp_db)
        assert topic["language"] == "sql"

    def test_picker_advances_after_study(self, tmp_db):
        initialise(db_path=tmp_db)
        first = next_topic_to_study(language="python", db_path=tmp_db)
        # Simulate mastering it.
        for _ in range(15):
            record_exposure(
                first["slug"], source="test",
                kind="exercise_pass", db_path=tmp_db,
            )
        second = next_topic_to_study(language="python", db_path=tmp_db)
        assert second["slug"] != first["slug"]


# ── fresh_blood channel ──────────────────────────────────────────────────

class TestFreshBloodChannel:
    def test_channel_emits_a_seed(self, tmp_db):
        seeds = inject_coding_curriculum_seeds(
            seen=set(), limit=1, db_path=tmp_db,
        )
        assert len(seeds) == 1
        s = seeds[0]
        assert s["channel"] == "coding_curriculum"
        assert s["concept"].startswith("coding_curriculum:")
        assert s["source_url"].startswith("curriculum://")
        assert len(s["seed_hash"]) == 16
        # And the exposure was recorded.
        slug = s["concept"].split(":", 1)[1]
        row = mastery_of(slug, db_path=tmp_db)
        assert row is not None and row["mastery_score"] >= 0.3

    def test_channel_dedups_via_seen(self, tmp_db):
        seeds1 = inject_coding_curriculum_seeds(
            seen=set(), limit=1, db_path=tmp_db,
        )
        hash1 = seeds1[0]["seed_hash"]
        # Second call with hash1 in seen should skip that slug and
        # either emit nothing or emit a different topic.
        seeds2 = inject_coding_curriculum_seeds(
            seen={hash1}, limit=1, db_path=tmp_db,
        )
        for s in seeds2:
            assert s["seed_hash"] != hash1

    def test_channel_never_writes_zero_mastery_rows(self, tmp_db):
        # Emit a few rounds, then check every row has mastery > 0.
        for _ in range(3):
            inject_coding_curriculum_seeds(
                seen=set(), limit=1, db_path=tmp_db,
            )
        for slug, row in all_mastery(db_path=tmp_db).items():
            assert row["mastery_score"] > 0
            assert row["last_source"]  # never empty
