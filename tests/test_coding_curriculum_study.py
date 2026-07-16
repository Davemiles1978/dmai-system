"""Tests for the study loop (PR YY-2)."""
from __future__ import annotations

import sqlite3
import time

import pytest

from components.coding_curriculum import (
    Exercise,
    GradingCase,
    all_supported_shapes,
    exercise_for_topic,
    grade_exercise,
    initialise,
    read_study_log,
    run_study_batch,
    run_study_round,
    study_stats,
)


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    # Redirect the study-log to a per-test file so runs don't interfere.
    from components.coding_curriculum import _study_engine
    monkeypatch.setattr(
        _study_engine, "_LOG_PATH", tmp_path / "study_log.jsonl",
    )
    return str(tmp_path / "curriculum_test.db")


# ── Exercises ───────────────────────────────────────────────────────────

class TestExerciseBuilders:
    def test_every_supported_shape_is_reachable(self):
        # Build one exercise per language x tier sample and confirm
        # only supported shapes are emitted.
        from components.coding_curriculum._taxonomy import CURRICULUM_TOPICS
        shapes_seen = set()
        for slug, t in CURRICULUM_TOPICS.items():
            ex = exercise_for_topic(slug)
            assert ex is not None
            assert ex.capability_shape in all_supported_shapes()
            shapes_seen.add(ex.capability_shape)
        # We expect the taxonomy to hit at least 3 distinct shapes.
        assert len(shapes_seen) >= 3

    def test_unknown_topic_returns_none(self):
        assert exercise_for_topic("not.a.real.topic") is None

    def test_exercise_id_is_stable(self):
        e1 = exercise_for_topic("python.core.variables")
        e2 = exercise_for_topic("python.core.variables")
        assert e1.exercise_id == e2.exercise_id


# ── Grader ──────────────────────────────────────────────────────────────

class TestGrader:
    def _ex(self):
        return Exercise(
            exercise_id="test",
            topic_slug="python.core.variables",
            brief="doubler",
            signature="def run(**kwargs) -> dict:",
            docstring="test",
            hint="",
            capability_shape="utility",
            grading=[
                GradingCase(
                    kwargs={"values": [1, 2, 3]},
                    predicate=(
                        "result.get('ok') is True and "
                        "list(result.get('result', [])) == [2, 4, 6]"
                    ),
                    description="basic",
                ),
            ],
        )

    def test_correct_solution_passes(self):
        ex = self._ex()
        code = (
            "def run(**kwargs):\n"
            "    return {'ok': True, 'result': [v*2 for v in kwargs.get('values', [])]}\n"
        )
        r = grade_exercise(ex, code)
        assert r["ok"] is True
        assert r["cases"][0]["passed"] is True
        assert r["runtime_ms"] >= 0

    def test_syntax_error_fails_cleanly(self):
        ex = self._ex()
        r = grade_exercise(ex, "def run(**kwargs)\n    return {}\n")
        assert r["ok"] is False
        assert "SyntaxError" in (r["reason"] or "")

    def test_missing_run_fails_cleanly(self):
        ex = self._ex()
        r = grade_exercise(ex, "x = 42\n")
        assert r["ok"] is False
        assert r["cases"][0]["error"].startswith("no run() function") or \
               "no run" in r["cases"][0]["error"]

    def test_infinite_loop_times_out(self):
        ex = self._ex()
        code = "def run(**kwargs):\n    while True:\n        pass\n"
        t0 = time.time()
        r = grade_exercise(ex, code, timeout_seconds=1.5)
        elapsed = time.time() - t0
        assert r["ok"] is False
        assert "timeout" in r["cases"][0]["error"].lower()
        # Must actually stop near the timeout, not hang forever.
        assert elapsed < 5.0

    def test_wrong_answer_fails_cleanly(self):
        ex = self._ex()
        code = "def run(**kwargs):\n    return {'ok': True, 'result': [0, 0, 0]}\n"
        r = grade_exercise(ex, code)
        assert r["ok"] is False
        assert r["cases"][0]["passed"] is False

    def test_candidate_exception_reports_error(self):
        ex = self._ex()
        code = "def run(**kwargs):\n    raise ValueError('boom')\n"
        r = grade_exercise(ex, code)
        assert r["ok"] is False
        assert "ValueError" in r["cases"][0]["error"]


# ── Study engine end-to-end ─────────────────────────────────────────────

class TestStudyEngine:
    def test_single_round_records_exposure(self, tmp_db):
        initialise(db_path=tmp_db)
        r = run_study_round(db_path=tmp_db)
        assert r["ok"] is True
        # Whatever the topic, its mastery must be non-zero after study.
        from components.coding_curriculum import mastery_of
        row = mastery_of(r["topic"], db_path=tmp_db)
        assert row is not None
        assert row["mastery_score"] > 0.0

    def test_single_round_pass_promotes_capability(self, tmp_db):
        r = run_study_round(db_path=tmp_db)
        # On a pass, we expect a coding_pattern row to exist.
        if r.get("passed"):
            conn = sqlite3.connect(tmp_db)
            try:
                rows = conn.execute(
                    "SELECT capability_type FROM capabilities "
                    "WHERE capability_type = 'coding_pattern'"
                ).fetchall()
            finally:
                conn.close()
            assert len(rows) >= 1
            assert r["capability_name"].startswith("coding_pattern__")

    def test_batch_returns_summary(self, tmp_db):
        s = run_study_batch(n=3, db_path=tmp_db)
        assert s["ok"] is True
        assert s["rounds"] == 3
        assert 0 <= s["passes"] <= 3
        assert 0 <= s["promotions"] <= s["passes"]

    def test_stats_track_pass_rate(self, tmp_db):
        run_study_batch(n=4, db_path=tmp_db)
        s = study_stats(db_path=tmp_db)
        assert s["ok"] is True
        assert s["log_entries"] >= 4
        assert 0.0 <= s["pass_rate"] <= 1.0
        assert s["curriculum_size"] >= 200

    def test_log_is_newest_first(self, tmp_db):
        run_study_batch(n=3, db_path=tmp_db)
        log = read_study_log(limit=10)
        assert len(log) >= 3
        # Timestamps should be non-increasing (newest first).
        ts = [x["ts"] for x in log]
        assert ts == sorted(ts, reverse=True)

    def test_never_writes_none_or_empty_source(self, tmp_db):
        run_study_batch(n=3, db_path=tmp_db)
        from components.coding_curriculum import all_mastery
        for slug, row in all_mastery(db_path=tmp_db).items():
            assert row["mastery_score"] > 0.0
            assert row["last_source"] and row["last_source"].strip()

    def test_no_hang_on_empty_topics(self, tmp_db, monkeypatch):
        # Simulate 'no topic to study' - engine returns cleanly.
        from components.coding_curriculum import _study_engine
        monkeypatch.setattr(
            _study_engine, "next_topic_to_study",
            lambda **_kw: None,
        )
        r = run_study_round(db_path=tmp_db)
        assert r["ok"] is False
        assert "no topic" in r["reason"].lower()
