"""Tests for components.generated._self_judge_review.

We monkeypatch judge_seed to avoid hitting the real vocabulary
tables - this file is only proving the wrapper shape (accept vs
reject vs defer).
"""
from __future__ import annotations

from components.generated import _self_judge_review as review


class _FakeVerdict:
    def __init__(self, verdict, confidence, reason,
                 knowledge_gap=""):
        self.verdict = verdict
        self.confidence = confidence
        self.reason = reason
        self.knowledge_gap = knowledge_gap


def test_review_accept(monkeypatch):
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("accept", 0.72, "aligned"))
    r = review.review_generated_module(
        concept="add two numbers",
        channel="utility",
        docstring="Adds two numbers together and returns the sum.",
    )
    assert r.ok is True
    assert r.verdict == "accept"
    assert r.confidence == 0.72


def test_review_reject(monkeypatch):
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("reject", 0.05, "drift"))
    r = review.review_generated_module(
        concept="add two numbers",
        channel="utility",
        docstring="This module renders a fractal.",
    )
    assert r.ok is False
    assert r.verdict == "reject"


def test_review_defer(monkeypatch):
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("defer", 0.4, "ambiguous",
                                     knowledge_gap="need_more_signal"))
    r = review.review_generated_module(
        concept="ambiguous",
        channel="utility",
        docstring="Does something.",
    )
    # defer on a non-gap channel is not "accept" -> ok is False, so
    # the materialiser will retry with the LLM fallback
    assert r.ok is False
    assert r.verdict == "defer"
    assert r.gap_summary == "need_more_signal"


# ── PR AAA-4: gap_driven defer tolerance ────────────────────────────

def test_review_defer_on_gap_driven_is_ok(monkeypatch):
    """PR AAA-4: gap-authored seeds treat 'defer' as ok. The docstring
    was already vetted upstream by the gap-analyser and the code has
    passed happy-path + smoke gates before we get here. An uncertain-
    band defer is just weighted-signal noise, not a real quality
    signal."""
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("defer", 0.38,
                                     "confidence=0.38 in uncertain band",
                                     knowledge_gap="low_insight_overlap"))
    r = review.review_generated_module(
        concept="gap consistency assertion cron",
        channel="gap_driven",
        docstring="Runs a periodic consistency check.",
    )
    assert r.ok is True   # <-- key change: gap defers pass
    assert r.verdict == "defer"
    assert r.confidence == 0.38


def test_review_defer_on_gap_underscore_channel_also_ok(monkeypatch):
    """The normaliser maps `gap_*` channels to `gap_driven` before
    calling judge_seed - confirm the tolerance still triggers."""
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("defer", 0.42, "uncertain"))
    r = review.review_generated_module(
        concept="gap semantic drift monitor",
        channel="gap_semantic_drift",  # gets normalised to gap_driven
        docstring="Monitors semantic drift.",
    )
    assert r.ok is True
    assert r.verdict == "defer"


def test_review_reject_on_gap_driven_is_still_not_ok(monkeypatch):
    """Rejects stay strict even on gap channels - if the docstring has
    genuinely drifted from the concept, we still bounce."""
    monkeypatch.setattr(review, "judge_seed",
                        lambda seed, conn, accept_threshold=0.55:
                        _FakeVerdict("reject", 0.05,
                                     "near-duplicate of insight i1"))
    r = review.review_generated_module(
        concept="gap infrastructure monitor",
        channel="gap_driven",
        docstring="Completely unrelated fractal renderer.",
    )
    assert r.ok is False   # <-- rejects still bounce
    assert r.verdict == "reject"
