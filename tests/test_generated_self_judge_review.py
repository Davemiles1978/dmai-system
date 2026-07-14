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
    # defer is not "accept" -> ok is False, so the materialiser will
    # retry with the LLM fallback
    assert r.ok is False
    assert r.verdict == "defer"
    assert r.gap_summary == "need_more_signal"
