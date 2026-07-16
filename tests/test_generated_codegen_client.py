"""Tests for components.generated._codegen_client.

We do NOT hit the network. We monkeypatch _post_openrouter and
verify prompt shape, cascade behaviour, code extraction, and
graceful failure paths.
"""
from __future__ import annotations

import json

import pytest

from components.generated import _codegen_client as cg


def _fake_response(content: str) -> dict:
    return {
        "choices": [
            {"message": {"content": content}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def test_extract_code_strips_python_fence():
    text = "here you go:\n```python\n"\
           'def run():\n    return 1\n```\n'
    assert 'def run()' in cg._extract_code(text)
    assert '```' not in cg._extract_code(text)


def test_extract_code_no_fence_returns_as_is():
    text = 'def run():\n    return 2\n'
    assert cg._extract_code(text).startswith("def run")


def test_request_code_returns_ok_when_response_has_run(monkeypatch):
    """PR XX-1: LLM path is now tier-2. Force the LLM path by using an
    unknown capability_type so the local template layer declines."""
    def fake(model, msgs, max_tokens):
        return _fake_response(
            '"""Doc."""\ndef run(**kw):\n    return 42\n'
        )
    monkeypatch.setattr(cg, "_post_openrouter", fake)
    att = cg.request_code(
        concept="sum", insight="add numbers",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert att.ok
    assert "def run" in att.source
    assert att.model == cg.MODEL_PRIMARY


def test_request_code_flags_missing_run(monkeypatch):
    def fake(model, msgs, max_tokens):
        return _fake_response("Sorry, I can't help with that.")
    monkeypatch.setattr(cg, "_post_openrouter", fake)
    att = cg.request_code(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert not att.ok
    assert att.reason == "no_run_in_response"


def test_request_code_flags_http_failure(monkeypatch):
    # PR VV: None from _post_openrouter now specifically means
    # OPENROUTER_API_KEY is unset (the only path that still returns
    # None). Real HTTP failures come back as a dict with __error__.
    monkeypatch.setattr(cg, "_post_openrouter", lambda *a, **k: None)
    att = cg.request_code(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert not att.ok
    assert att.reason == "openrouter_key_unset"


def test_request_code_surfaces_http_error_details(monkeypatch):
    """PR VV: HTTP failures produce a concrete reason with status +
    body snippet instead of the opaque http_or_auth_failure."""
    monkeypatch.setattr(cg, "_post_openrouter", lambda *a, **k: {
        "__error__": "http_401",
        "http_status": 401,
        "body_snippet": '{"error":{"message":"invalid api key"}}',
    })
    att = cg.request_code(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert not att.ok
    assert "http_401" in att.reason
    assert "401" in att.reason
    assert "invalid api key" in att.reason


def test_request_code_flags_malformed(monkeypatch):
    monkeypatch.setattr(cg, "_post_openrouter",
                        lambda *a, **k: {"choices": []})
    att = cg.request_code(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert not att.ok
    assert att.reason == "malformed_response"


def test_cascade_stops_on_primary_success(monkeypatch):
    calls = []
    def fake(model, msgs, max_tokens):
        calls.append(model)
        return _fake_response('"""D."""\ndef run(**k):\n    return 1\n')
    monkeypatch.setattr(cg, "_post_openrouter", fake)
    atts = cg.request_code_cascade(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert len(atts) == 1
    assert atts[0].ok
    assert calls == [cg.MODEL_PRIMARY]


def test_cascade_falls_back_when_primary_fails(monkeypatch):
    calls = []
    def fake(model, msgs, max_tokens):
        calls.append(model)
        if model == cg.MODEL_PRIMARY:
            return _fake_response("nope no code")
        return _fake_response('"""D."""\ndef run(**k):\n    return 2\n')
    monkeypatch.setattr(cg, "_post_openrouter", fake)
    atts = cg.request_code_cascade(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
    )
    assert len(atts) == 2
    assert not atts[0].ok
    assert atts[1].ok
    assert calls == [cg.MODEL_PRIMARY, cg.MODEL_FALLBACK]


def test_retry_hint_is_appended(monkeypatch):
    captured = {}
    def fake(model, msgs, max_tokens):
        captured["msgs"] = msgs
        return _fake_response('"""D."""\ndef run(**k):\n    return 3\n')
    monkeypatch.setattr(cg, "_post_openrouter", fake)
    cg.request_code(
        concept="c", insight="i",
        capability_type="unknown_shape_forces_llm", happy_kwargs={},
        retry_reasons=["banned_call: eval", "missing_docstring"],
    )
    joined = "\n".join(m["content"] for m in captured["msgs"])
    assert "banned_call: eval" in joined
    assert "missing_docstring" in joined


def test_openrouter_call_skipped_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Should short-circuit to None without attempting HTTP
    assert cg._post_openrouter("m", [], 10) is None
