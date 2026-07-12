"""Tests for the JSON error handlers on /api/* (PR C).

Root cause covered: Flask's default 404/405/500 responses are HTML pages,
which broke the admin panel's api() helper (it called response.json() on
every reply). The new handlers return JSON for /api/* paths so the client
can render partial state even when a single endpoint fails.

These tests build a minimal Flask app that reproduces the handler setup
from dmai_core_complete rather than importing that module wholesale (it
takes ~30 s to import due to the boot side effects). Behaviour under
test is the handlers themselves.
"""
from __future__ import annotations

import json

import pytest
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException as _HTTPException


@pytest.fixture
def app():
    app = Flask(__name__)

    # ── Same handler setup as dmai_core_complete ─────────────────────────
    def _wants_json_error() -> bool:
        try:
            path = request.path or ""
        except Exception:
            return False
        if path.startswith("/api/"):
            return True
        try:
            accept = request.headers.get("Accept", "") or ""
        except Exception:
            accept = ""
        return "application/json" in accept.lower()

    @app.errorhandler(_HTTPException)
    def _api_http_error(e):
        if not _wants_json_error():
            return e
        return jsonify({
            "ok": False,
            "status": e.code,
            "error": e.name,
            "message": e.description,
        }), e.code

    @app.errorhandler(Exception)
    def _api_unhandled_error(e):
        if not _wants_json_error():
            raise e
        return jsonify({
            "ok": False,
            "status": 500,
            "error": "Internal Server Error",
            "message": str(e),
        }), 500

    # ── Minimal routes ───────────────────────────────────────────────────
    @app.route("/api/only-post", methods=["POST"])
    def only_post():
        return jsonify({"ok": True})

    @app.route("/api/exploding")
    def exploding():
        raise RuntimeError("boom")

    @app.route("/browser-page")
    def browser_page():
        return "<html>...</html>"

    return app


def test_api_404_returns_json(app):
    c = app.test_client()
    r = c.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.content_type.startswith("application/json")
    body = r.get_json()
    assert body["ok"] is False
    assert body["status"] == 404
    assert body["error"] == "Not Found"


def test_api_405_returns_json_not_html(app):
    """The exact case that was breaking the admin panel: GET on a
    POST-only endpoint used to render Flask's HTML 405 page."""
    c = app.test_client()
    r = c.get("/api/only-post")
    assert r.status_code == 405
    assert r.content_type.startswith("application/json")
    body = r.get_json()
    assert body["ok"] is False
    assert body["status"] == 405
    assert body["error"] == "Method Not Allowed"
    # Body must not contain HTML tags — that's the whole point.
    raw = r.get_data(as_text=True)
    assert "<html" not in raw.lower()
    assert "<!doctype" not in raw.lower()


def test_api_unhandled_exception_returns_json_500(app):
    c = app.test_client()
    r = c.get("/api/exploding")
    assert r.status_code == 500
    assert r.content_type.startswith("application/json")
    body = r.get_json()
    assert body["ok"] is False
    assert body["status"] == 500
    assert body["error"] == "Internal Server Error"
    assert body["message"] == "boom"


def test_non_api_404_still_returns_html_for_browser(app):
    """Handlers must NOT hijack browser 404s (those still render HTML,
    so bookmarks / typos in the URL bar look like normal 404 pages)."""
    c = app.test_client()
    r = c.get("/no-such-page")
    assert r.status_code == 404
    # werkzeug default 404 is text/html
    assert "text/html" in (r.content_type or "")


def test_json_error_honours_accept_header(app):
    """Even a non-/api/ path returns JSON when the caller asks for it."""
    c = app.test_client()
    r = c.get("/no-such-page", headers={"Accept": "application/json"})
    assert r.status_code == 404
    assert r.content_type.startswith("application/json")
    body = r.get_json()
    assert body["ok"] is False
    assert body["status"] == 404


def test_json_error_body_snippet_is_useful(app):
    """The 404/500 JSON must include a human-readable 'message'."""
    c = app.test_client()
    for path in ("/api/does-not-exist", "/api/exploding"):
        r = c.get(path)
        body = r.get_json()
        assert isinstance(body.get("message"), str)
        assert body["message"]
