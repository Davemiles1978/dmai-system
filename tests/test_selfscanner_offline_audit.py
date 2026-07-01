"""Tests for PR #161 — SelfScanner offline route audit (replace test_client).

`SelfScanner._audit_routes` used to dispatch in-process `test_client().get()`
probes against every GET route. Those synthetic requests ran the real view
handlers; handlers like persona_registry.resolve() acquire the SQLite write
mutex mid-request, starving the vocabulary_ingester flush (write_mutex_timeout
/ "database is locked" churn — see post_pr160_verification.md).

The audit now walks Flask's url_map and introspects view functions offline. No
request is dispatched, no write mutex is acquired, no DB connection is opened.
"""

import sqlite3
import sys
from pathlib import Path
from unittest import mock

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.self_scanner import SelfScanner


def _make_app(write_spy=None):
    """App with a reading route, a writing route, an args route, and a stub."""
    app = Flask(__name__)

    @app.route("/read")
    def read():
        return {"status": "ok"}, 200

    @app.route("/write", methods=["POST"])
    def write():
        if write_spy is not None:
            write_spy()
        return {"status": "written"}, 200

    @app.route("/item/<int:item_id>")
    def item(item_id):
        return {"status": "ok", "id": item_id}, 200

    @app.route("/stub")
    def stub():
        return {"status": "not_implemented"}, 200

    return app


def test_audit_dispatches_no_requests_and_opens_no_db():
    app = _make_app()
    scanner = SelfScanner(app=app)

    # A view whose dispatch would open a DB connection — must never run.
    with mock.patch.object(app, "test_client") as fake_client, \
         mock.patch.object(sqlite3, "connect") as fake_connect:
        broken = scanner._audit_routes()

    fake_client.assert_not_called()
    fake_connect.assert_not_called()
    assert isinstance(broken, list)


def test_route_count_matches_url_map():
    app = _make_app()
    scanner = SelfScanner(app=app)
    assert len(scanner._iter_route_rules()) == len(list(app.url_map.iter_rules()))


def test_stub_route_flagged_offline():
    app = _make_app()
    scanner = SelfScanner(app=app)
    broken = scanner._audit_routes()
    paths = {r["path"] for r in broken}
    assert "/stub" in paths
    # Healthy reading route is not flagged.
    assert "/read" not in paths
    # Non-GET writing route is skipped by the audit.
    assert "/write" not in paths


def test_missing_view_function_flagged():
    app = _make_app()
    # Simulate a rule whose endpoint has no registered view function.
    app.view_functions.pop("stub", None)
    scanner = SelfScanner(app=app)
    broken = scanner._audit_routes()
    assert any(r["path"] == "/stub" and r["error"] == "no_view_function" for r in broken)


def test_audit_does_not_disturb_write_path():
    """Regression: the audit must not invoke any view handler, so a route that
    would write on dispatch (mirroring persona_registry.resolve acquiring the
    write mutex / triggering a vocab flush) is never executed."""
    write_spy = mock.Mock()
    app = _make_app(write_spy=write_spy)

    # Add a GET route that writes on dispatch, like /api/personas/resolve.
    @app.route("/api/personas/resolve")
    def resolve():
        write_spy()
        return {"status": "ok"}, 200

    scanner = SelfScanner(app=app)
    scanner._audit_routes()

    # Old test_client() audit would have dispatched this GET and called the
    # write path; the offline audit never does.
    assert write_spy.call_count == 0
