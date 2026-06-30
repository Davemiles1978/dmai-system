"""Tests for PR #160 — CB-01 ignores internal audit probes (X-Internal-Probe).

`SelfScanner._audit_routes` walks every GET route at startup with an in-process
`test_client().get()`. Routes that 500 in that synthetic pass were being counted
as real 5xx by `after_request_hook`, tripping CB-01 at ~2.8%.

The fix tags every probe with `X-Internal-Probe: 1` and makes `after_request_hook`
early-return for those responses before `record_response` runs. These tests pin
both halves plus the end-to-end behaviour through a live scanner run.
"""

import sys
from pathlib import Path

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent))

from circuit_breaker import CircuitBreakerManager, after_request_hook
from components.self_scanner import SelfScanner


@pytest.fixture
def manager():
    """Fresh CB manager singleton per test so the CB-01 ring buffer is clean."""
    CircuitBreakerManager._instance = None
    mgr = CircuitBreakerManager.get()
    yield mgr
    CircuitBreakerManager._instance = None


def _make_app():
    app = Flask(__name__)

    @app.route("/ok")
    def ok():
        return {"status": "ok"}, 200

    @app.route("/boom")
    def boom():
        return {"status": "error"}, 500

    app.after_request(after_request_hook)
    return app


def test_internal_probe_response_not_counted(manager):
    app = _make_app()
    with app.test_client() as client:
        resp = client.get("/boom", headers={"X-Internal-Probe": "1"})
    assert resp.status_code == 500
    # Nothing recorded: the probe was filtered before record_response.
    assert len(manager._error_window) == 0


def test_normal_request_still_counted(manager):
    app = _make_app()
    with app.test_client() as client:
        resp = client.get("/boom")
    assert resp.status_code == 500
    # The real 5xx was recorded.
    assert len(manager._error_window) == 1
    _, is_5xx = manager._error_window[0]
    assert is_5xx is True


def test_scanner_sends_internal_probe_header(manager):
    seen_headers = []

    app = Flask(__name__)

    @app.before_request
    def _capture():
        from flask import request
        seen_headers.append(dict(request.headers))

    @app.route("/healthy")
    def healthy():
        return {"status": "ok"}, 200

    @app.route("/broken")
    def broken():
        return {"status": "error"}, 500

    app.after_request(after_request_hook)

    scanner = SelfScanner(app=app)
    broken_routes = scanner._audit_routes()

    # The always-failing route is detected by the audit...
    assert any(r["path"] == "/broken" for r in broken_routes)
    # ...every probe carried the X-Internal-Probe header...
    assert seen_headers, "scanner issued no probes"
    assert all(h.get("X-Internal-Probe") == "1" for h in seen_headers)
    # ...and the synthetic 500 from the audit run was NOT recorded by CB-01.
    assert len(manager._error_window) == 0
