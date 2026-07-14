"""Tests for the /admin/procurement HTML page (PR K.1).

The page is server-side rendered: the route reads the procurement store
directly and injects the shortlist rows into the HTML body, so the rows
are assertable without executing any client-side JS.

DATA_PATH is pointed at a temp dir *before* importing the app so the
procurement store (and the app's boot side effects) stay isolated. A full
research pass is run against seeded workload + treasury DBs so the store
holds a real 7-row shortlist.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

_TMP = tempfile.mkdtemp(prefix="proc_page_")
os.environ["DATA_PATH"] = _TMP

from components.workload import workload_profiler as wp  # noqa: E402
from components.treasury import treasury_ledger as tl  # noqa: E402
from components.procurement import researcher  # noqa: E402


def _seed_and_run():
    """Seed workload + treasury at their *default* DATA_PATH locations and
    run one research pass.

    We seed the default DBs (not tmp-named ones) so that any research run
    the app kicks off at boot uses the same seeded inputs and produces a
    consistent last_summary (treasury balance, top-pick capex).
    """
    from components.workload.workload_profiler import default_workload_path
    from components.treasury.treasury_ledger import default_treasury_path
    wlp = default_workload_path()
    trp = default_treasury_path()
    wp.init_workload_db(wlp)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(wlp) as c:
        for cpu, rss in [(1000.0, 210.0), (1000.0 + 43200.0, 420.0)]:
            c.execute(
                "INSERT INTO workload_samples"
                "(ts, cpu_percent, cpu_seconds_total, mem_rss_mb, "
                " mem_peak_rss_mb) VALUES (?,?,?,?,?)",
                (now, 10.0, cpu, rss, rss),
            )
        c.commit()
    tl.init_treasury_db(trp)
    tl.record_manual(kind="manual_credit", amount_gbp=5000.0,
                     description="seed", db_path=trp)
    # Writes procurement DB + last_summary at the default DATA_PATH location.
    return researcher.run_research()


@pytest.fixture(scope="module")
def client():
    summary = _seed_and_run()
    assert summary["candidate_count"] == 7
    from dmai_core_complete import app
    app.config["TESTING"] = True
    return app.test_client(), summary


def test_page_returns_html(client):
    c, _ = client
    resp = c.get("/admin/procurement")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type


def test_page_contains_all_seven_shortlist_rows(client):
    c, _ = client
    body = c.get("/admin/procurement").get_data(as_text=True)
    # Each rank 1..7 is rendered in a data-sort'd rank cell.
    for rank in range(1, 8):
        assert f'data-sort="{rank}"' in body


def test_page_contains_treasury_and_top_pick(client):
    c, summary = client
    body = c.get("/admin/procurement").get_data(as_text=True)
    # Treasury balance (£5,000.00) and the rank-1 capex (£449.00).
    assert "5,000.00" in body
    assert "449.00" in body


def test_force_refresh_button_posts_to_run_endpoint(client, monkeypatch):
    c, _ = client
    body = c.get("/admin/procurement").get_data(as_text=True)
    # Button wiring: POST to the run endpoint.
    assert "/api/admin/procurement-run" in body
    assert "method: 'POST'" in body

    # And the endpoint itself answers a POST with 200 (run stubbed).
    import components.procurement.researcher as _pr
    monkeypatch.setattr(_pr, "run_research",
                        lambda **kw: {"run_ts": "stub-run"})
    try:
        import components.procurement.loop as _loop
        monkeypatch.setattr(_loop, "_LOOP", None, raising=False)
    except Exception:
        pass
    resp = c.post("/api/admin/procurement-run")
    assert resp.status_code == 200
