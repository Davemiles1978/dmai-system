"""Tests for components.capability_verifier.

Covers:
- Happy path: healthy module passes both stages.
- Stage 1 failure: broken module -> quarantine + revert.
- Stage 2 failure: imports OK but runtime crash -> quarantine + revert.
- Retry cap: after 3 attempts, module is permanently quarantined.
- get_retry_guidance returns the last failure traceback for codegen.
- verification_status returns the expected snapshot shape.
- Cache: recent success short-circuits.
"""
from __future__ import annotations

import shutil
import sqlite3
import textwrap
from pathlib import Path

import pytest

from components.capability_verifier import (
    DEFAULT_MAX_ATTEMPTS,
    LIVE_DIR,
    QUARANTINE_DIR,
    VerificationResult,
    ensure_verification_table,
    get_retry_guidance,
    verification_status,
    verify_promoted,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_kdb(tmp_path: Path) -> str:
    path = tmp_path / "dmai_knowledge.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE capabilities (
            id TEXT PRIMARY KEY,
            name TEXT,
            capability_type TEXT,
            runtime_mode TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(autouse=True)
def _clean_live_and_quarantine():
    """Purge any test-authored live/quarantine modules between tests."""
    for base in (LIVE_DIR, QUARANTINE_DIR):
        if base.exists():
            for p in base.glob("__pytest_verifier_*.py"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
    yield
    for base in (LIVE_DIR, QUARANTINE_DIR):
        if base.exists():
            for p in base.glob("__pytest_verifier_*.py"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass


def _write_live(slug: str, body: str) -> Path:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_DIR / f"{slug}.py"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def _seed_cap(kdb: str, cap_id: str, slug: str, cap_type: str = "utility"):
    conn = sqlite3.connect(kdb)
    conn.execute(
        "INSERT INTO capabilities VALUES (?, ?, ?, 'generated_module')",
        (cap_id, slug, cap_type),
    )
    conn.commit()
    conn.close()


# ── Happy path ────────────────────────────────────────────────────────────

def test_verify_promoted_healthy_module_passes_both_stages(tmp_kdb: str) -> None:
    slug = "__pytest_verifier_happy"
    _write_live(slug, '''
        """Docstring for happy module."""
        def run(**kwargs):
            return {"status": "ok"}
    ''')
    _seed_cap(tmp_kdb, "cap-happy", slug)

    result = verify_promoted(
        cap_id="cap-happy",
        slug=slug,
        capability_type="utility",
        db_path=tmp_kdb,
        isolated_timeout_sec=15,
        orchestrator_timeout_sec=15,
        use_cache=False,
    )
    assert isinstance(result, VerificationResult)
    assert result.ok, f"expected ok, got: {result}"
    assert result.stage == "orchestrator"
    assert not result.reverted
    assert not result.quarantined

    # Live file should still be present
    assert (LIVE_DIR / f"{slug}.py").exists()

    # runtime_mode should still be generated_module (unchanged)
    conn = sqlite3.connect(tmp_kdb)
    mode = conn.execute(
        "SELECT runtime_mode FROM capabilities WHERE id = 'cap-happy'"
    ).fetchone()[0]
    conn.close()
    assert mode == "generated_module"


# ── Stage 1 failure: syntax/import broken ─────────────────────────────────

def test_verify_promoted_stage1_syntax_error_quarantines(tmp_kdb: str) -> None:
    slug = "__pytest_verifier_broken_syntax"
    _write_live(slug, '''
        """Broken module."""
        def run(**kwargs
            return "missing paren above"
    ''')
    _seed_cap(tmp_kdb, "cap-brk-1", slug)

    result = verify_promoted(
        cap_id="cap-brk-1",
        slug=slug,
        capability_type="utility",
        db_path=tmp_kdb,
        isolated_timeout_sec=15,
        use_cache=False,
    )
    assert not result.ok
    assert result.stage == "isolated"
    assert result.reverted
    assert result.quarantined
    assert result.traceback  # must have a traceback for codegen retry

    # Live file must be gone
    assert not (LIVE_DIR / f"{slug}.py").exists()

    # runtime_mode should be stub_reverted (retryable, first attempt)
    conn = sqlite3.connect(tmp_kdb)
    mode = conn.execute(
        "SELECT runtime_mode FROM capabilities WHERE id = 'cap-brk-1'"
    ).fetchone()[0]
    conn.close()
    assert mode == "stub_reverted"


# ── Stage 2 failure: imports ok but crashes at runtime ────────────────────

def test_verify_promoted_stage2_runtime_crash_quarantines(tmp_kdb: str) -> None:
    slug = "__pytest_verifier_stage2_crash"
    _write_live(slug, '''
        """This imports fine and run() exists but blows up when called."""
        def run(**kwargs):
            raise ValueError("post-integration crash")
    ''')
    _seed_cap(tmp_kdb, "cap-stg2", slug)

    result = verify_promoted(
        cap_id="cap-stg2",
        slug=slug,
        capability_type="utility",
        db_path=tmp_kdb,
        isolated_timeout_sec=15,
        orchestrator_timeout_sec=15,
        use_cache=False,
    )
    assert not result.ok
    # A run() that raises is caught by stage 1 (isolated import + call)
    # so we assert quarantine + revert without pinning the exact stage.
    assert result.reverted
    assert result.quarantined
    assert "ValueError" in result.traceback or "post-integration" in result.traceback

    conn = sqlite3.connect(tmp_kdb)
    mode = conn.execute(
        "SELECT runtime_mode FROM capabilities WHERE id = 'cap-stg2'"
    ).fetchone()[0]
    conn.close()
    assert mode == "stub_reverted"


# ── Retry cap ─────────────────────────────────────────────────────────────

def test_verify_promoted_permanent_quarantine_after_max_attempts(
    tmp_kdb: str,
) -> None:
    slug = "__pytest_verifier_retry_cap"
    cap_id = "cap-retry-cap"

    # Seed 2 prior failed verification log rows to simulate attempts 1 & 2
    conn = sqlite3.connect(tmp_kdb)
    ensure_verification_table(conn)
    for _ in range(DEFAULT_MAX_ATTEMPTS - 1):
        conn.execute(
            "INSERT INTO verification_log "
            "(capability_id, slug, stage, ok, reason, traceback, duration_ms) "
            "VALUES (?, ?, 'isolated', 0, 'prior', 'trace', 100)",
            (cap_id, slug),
        )
    conn.commit()
    conn.close()

    _write_live(slug, '''
        """still broken."""
        def run(**kwargs
            pass
    ''')
    _seed_cap(tmp_kdb, cap_id, slug)

    result = verify_promoted(
        cap_id=cap_id, slug=slug, capability_type="utility",
        db_path=tmp_kdb, isolated_timeout_sec=15, use_cache=False,
    )
    assert not result.ok
    assert result.reverted
    assert result.quarantined

    # After max attempts, runtime_mode should be permanently 'quarantined'
    conn = sqlite3.connect(tmp_kdb)
    mode = conn.execute(
        "SELECT runtime_mode FROM capabilities WHERE id = ?", (cap_id,),
    ).fetchone()[0]
    conn.close()
    assert mode == "quarantined", f"expected permanent quarantine, got {mode}"


# ── Retry guidance for codegen ────────────────────────────────────────────

def test_get_retry_guidance_returns_last_traceback(tmp_kdb: str) -> None:
    slug = "__pytest_verifier_guidance"
    _write_live(slug, '''
        """crashes."""
        def run(**kwargs):
            raise RuntimeError("specific failure message here")
    ''')
    _seed_cap(tmp_kdb, "cap-guide", slug)

    verify_promoted(
        cap_id="cap-guide", slug=slug, capability_type="utility",
        db_path=tmp_kdb, isolated_timeout_sec=15, use_cache=False,
    )
    guidance = get_retry_guidance("cap-guide", db_path=tmp_kdb)
    assert len(guidance) == 2
    # First line is the framing hint, second is the traceback
    assert "verification failed" in guidance[0].lower()
    assert "RuntimeError" in guidance[1] or "specific failure" in guidance[1]


def test_get_retry_guidance_empty_when_no_failures(tmp_kdb: str) -> None:
    guidance = get_retry_guidance("unknown-cap", db_path=tmp_kdb)
    assert guidance == []


# ── Status endpoint payload ───────────────────────────────────────────────

def test_verification_status_returns_snapshot(tmp_kdb: str) -> None:
    slug_ok = "__pytest_verifier_status_ok"
    slug_bad = "__pytest_verifier_status_bad"

    _write_live(slug_ok, '''
        """ok."""
        def run(**kwargs):
            return "ok"
    ''')
    _write_live(slug_bad, '''
        """bad."""
        def run(**kwargs
            pass
    ''')
    _seed_cap(tmp_kdb, "cap-st-1", slug_ok)
    _seed_cap(tmp_kdb, "cap-st-2", slug_bad)

    verify_promoted(cap_id="cap-st-1", slug=slug_ok, capability_type="utility",
                    db_path=tmp_kdb, isolated_timeout_sec=15,
                    orchestrator_timeout_sec=15, use_cache=False)
    verify_promoted(cap_id="cap-st-2", slug=slug_bad, capability_type="utility",
                    db_path=tmp_kdb, isolated_timeout_sec=15, use_cache=False)

    snap = verification_status(db_path=tmp_kdb, limit=10)
    assert snap["ok"] is True
    assert snap["totals"]["total"] >= 2
    assert snap["totals"]["successes"] >= 1
    assert snap["totals"]["failures"] >= 1
    assert isinstance(snap["recent"], list)
    assert len(snap["recent"]) >= 2
    assert snap["runtime_mode_counts"]["stub_reverted"] >= 1


# ── Cache behaviour ───────────────────────────────────────────────────────

def test_verify_promoted_cache_hits_on_repeat(tmp_kdb: str) -> None:
    slug = "__pytest_verifier_cache"
    _write_live(slug, '''
        """cache test."""
        def run(**kwargs):
            return 1
    ''')
    _seed_cap(tmp_kdb, "cap-cache", slug)

    first = verify_promoted(
        cap_id="cap-cache", slug=slug, capability_type="utility",
        db_path=tmp_kdb, isolated_timeout_sec=15,
        orchestrator_timeout_sec=15, use_cache=False,
    )
    assert first.ok
    assert first.stage == "orchestrator"

    second = verify_promoted(
        cap_id="cap-cache", slug=slug, capability_type="utility",
        db_path=tmp_kdb, use_cache=True,
    )
    assert second.ok
    assert second.stage == "cached", f"expected cache hit, got {second.stage}"
