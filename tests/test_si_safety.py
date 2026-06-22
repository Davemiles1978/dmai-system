"""
UT-4.x Self-Improvement Safety Tests
======================================
Tests KPI authenticity gate, atomic writes, regression detection,
no-auto-retrain guarantee, and state corruption handling.
All tests run without a live AI provider.
"""

import json
import os
import sys
import tempfile
import time
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "components"))

# We import from the patched si_core in fixes/
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from si_core_patched import SICore, _atomic_write_json
except ImportError:
    # Fallback to original if patched not available
    from si_core import SICore
    from si_core_patched import _atomic_write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_si(tmp_path=None):
    """Create a SICore instance pointing at a temp data directory."""
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    return SICore(data_path=tmp_path), tmp_path


def make_valid_token(source="training_session"):
    """Create a valid response token."""
    return SICore.make_token(source, session_id="test-session-001")


def make_expired_token():
    """Create a token that is 10 minutes old (expired)."""
    return {
        "source": "training_session",
        "session_id": "test-session-old",
        "timestamp": (datetime.utcnow() - timedelta(minutes=10)).isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# UT-4.1: No phantom KPI increments
# ---------------------------------------------------------------------------

def test_kpi_rejected_without_token():
    """KPI update must be rejected when no token is provided."""
    si, _ = make_temp_si()
    before = si.current_kpis["skill_acquisition_rate"]
    si.update_kpi_1_skill_acquisition(5.0, response_token=None)
    after = si.current_kpis["skill_acquisition_rate"]
    assert after == before, "KPI was updated without a valid token"


def test_kpi_accepted_with_valid_token():
    """KPI update must succeed with a valid token."""
    si, _ = make_temp_si()
    token = make_valid_token()
    si.update_kpi_1_skill_acquisition(3.0, response_token=token)
    assert si.current_kpis["skill_acquisition_rate"] == 3.0


def test_kpi_rejected_with_expired_token():
    """KPI update must be rejected with an expired token (>300s old)."""
    si, _ = make_temp_si()
    before = si.current_kpis["skill_acquisition_rate"]
    token = make_expired_token()
    si.update_kpi_1_skill_acquisition(9.9, response_token=token)
    assert si.current_kpis["skill_acquisition_rate"] == before


def test_kpi_rejected_with_invalid_source():
    """KPI update must be rejected when source is not in allowed list."""
    si, _ = make_temp_si()
    before = si.current_kpis["skill_acquisition_rate"]
    bad_token = {"source": "unknown_source", "session_id": "x", "timestamp": datetime.utcnow().isoformat() + "Z"}
    si.update_kpi_1_skill_acquisition(7.0, response_token=bad_token)
    assert si.current_kpis["skill_acquisition_rate"] == before


@pytest.mark.parametrize("run", range(10))
def test_zero_phantom_increments_10_runs(run):
    """Ten consecutive runs without tokens must produce zero KPI changes."""
    si, _ = make_temp_si()
    before = dict(si.current_kpis)
    for _ in range(5):
        si.update_kpi_1_skill_acquisition(99.0, response_token=None)
        si.update_kpi_3_zero_shot(True, response_token=None)
    for key in ["skill_acquisition_rate", "zero_shot_success_count"]:
        assert si.current_kpis[key] == before[key], \
            f"Phantom increment on {key} in run {run}"


# ---------------------------------------------------------------------------
# UT-4.2: Atomic write verification
# ---------------------------------------------------------------------------

def test_atomic_write_creates_file():
    """_atomic_write_json must create the target file."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json"
        _atomic_write_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["key"] == "value"


def test_atomic_write_correct_content():
    """Atomically written file must contain exactly the provided data."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "data.json"
        payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        _atomic_write_json(path, payload)
        assert json.loads(path.read_text()) == payload


def test_atomic_write_uses_replace(tmp_path):
    """_atomic_write_json must use os.replace (atomic rename) not direct write."""
    replace_called = []
    real_replace = os.replace

    def mock_replace(src, dst):
        replace_called.append((src, dst))
        real_replace(src, dst)

    path = tmp_path / "state.json"
    with patch("os.replace", side_effect=mock_replace):
        _atomic_write_json(path, {"test": True})

    assert len(replace_called) == 1, "os.replace was not called"
    assert str(replace_called[0][1]) == str(path)


def test_atomic_write_no_partial_file_on_empty_data():
    """Writing empty dict atomically must produce a valid empty-object JSON file."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "empty.json"
        _atomic_write_json(path, {})
        assert json.loads(path.read_text()) == {}


# ---------------------------------------------------------------------------
# UT-4.3: Regression alert accuracy
# ---------------------------------------------------------------------------

REGRESSION_CASES = [
    # (baseline, new_value, expect_alert, expected_severity)
    (1.0,  0.80, True,  "HIGH"),      # 20% drop
    (1.0,  0.60, True,  "CRITICAL"),  # 40% drop
    (1.0,  0.95, False, None),        # 5% drop — no alert
    (1.0,  1.00, False, None),        # no change
    (1.0,  1.10, False, None),        # improvement — no alert
    (0.5,  0.40, True,  "HIGH"),      # 20% drop from 0.5
    (0.5,  0.30, True,  "CRITICAL"),  # 40% drop from 0.5
    (100,  85,   True,  "HIGH"),      # 15% drop (exact threshold)
    (100,  86,   False, None),        # 14% drop — just under threshold
    (100,  69,   True,  "CRITICAL"),  # 31% drop
]


@pytest.mark.parametrize("baseline,new_value,expect_alert,expected_severity", REGRESSION_CASES)
def test_regression_detection(baseline, new_value, expect_alert, expected_severity):
    """Regression detection must fire at >=15% drop with correct severity labels."""
    si, tmp = make_temp_si()
    # Write a fake baseline
    baseline_data = {"skill_acquisition_rate": baseline}
    si.save_benchmark_baseline(baseline_data)

    alert = si.check_regression("skill_acquisition_rate", new_value)

    if expect_alert:
        assert alert is not None, f"Expected alert for {baseline}->{new_value} drop"
        assert alert["severity"] == expected_severity, \
            f"Wrong severity: got {alert['severity']}, expected {expected_severity}"
    else:
        assert alert is None, f"Unexpected alert for {baseline}->{new_value}"


@pytest.mark.parametrize("run", range(5))
def test_auto_retraining_never_triggered(run):
    """Regression alerts must always have auto_retraining_triggered=False."""
    si, _ = make_temp_si()
    si.save_benchmark_baseline({"skill_acquisition_rate": 1.0})
    alert = si.check_regression("skill_acquisition_rate", 0.5)
    if alert:
        assert alert["auto_retraining_triggered"] is False
        assert alert["requires_human_review"] is True


# ---------------------------------------------------------------------------
# UT-4.4: No auto-retrain without approval
# ---------------------------------------------------------------------------

def test_no_auto_retrain_method():
    """SICore must not have auto_retrain, auto_apply, or auto_improve methods."""
    si, _ = make_temp_si()
    forbidden = ["auto_retrain", "auto_apply", "auto_improve", "apply_without_approval"]
    for name in forbidden:
        assert not hasattr(si, name), f"SICore has forbidden method: {name}"


def test_no_auto_retrain_attribute():
    """SICore must not have auto_retraining_triggered=True attribute."""
    si, _ = make_temp_si()
    val = getattr(si, "auto_retraining_triggered", False)
    assert val is False


def test_regression_alert_contains_human_review_flag():
    """Every regression alert must contain requires_human_review=True."""
    si, _ = make_temp_si()
    si.save_benchmark_baseline({"skill_acquisition_rate": 1.0})
    alert = si.check_regression("skill_acquisition_rate", 0.7)
    assert alert is not None
    assert alert.get("requires_human_review") is True


def test_kpi_write_never_self_triggers_retrain():
    """Writing a KPI must not set auto_retraining_triggered to True."""
    si, _ = make_temp_si()
    token = make_valid_token()
    si.update_kpi_1_skill_acquisition(1.0, response_token=token)
    assert getattr(si, "auto_retraining_triggered", False) is False


def test_check_regression_returns_dict_not_action():
    """check_regression must return a plain dict, not trigger any action."""
    si, _ = make_temp_si()
    si.save_benchmark_baseline({"skill_acquisition_rate": 1.0})
    result = si.check_regression("skill_acquisition_rate", 0.5)
    assert isinstance(result, dict)
    # Must NOT contain any action key
    assert "action" not in result
    assert "execute" not in result


# ---------------------------------------------------------------------------
# UT-4.5: State corruption detection
# ---------------------------------------------------------------------------

def test_corrupt_state_not_silently_overwritten():
    """SICore must not silently overwrite a corrupt state file with bad data."""
    with tempfile.TemporaryDirectory() as td:
        state_path = Path(td) / "si_state.json"
        corrupt_content = "{invalid json <<<"
        state_path.write_text(corrupt_content)

        # Init SICore pointing at this directory
        si = SICore(data_path=Path(td))

        # File must still contain the original corrupt content
        assert state_path.read_text() == corrupt_content, \
            "Corrupt state file was silently overwritten"


def test_corrupt_state_handled_gracefully():
    """SICore must initialise without crashing even with corrupt state file."""
    with tempfile.TemporaryDirectory() as td:
        state_path = Path(td) / "si_state.json"
        state_path.write_text("NOT JSON AT ALL !!!!")
        try:
            si = SICore(data_path=Path(td))
            # Must have initialised with defaults
            assert si.current_kpis is not None
        except Exception as e:
            pytest.fail(f"SICore crashed on corrupt state: {e}")


def test_missing_state_file_handled():
    """SICore must initialise cleanly when no state file exists."""
    with tempfile.TemporaryDirectory() as td:
        si = SICore(data_path=Path(td))
        assert si.current_kpis["skill_acquisition_rate"] == 0.0


def test_partial_state_file_handled():
    """SICore must not crash on partially written (truncated) state file."""
    with tempfile.TemporaryDirectory() as td:
        state_path = Path(td) / "si_state.json"
        # Simulate truncated write
        state_path.write_text('{"skill_acquisition_rate": 0.5, "transfer_l')
        try:
            si = SICore(data_path=Path(td))
            assert si is not None
        except Exception as e:
            pytest.fail(f"SICore crashed on truncated state: {e}")


def test_valid_state_file_loaded_correctly():
    """SICore must load a valid state file and restore KPI values."""
    with tempfile.TemporaryDirectory() as td:
        si1, _ = make_temp_si(Path(td))
        token = make_valid_token()
        si1.update_kpi_1_skill_acquisition(2.5, response_token=token)
        si1.save_state()

        si2 = SICore(data_path=Path(td))
        assert si2.current_kpis["skill_acquisition_rate"] == 2.5
