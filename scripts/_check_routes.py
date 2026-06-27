#!/usr/bin/env python3
"""Route registration + accessibility smoke test.

Imports dmai_core_complete (must succeed), then:
  - Asserts Flask url_map has at least MIN_ROUTES rules
  - Uses app.test_client() to hit a fixed set of stable known routes
    and asserts each returns non-404 (auth failures like 401/403 are fine —
    we're only verifying the route is registered).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import dmai_core_complete` works no matter cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_ROUTES = 100
# Routes known to exist on the last-good production HEAD.
# Each tuple is (method, path, expected_non_404_statuses_acceptable=True_means_any_non_404).
STABLE_ROUTES = [
    ("GET", "/health"),
    ("GET", "/api/status"),
    ("GET", "/api/kaizen/repair-stats"),
    ("GET", "/api/research/autonomous/status"),
    ("GET", "/api/ingestor/status"),
]


def main() -> int:
    # Minimal env so the app can import without real secrets / persistent DB
    os.environ.setdefault("RENDER", "false")
    os.environ.setdefault("DATA_PATH", "/tmp/dmai_preflight_data")
    os.environ.setdefault("MASTER_PASSWORD", "dummy")
    os.environ.setdefault("JWT_SECRET", "dummy_jwt_secret_for_preflight_only")
    os.makedirs(os.environ["DATA_PATH"], exist_ok=True)

    try:
        import dmai_core_complete  # type: ignore
    except Exception as e:
        print(f"[FAIL] route smoke: cannot import dmai_core_complete: {type(e).__name__}: {e}")
        return 1

    app = getattr(dmai_core_complete, "app", None)
    if app is None:
        print("[FAIL] route smoke: dmai_core_complete has no 'app' attribute")
        return 1

    n_rules = len(app.url_map._rules)
    if n_rules < MIN_ROUTES:
        print(f"[FAIL] route smoke: only {n_rules} routes registered (expected >= {MIN_ROUTES})")
        return 1
    print(f"[PASS] route registration: {n_rules} routes registered")

    failures = []
    client = app.test_client()
    for method, path in STABLE_ROUTES:
        try:
            resp = client.open(path, method=method)
            if resp.status_code == 404:
                failures.append((method, path, 404, "route not registered"))
            else:
                print(f"[PASS] route accessible: {method} {path} -> {resp.status_code}")
        except Exception as e:
            failures.append((method, path, "ERR", f"{type(e).__name__}: {e}"))

    if failures:
        for method, path, code, note in failures:
            print(f"[FAIL] route accessible: {method} {path} -> {code} ({note})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
