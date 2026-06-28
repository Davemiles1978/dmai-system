#!/usr/bin/env bash
# DMAI deploy preflight.
#
# Runs the checks that py_compile alone CANNOT catch. The bug that crashed
# production on 2026-06-26 (undefined @require_master_password decorator) would
# have been caught here at check C and check D.
#
# Exit 0  → safe to push.
# Exit 1+ → DO NOT PUSH. Fix the failure first.
#
# Usage:
#   bash scripts/preflight.sh
#
# The pre-push hook (scripts/git-hooks/pre-push) runs this automatically when
# pushing to refs/heads/main. To bypass in a genuine emergency:
#   git push --no-verify

set -uo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

# Auto-activate local venv if it exists and no venv is currently active.
# Render runs in its own env with all deps installed; locally we need the venv
# for runtime imports (flask_cors, etc.) to be available during check B.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    for VENV_PATH in /tmp/dmai_venv .venv venv; do
        if [[ -f "$VENV_PATH/bin/activate" ]]; then
            # shellcheck disable=SC1090,SC1091
            source "$VENV_PATH/bin/activate"
            echo "[info] activated venv at $VENV_PATH"
            break
        fi
    done
fi

PY="${PYTHON:-python3}"
FAILED=0
START_TS=$(date +%s)

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

run_check() {
    local name="$1"; shift
    echo ""
    echo "→ $name"
    if "$@"; then
        return 0
    else
        local rc=$?
        echo "[FAIL] $name (exit $rc)"
        FAILED=$((FAILED + 1))
        return $rc
    fi
}

# ---------------------------------------------------------------------------
# Check A — byte-compile
# ---------------------------------------------------------------------------
section "Check A: byte-compile"
if "$PY" -m py_compile dmai_core_complete.py; then
    echo "[PASS] py_compile dmai_core_complete.py"
else
    echo "[FAIL] py_compile dmai_core_complete.py"
    FAILED=$((FAILED + 1))
fi

if [[ -d components ]]; then
    if "$PY" -m compileall -q -f components/ > /tmp/preflight_compileall.log 2>&1; then
        echo "[PASS] compileall components/ ($(find components -name '*.py' | wc -l | tr -d ' ') files)"
    else
        echo "[FAIL] compileall components/"
        cat /tmp/preflight_compileall.log
        FAILED=$((FAILED + 1))
    fi
fi

# ---------------------------------------------------------------------------
# Check B — full import smoke (THE check that catches today's bug class)
# ---------------------------------------------------------------------------
section "Check B: full import smoke"

export RENDER=false
export DATA_PATH=/tmp/dmai_preflight_data
export MASTER_PASSWORD=dummy
export JWT_SECRET=dummy_jwt_secret_for_preflight_only
mkdir -p "$DATA_PATH"

if timeout 60 "$PY" -c "import dmai_core_complete; print('[PASS] import dmai_core_complete: OK')" 2> /tmp/preflight_import.err; then
    cat /tmp/preflight_import.err 2>/dev/null || true
else
    rc=$?
    if [[ $rc -eq 124 ]]; then
        echo "[FAIL] import dmai_core_complete: TIMEOUT (>60s)"
    else
        echo "[FAIL] import dmai_core_complete: exit $rc"
    fi
    echo "--- traceback ---"
    cat /tmp/preflight_import.err
    echo "-----------------"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# Check C — static decorator scan (AST)
# ---------------------------------------------------------------------------
section "Check C: static decorator scan"
if "$PY" scripts/_check_decorators.py; then
    :
else
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# Check D + E — route registration + accessibility via test client
# ---------------------------------------------------------------------------
# Skip if import failed (no point — would just re-fail with same trace)
if [[ $FAILED -eq 0 ]]; then
    section "Checks D+E: route registration & accessibility"
    if "$PY" scripts/_check_routes.py; then
        :
    else
        FAILED=$((FAILED + 1))
    fi
else
    echo ""
    echo "→ Skipping checks D+E because earlier checks failed"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

echo ""
echo "============================================================"
if [[ $FAILED -eq 0 ]]; then
    echo "  ALL PREFLIGHT CHECKS PASSED ($DURATION s)"
    echo "============================================================"
    exit 0
else
    echo "  PREFLIGHT FAILED: $FAILED check(s) failed ($DURATION s)"
    echo "  DO NOT PUSH. Fix the issues above first."
    echo "  Emergency bypass: git push --no-verify"
    echo "============================================================"
    exit 1
fi
