#!/bin/bash
# scripts/track_phase1.sh - Core Intelligence Restoration Tracker

echo "📊 PHASE 1: CORE INTELLIGENCE RESTORATION"
echo "=================================="

# 1.1 Fix Daemon (preexec_fn check)
PREEXEC_COUNT=$(grep -c "preexec_fn" scripts/dmai_daemon_fixed.py 2>/dev/null || echo "0")
if [ "$PREEXEC_COUNT" -eq "0" ]; then
    echo "1.1 Daemon Fix: ✅ FIXED"
else
    echo "1.1 Daemon Fix: ❌ $PREEXEC_COUNT occurrences remain"
fi

# 1.2 All Services Running
RUNNING_SERVICES=$(ps aux | grep -E "harvester|book|web|dark|music|voice|dual" | grep -v grep | grep -v "track" | wc -l | tr -d ' ')
echo "1.2 Running Services: $RUNNING_SERVICES/8"

# 1.3 Render Entry Point
if [ -f "render_start.py" ]; then
    echo "1.3 Render Entry: ✅"
else
    echo "1.3 Render Entry: ❌"
fi

# 1.4 Telegram Responsive
if grep -q "TELEGRAM" .env 2>/dev/null; then
    echo "1.4 Telegram: ✅ Ready"
else
    echo "1.4 Telegram: ❌ No token"
fi
