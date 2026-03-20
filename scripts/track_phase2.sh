#!/bin/bash
# scripts/track_phase2.sh - Evolution Engine Activation Tracker

echo "📊 PHASE 2: EVOLUTION ENGINE ACTIVATION"
echo "=================================="

# 2.2 Promotion Tracker
if [ -f "evolution/promotion_tracker.py" ]; then
    echo "2.2 Promotion Tracker: ✅"
else
    echo "2.2 Promotion Tracker: ❌"
fi

# 2.4 Cross-Breeding Active
if [ -f "evolution/cross_breeder.py" ]; then
    echo "2.4 Cross-Breeding: ✅ Ready"
else
    echo "2.4 Cross-Breeding: ❌ Missing"
fi

# 2.5 Innovation Filter
if [ -f "evolution/innovation_filter.py" ]; then
    echo "2.5 Innovation Filter: ✅"
else
    echo "2.5 Innovation Filter: ❌"
fi

# 2.8 Models in /evolved
EVOLVED_COUNT=$(ls -la agents/evolved/ 2>/dev/null | grep -c "^d" || echo "0")
echo "2.8 Evolved Models: $EVOLVED_COUNT"

# 2.9 Models Promoted
PROMOTED_COUNT=$(ls -la agents/hall_of_fame/ 2>/dev/null | grep -c "^l" || echo "0")
echo "2.9 Promoted Models: $PROMOTED_COUNT"

# 2.10 Weaknesses
if [ -f "evolution/system_weakness_scanner.py" ]; then
    echo "2.10 Weaknesses: Scanner exists"
else
    echo "2.10 Weaknesses: Scanner missing"
fi
