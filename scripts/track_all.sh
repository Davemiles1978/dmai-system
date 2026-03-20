#!/bin/bash
# scripts/track_all.sh - DMAI Master Progress Tracker

echo "📊 DMAI MASTER PROGRESS REPORT"
echo "Generated: $(date)"
echo "========================================"

# Run all phase trackers
for phase in 0 1 2 3; do
    echo ""
    if [ -f "scripts/track_phase${phase}.sh" ]; then
        bash "scripts/track_phase${phase}.sh"
    else
        echo "Phase $phase tracker not found"
    fi
    echo "----------------------------------------"
done

# Save metrics
cat > CURRENT_METRICS.json << EOF
{
  "date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "generation": 72,
  "services_running": $(ps aux | grep -E "harvester|book|web|dark|music|voice|dual" | grep -v grep | grep -v "track" | wc -l | tr -d ' '),
  "evolved_models": $(ls -la agents/evolved/ 2>/dev/null | grep -c "^d" || echo 0),
  "promoted_models": $(ls -la agents/hall_of_fame/ 2>/dev/null | grep -c "^l" || echo 0)
}
