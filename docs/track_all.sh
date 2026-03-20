#!/bin/bash
# scripts/track_all.sh
# DMAI Master Progress Tracker
# Run: ./scripts/track_all.sh --update-master

echo "📊 DMAI MASTER PROGRESS REPORT"
echo "Generated: $(date)"
echo "==================================="

# Load environment
source venv/bin/activate 2>/dev/null

# Function to check phase
check_phase() {
    local phase=$1
    local script="scripts/track_phase${phase}.sh"
    if [ -f "$script" ]; then
        bash "$script"
    else
        echo "Phase $phase tracker not found"
    fi
    echo "-----------------------------------"
}

# Run all phase trackers
for phase in 0 1 2 3 4 5 6 7 8 9; do
    check_phase $phase
done

# Calculate overall progress
echo "📈 OVERALL PROGRESS"
total_tasks=$(grep -c "^| [0-9]" DMAI_MASTER_PLAN_v3.0.md)
completed_tasks=$(grep -c "✅" DMAI_MASTER_PLAN_v3.0.md)
partial_tasks=$(grep -c "⚠️" DMAI_MASTER_PLAN_v3.0.md)
not_started=$(grep -c "🔴" DMAI_MASTER_PLAN_v3.0.md)

echo "Total Tasks: $total_tasks"
echo "✅ Complete: $completed_tasks"
echo "⚠️ Partial: $partial_tasks"
echo "🔴 Not Started: $not_started"
echo "Progress: $(( (completed_tasks * 100) / total_tasks ))%"

# Update master document if flag set
if [ "$1" == "--update-master" ]; then
    sed -i '' "s/Overall Progress: [0-9]*%/Overall Progress: $(( (completed_tasks * 100) / total_tasks ))%/" DMAI_MASTER_PLAN_v3.0.md
    echo "✅ Master document updated"
fi
