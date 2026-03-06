#!/bin/bash
# Check status of all DMAI services

cd /Users/davidmiles/Desktop/AI-Evolution-System

echo "📊 DMAI SERVICES STATUS"
echo "======================="
echo ""

check_service() {
    local name=$1
    local pattern=$2
    if pgrep -f "$pattern" > /dev/null; then
        echo "✅ $name: RUNNING (PID: $(pgrep -f "$pattern" | head -1))"
    else
        echo "❌ $name: STOPPED"
    fi
}

check_service "Evolution Engine" "evolution_engine.py"
check_service "Web Researcher" "web_researcher.py"
check_service "Dark Researcher" "dark_researcher.py"
check_service "Book Reader" "book_reader.py"
check_service "Music Learner" "music_learner.py"
check_service "Voice Service" "dmai_voice_with_learning.py"

echo ""
TOTAL=$(pgrep -f "evolution|web|dark|book|music|dmai_voice" | wc -l | tr -d ' ')
echo "Total DMAI processes: $TOTAL"
