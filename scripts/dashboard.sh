#!/bin/bash
# Clean DMAI Dashboard

cd /Users/davidmiles/Desktop/AI-Evolution-System

while true; do 
    clear
    echo "📊 DMAI DASHBOARD - $(date)"
    echo "================================"
    
    # Get evolution generation
    GEN=$(grep "EVOLUTION CYCLE" logs/evolution.log 2>/dev/null | tail -1 | awk '{print $5}')
    if [ -z "$GEN" ]; then
        GEN="Waiting for first cycle"
    fi
    echo "Evolution Gen: $GEN"
    
    # Get vocabulary
    VOCAB=$(grep "vocab:" logs/evolution.log 2>/dev/null | tail -1 | awk '{print $NF}' | tr -d ')')
    if [ -z "$VOCAB" ]; then
        VOCAB="Learning..."
    fi
    echo "Vocabulary: $VOCAB words"
    
    echo ""
    echo "Running Services:"
    echo "----------------"
    
    # Check each service specifically
    pgrep -f "evolution_engine.py" > /dev/null && echo "  ✅ Evolution Engine" || echo "  ❌ Evolution Engine"
    pgrep -f "web_researcher.py" > /dev/null && echo "  ✅ Web Researcher" || echo "  ❌ Web Researcher"
    pgrep -f "dark_researcher.py" > /dev/null && echo "  ✅ Dark Researcher" || echo "  ❌ Dark Researcher"
    pgrep -f "book_reader.py" > /dev/null && echo "  ✅ Book Reader" || echo "  ❌ Book Reader"
    pgrep -f "music_learner.py" > /dev/null && echo "  ✅ Music Learner" || echo "  ❌ Music Learner"
    pgrep -f "dmai_voice_with_learning.py" > /dev/null && echo "  ✅ Voice Service" || echo "  ❌ Voice Service"
    
    echo ""
    echo "Total active: $(pgrep -f "evolution|web|dark|book|music|dmai_voice" | wc -l) processes"
    
    # Show last log entries
    echo ""
    echo "Recent Activity:"
    echo "----------------"
    tail -3 logs/evolution.log 2>/dev/null | sed 's/^.*EVOLUTION - //' | tail -1
    
    sleep 5
done
