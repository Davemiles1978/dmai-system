#!/bin/bash
echo "🎤 DMAI VOICE SYSTEM STATUS"
echo "=========================="

# Check if running
if pgrep -f dmai_voice_with_learning.py > /dev/null; then
    PID=$(pgrep -f dmai_voice_with_learning.py)
    echo "✅ RUNNING (PID: $PID)"
    
    # Get uptime
    ps -o etime= -p $PID | xargs
else
    echo "❌ NOT RUNNING"
fi

echo ""

# Show vocabulary stats
/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/vocabulary_manager.py

echo ""

# Show recent log entries
echo "📝 Recent vocabulary additions:"
tail -5 /Users/davidmiles/Desktop/AI-Evolution-System/logs/vocabulary_changes.log 2>/dev/null || echo "   No recent additions"

echo ""
echo "🔒 Protected files:"
ls -lO /Users/davidmiles/Desktop/AI-Evolution-System/voice/dmai_voice_with_learning.py 2>/dev/null | awk '{print "   Voice script: " $1 " " $10}' || echo "   Voice script: (checking...)"
ls -lO /Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/vocabulary_master.json 2>/dev/null | awk '{print "   Vocabulary: " $1 " " $10}' || echo "   Vocabulary: (checking...)"
