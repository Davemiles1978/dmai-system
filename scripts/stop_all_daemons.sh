#!/bin/bash
# Stop all DMAI services

echo "🛑 Stopping all DMAI services..."

# Kill all service processes
pkill -f "evolution_engine.py"
pkill -f "web_researcher.py"
pkill -f "dark_researcher.py"
pkill -f "book_reader.py"
pkill -f "music_learner.py"
pkill -f "dmai_voice_with_learning.py"

# Kill any wrapper scripts
pkill -f "while true.*start_all_daemons"

sleep 2

# Count remaining
REMAINING=$(pgrep -f "evolution_engine|web_researcher|dark_researcher|book_reader|music_learner|dmai_voice" | wc -l | tr -d ' ')

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All services stopped"
else
    echo "⚠️  $REMAINING services still running. Force killing..."
    pkill -9 -f "evolution_engine|web_researcher|dark_researcher|book_reader|music_learner|dmai_voice"
    echo "✅ Force kill complete"
fi
