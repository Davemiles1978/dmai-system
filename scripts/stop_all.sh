#!/bin/bash
# Stop all DMAI services

cd /Users/davidmiles/Desktop/AI-Evolution-System

echo "🛑 Stopping all DMAI services..."
echo "$(date) - Stopping services" >> logs/daemon.log

# Kill all service processes
pkill -f "evolution_engine.py"
pkill -f "web_researcher.py"
pkill -f "dark_researcher.py"
pkill -f "book_reader.py"
pkill -f "music_learner.py"
pkill -f "dmai_voice_with_learning.py"

sleep 3

# Verify they're stopped
RUNNING=$(pgrep -f "evolution|web|dark|book|music|dmai_voice" | wc -l | tr -d ' ')
if [ "$RUNNING" = "0" ]; then
    echo "✅ All services stopped"
    echo "$(date) - All services stopped" >> logs/daemon.log
else
    echo "⚠️  $RUNNING services still running:"
    pgrep -f "evolution|web|dark|book|music|dmai_voice"
    echo "$(date) - WARNING: $RUNNING services still running" >> logs/daemon.log
fi
