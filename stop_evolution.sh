#!/bin/bash
# Stop the evolution system

if [ -f evolution.pid ]; then
    PID=$(cat evolution.pid)
    echo "🛑 Stopping evolution (PID: $PID)..."
    kill $PID 2>/dev/null
    rm evolution.pid
    echo "✅ Evolution stopped"
else
    echo "❌ No evolution running"
fi
