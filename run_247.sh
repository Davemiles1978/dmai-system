#!/bin/bash
# 24/7 Evolution Runner
# Runs evolution continuously in the background

echo "🚀 Starting 24/7 AI Evolution System"
echo "===================================="

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Function to run evolution
run_evolution() {
    while true; do
        echo "$(date): Starting evolution cycle" >> logs/evolution.log
        python evolution_engine.py cycle >> logs/evolution.log 2>&1
        
        # Wait 1 hour between cycles
        echo "$(date): Cycle complete. Waiting 1 hour..." >> logs/evolution.log
        sleep 3600
    done
}

# Start evolution in background
run_evolution &

# Save PID
echo $! > evolution.pid
echo "✅ Evolution running with PID: $(cat evolution.pid)"
echo "📝 Logs: tail -f logs/evolution.log"

# Keep script running
wait
