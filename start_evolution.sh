#!/bin/bash
# Start the evolution system

source venv/bin/activate

if [ -f evolution.pid ]; then
    echo "❌ Evolution already running with PID: $(cat evolution.pid)"
    echo "Run ./stop_evolution.sh first"
    exit 1
fi

echo "🚀 Starting AI Evolution System..."
nohup ./run_247.sh > /dev/null 2>&1 &
echo $! > evolution.pid
echo "✅ Evolution started with PID: $(cat evolution.pid)"
echo "📝 View logs: tail -f logs/evolution.log"
