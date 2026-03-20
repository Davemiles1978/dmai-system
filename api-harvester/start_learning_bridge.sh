#!/bin/bash
cd /Users/davidmiles/Desktop/dmai-system
source venv/bin/activate

# Kill any existing bridge
pkill -f harvester_to_dmai_bridge.py

# Start the bridge
nohup python3 api-harvester/harvester_to_dmai_bridge.py > logs/learning_bridge.log 2>&1 &

echo "🚀 DMAI Learning Bridge started with PID $!"
