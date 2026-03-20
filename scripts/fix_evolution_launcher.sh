#!/bin/bash
# Fixed evolution service launcher

cd /Users/davidmiles/Desktop/dmai-system
source venv/bin/activate

# Kill any existing evolution processes
pkill -f "evolution_engine.py"

# Start evolution with correct path
python evolution/evolution_engine.py --continuous >> logs/evolution_fixed.log 2>&1 &
echo "Evolution service started with PID: $!"
