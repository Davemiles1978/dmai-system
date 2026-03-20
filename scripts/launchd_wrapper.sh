#!/bin/bash
# Wrapper script for launchd to properly set up DMAI environment

# Set up environment
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/davidmiles/Desktop/dmai-system/venv/bin:$PATH"
export PYTHONPATH="/Users/davidmiles/Desktop/dmai-system:$PYTHONPATH"
export HOME="/Users/davidmiles"

# Change to the DMAI directory
cd /Users/davidmiles/Desktop/dmai-system || exit 1

# Activate virtual environment (properly)
source /Users/davidmiles/Desktop/dmai-system/venv/bin/activate

# Run the daemon
exec python3 /Users/davidmiles/Desktop/dmai-system/scripts/dmai_daemon.py
