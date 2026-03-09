#!/bin/bash
# Simple wrapper for launchd

# Change to the DMAI directory
cd /Users/davidmiles/Desktop/dmai-system || exit 1

# Set Python path explicitly
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/davidmiles/Desktop/dmai-system/venv/bin:$PATH"
export PYTHONPATH="/Users/davidmiles/Desktop/dmai-system:$PYTHONPATH"
export HOME="/Users/davidmiles"

# Run the daemon with full python path
/Users/davidmiles/Desktop/dmai-system/venv/bin/python3 /Users/davidmiles/Desktop/dmai-system/scripts/dmai_daemon_fixed.py
