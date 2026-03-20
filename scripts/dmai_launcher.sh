#!/bin/bash
# DMAI Launcher Script - Starts services with proper environment

# Get absolute path to DMAI root
DMAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DMAI_ROOT"

# Activate virtual environment
source "$DMAI_ROOT/venv/bin/activate"

# Set Python path
export PYTHONPATH="$DMAI_ROOT:$PYTHONPATH"

# Run the specified script with all arguments
python3 "$@"
