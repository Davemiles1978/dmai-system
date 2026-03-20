#!/bin/bash
cd /Users/davidmiles/Desktop/dmai-system
source venv/bin/activate

echo "🧹 Weekly DMAI Evolution Cleanup - $(date)"
python3 evolution/evolution_cleanup.py --auto
echo "✅ Cleanup complete"
