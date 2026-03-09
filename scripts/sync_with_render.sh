#!/bin/bash
# Sync local system with Render services

cd /Users/davidmiles/Desktop/dmai-system
source venv/bin/activate

echo "🔄 Syncing with Render Services - $(date)"
echo "========================================"

# Run one-time sync
python scripts/render_bridge.py

# Show stats
python scripts/render_bridge.py --stats

echo ""
echo "✅ Sync complete"
echo "📁 Check render_discoveries.json for data"
