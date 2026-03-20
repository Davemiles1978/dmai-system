#!/bin/bash
# Force cleanup of all DMAI processes and ports

echo "========================================="
echo "🧹 DMAI Force Cleanup"
echo "========================================="

# Kill all DMAI Python processes
echo "📋 Killing DMAI processes..."
pkill -f "python.*dmai_core_clean.py" 2>/dev/null
pkill -f "python.*dmai_web.py" 2>/dev/null
pkill -f "harvester" 2>/dev/null
pkill -f "github_scraper" 2>/dev/null

# Kill anything on common ports
for port in 5000 5001 5002 5003 5004 5005; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔍 Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null
    fi
done

sleep 2

# Verify cleanup
echo ""
echo "📊 Remaining Python processes:"
ps aux | grep -E "python.*dmai" | grep -v grep || echo "   None - cleanup successful"

echo ""
echo "📊 Port status:"
for port in 5000 5001 5002 5003 5004 5005; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "   Port $port: IN USE"
    else
        echo "   Port $port: free"
    fi
done

echo ""
echo "✅ Cleanup complete"
