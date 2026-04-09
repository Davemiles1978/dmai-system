#!/bin/bash
# DMAI Production Deployment - Dynamic Port Allocation

set -e

echo "🚀 Starting DMAI with dynamic port allocation..."

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Kill existing processes (any port)
pkill -f "python3.*dmai_core" 2>/dev/null || true
pkill -f "gunicorn.*dmai_core" 2>/dev/null || true

# Find available port dynamically
find_free_port() {
    local port=$1
    while true; do
        if ! lsof -i :$port > /dev/null 2>&1; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        if [ $port -gt 5010 ]; then
            echo "5001"  # fallback
            return 0
        fi
    done
}

PORT=$(find_free_port 5001)
echo "📍 Using port: $PORT"

# Install dependencies
pip3 install -r requirements.txt

# Start server with dynamic port
export PORT=$PORT
python3 dmai_core_complete.py &

echo "✅ DMAI running on http://localhost:$PORT"
echo "📊 Status: http://localhost:$PORT/api/status"
echo "💬 Chat: http://localhost:$PORT/chat"
