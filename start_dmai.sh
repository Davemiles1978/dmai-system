#!/bin/bash
# DMAI Startup Script - With Automatic Port Conflict Resolution

echo "========================================="
echo "🚀 DMAI Startup Script"
echo "========================================="

# Configuration
DESIRED_PORT=5001
MAX_PORT_ATTEMPTS=10

# Function to kill process using a specific port
kill_process_on_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "🔍 Found process $pid using port $port - killing it..."
        kill -9 $pid 2>/dev/null
        sleep 1
        # Verify it's gone
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "⚠️  Process still running, trying again..."
            kill -9 $pid 2>/dev/null
            sleep 1
        fi
        echo "✅ Port $port freed"
    fi
}

# Function to find an available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    local attempts=0
    
    while [ $attempts -lt $MAX_PORT_ATTEMPTS ]; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        attempts=$((attempts + 1))
    done
    
    echo "❌ Could not find available port after $MAX_PORT_ATTEMPTS attempts"
    return 1
}

# Kill any existing DMAI processes
echo "📋 Cleaning up existing DMAI processes..."
pkill -f "python.*dmai_core_clean.py" 2>/dev/null
pkill -f "python.*dmai_web.py" 2>/dev/null
pkill -f "harvester" 2>/dev/null
pkill -f "github_scraper" 2>/dev/null
sleep 2

# Kill any process on the desired port
kill_process_on_port $DESIRED_PORT

# Find an available port
PORT=$(find_available_port $DESIRED_PORT)
if [ $? -ne 0 ]; then
    echo "❌ Failed to find available port"
    exit 1
fi

echo "✅ Using port $PORT"

# Kill any process that might have taken this port during our checks
kill_process_on_port $PORT

# Start core
echo "📡 Starting DMAI Core..."
python3 dmai_core_clean.py > core.log 2>&1 &
CORE_PID=$!
echo "   Core PID: $CORE_PID"

# Wait for core to initialize
sleep 3

# Check if core is still running
if ! ps -p $CORE_PID > /dev/null; then
    echo "❌ Core failed to start - check core.log"
    exit 1
fi

# Start web with the selected port
echo "🌐 Starting DMAI Web on port $PORT..."
export PORT=$PORT
export DMAI_PORT=$PORT  # Set both for compatibility
python3 dmai_web.py > web.log 2>&1 &
WEB_PID=$!
echo "   Web PID: $WEB_PID"

# Wait a moment for web to start
sleep 2

echo ""
echo "========================================="
echo "✅ DMAI Running"
echo "========================================="
echo "📊 Core PID: $CORE_PID"
echo "🌐 Web PID: $WEB_PID"
echo "🔌 Web Port: $PORT"
echo "📝 Logs: tail -f core.log web.log"
echo "🌍 URL: http://localhost:$PORT"
echo "========================================="

# Save port to file for reference
echo $PORT > .dmai_port

# Final status check
if ps -p $CORE_PID > /dev/null; then
    echo "✅ Core is running"
else
    echo "❌ Core failed - check core.log"
fi

if ps -p $WEB_PID > /dev/null; then
    echo "✅ Web is running on http://localhost:$PORT"
    echo "   Admin login: http://localhost:$PORT/admin"
    
    # Quick health check
    sleep 1
    curl -s http://localhost:$PORT/health > /dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Health check passed"
    else
        echo "⚠️  Health check failed - web might still be starting"
    fi
else
    echo "❌ Web failed to start - check web.log"
    tail -5 web.log
fi
