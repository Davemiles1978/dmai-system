#!/bin/bash
# Production daemon starter for DMAI services
# Runs 24/7, survives reboots

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd /Users/davidmiles/Desktop/AI-Evolution-System

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> logs/daemon.log
}

log "🚀 Starting all DMAI services..."

# Function to start a service with restart protection
start_service() {
    local name=$1
    local cmd=$2
    local log_file="logs/${name}.log"
    
    # Check if already running
    if pgrep -f "$cmd" > /dev/null; then
        log "⚠️ $name already running"
        return
    fi
    
    log "Starting $name..."
    
    # Use nohup to detach from terminal
    nohup bash -c "while true; do
        echo \"\$(date) - Starting $name\" >> $log_file
        $cmd >> $log_file 2>&1
        echo \"\$(date) - $name crashed, restarting in 5s...\" >> $log_file
        sleep 5
    done" > /dev/null 2>&1 &
    
    log "✅ $name started with PID $!"
    sleep 2
}

# Start all services
start_service "evolution" "python3 evolution/evolution_engine.py --continuous"
start_service "web_researcher" "python3 services/web_researcher.py --continuous"
start_service "dark_researcher" "python3 services/dark_researcher.py --continuous"
start_service "book_reader" "python3 services/book_reader.py --continuous"
start_service "music_learner" "python3 services/music_learner.py --continuous"
start_service "voice" "python3 voice/dmai_voice_with_learning.py"

sleep 3
TOTAL=$(pgrep -f "evolution|web|dark|book|music|dmai_voice" | wc -l | tr -d ' ')
log "✅ All services started successfully"
log "Total processes: $TOTAL"
echo "✅ All services started - Total processes: $TOTAL"
