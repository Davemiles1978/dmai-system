#!/bin/bash
# ULTIMATE fix - Force ALL services to use venv Python

VENV_PYTHON="/Users/davidmiles/Desktop/dmai-system/venv/bin/python3"
PROJECT_DIR="/Users/davidmiles/Desktop/dmai-system"

cd "$PROJECT_DIR" || exit 1

echo "🚀 FORCING all services to use: $VENV_PYTHON"
echo "=========================================="

# Kill ALL Python processes (careful!)
pkill -f "python.*web_researcher"
pkill -f "python.*dark_researcher" 
pkill -f "python.*book_reader"
pkill -f "python.*music_learner"
pkill -f "python.*evolution_engine"
pkill -f "python.*dmai_voice"
sleep 2

# Function to start service with explicit venv python
start_service() {
    local name=$1
    local script=$2
    local args=$3
    local log_file="logs/${name}.log"
    
    echo "Starting $name..."
    
    # Use FULL PATH to venv python - no shortcuts!
    nohup bash -c "
        while true; do
            echo \"\$(date) - Starting $name\" >> \"$log_file\"
            $VENV_PYTHON \"$script\" $args >> \"$log_file\" 2>&1
            echo \"\$(date) - $name crashed, restarting in 5s...\" >> \"$log_file\"
            sleep 5
        done
    " > /dev/null 2>&1 &
    
    echo "✅ $name started"
    sleep 1
}

# Start all services
start_service "web_researcher" "services/web_researcher.py" "--continuous"
start_service "dark_researcher" "services/dark_researcher.py" "--continuous"
start_service "book_reader" "services/book_reader.py" "--continuous"
start_service "music_learner" "services/music_learner.py" "--continuous"
start_service "evolution" "evolution/evolution_engine.py" "--continuous"
start_service "voice" "voice/dmai_voice_with_learning.py" ""

sleep 3

echo ""
echo "=========================================="
echo "✅ VERIFYING all services use venv Python:"
echo "=========================================="

# Verify each service
for service in web_researcher dark_researcher book_reader music_learner evolution_engine dmai_voice; do
    PID=$(pgrep -f "$service" | head -1)
    if [ ! -z "$PID" ]; then
        USED_PYTHON=$(ps -p $PID -o command= | head -1 | awk '{print $1}')
        if [[ "$USED_PYTHON" == *"venv/bin/python3"* ]]; then
            echo "✅ $service: Using venv Python"
        else
            echo "❌ $service: Using system Python - killing..."
            kill -9 $PID 2>/dev/null
            # Map service names to correct paths
            case "$service" in
                "evolution_engine") script_path="evolution/evolution_engine.py" ;;
                "web_researcher") script_path="services/web_researcher.py" ;;
                "dark_researcher") script_path="services/dark_researcher.py" ;;
                "book_reader") script_path="services/book_reader.py" ;;
                "music_learner") script_path="services/music_learner.py" ;;
                "dmai_voice") script_path="voice/dmai_voice_with_learning.py" ;;
            esac
            start_service "$service" "$script_path" "--continuous"
        fi
    fi
done

echo ""
echo "📊 Running services:"
ps aux | grep -E "web_researcher|dark_researcher|book_reader|music_learner|evolution_engine|dmai_voice" | grep -v grep
