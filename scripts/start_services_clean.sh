#!/bin/bash
# Clean service starter for DMAI system

cd /Users/davidmiles/Desktop/dmai-system || exit 1
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🛑 Killing existing services..."
pkill -f "python.*(web_researcher|dark_researcher|book_reader|music_learner|evolution_engine|dmai_voice)"
pkill -f "bash -c.*while true"
sleep 2

echo "🚀 Starting DMAI Services - $(date)"
echo "=================================="

# Function to start a service
start_service() {
    local name=$1
    local script=$2
    local log="logs/${name}.log"
    
    echo "Starting $name from $script..."
    
    # Check if script exists
    if [ ! -f "$script" ]; then
        echo "❌ ERROR: Script not found: $script"
        return 1
    fi
    
    # Start process
    python "$script" --continuous >> "$log" 2>&1 &
    local pid=$!
    echo "✅ $name started with PID: $pid"
    sleep 1
}

# Start evolution service (single instance)
if [ -f "evolution/evolution_engine.py" ]; then
    start_service "evolution" "evolution/evolution_engine.py"
else
    echo "⚠️  evolution_engine.py not found"
fi

# Start voice service
if [ -f "voice/dmai_voice_with_learning.py" ]; then
    start_service "voice" "voice/dmai_voice_with_learning.py"
else
    echo "⚠️  voice service not found"
fi

# Start other services
for service in web_researcher dark_researcher book_reader music_learner; do
    if [ -f "services/${service}.py" ]; then
        start_service "$service" "services/${service}.py"
    fi
done

echo "=================================="
echo "📊 Running services:"
ps aux | grep -E "python.*(evolution|voice|researcher|reader|learner)" | grep -v grep

echo ""
echo "📝 Monitor logs with:"
echo "tail -f logs/evolution.log logs/voice.log"
echo "=================================="

# Check knowledge graph health using class
echo ""
echo "🔍 Checking Knowledge Graph health:"
if [ -f "knowledge_graph.py" ]; then
    python -c "
import sys
sys.path.append('.')
try:
    from knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    if kg.is_healthy():
        print('✅ Knowledge Graph is healthy')
    else:
        print('❌ Knowledge Graph is not healthy')
except Exception as e:
    print(f'❌ Error checking Knowledge Graph: {e}')
"
else
    echo "⚠️  knowledge_graph.py not found"
fi
