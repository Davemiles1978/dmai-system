#!/bin/bash
# Check evolution status

source venv/bin/activate

echo "📊 EVOLUTION SYSTEM STATUS"
echo "=========================="

if [ -f evolution.pid ]; then
    PID=$(cat evolution.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Evolution RUNNING (PID: $PID)"
    else
        echo "⚠️  Evolution STOPPED (stale PID file)"
        rm evolution.pid
    fi
else
    echo "⏸️  Evolution STOPPED"
fi

# Show latest logs
if [ -f logs/evolution.log ]; then
    echo -e "\n📝 Last 5 log entries:"
    tail -5 logs/evolution.log
fi

# Show checkpoints
CHECKPOINTS=$(ls -d checkpoints/generation_* 2>/dev/null | wc -l)
echo -e "\n📁 Checkpoints saved: $CHECKPOINTS"

# Show best versions
if [ -d checkpoints/best_versions ]; then
    echo -e "\n🏆 Best versions by repo:"
    for repo in checkpoints/best_versions/*/; do
        if [ -d "$repo" ]; then
            echo "  • $(basename $repo)"
        fi
    done
fi
