#!/bin/bash
# Start the hybrid evolution system

cd /Users/davidmiles/Desktop/AI-Evolution-System

echo "🚀 DMAI HYBRID EVOLUTION SYSTEM"
echo "================================"

case "$1" in
    once)
        echo "Running one evolution cycle..."
        /Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3 \
            /Users/davidmiles/Desktop/AI-Evolution-System/evolution/orchestrator/hybrid_evolution_orchestrator.py --once
        ;;
    cycles)
        if [ -z "$2" ]; then
            echo "Usage: $0 cycles <number>"
            exit 1
        fi
        echo "Running $2 evolution cycles..."
        /Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3 \
            /Users/davidmiles/Desktop/AI-Evolution-System/evolution/orchestrator/hybrid_evolution_orchestrator.py --cycles "$2"
        ;;
    continuous)
        echo "Running continuous evolution (Ctrl+C to stop)..."
        /Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3 \
            /Users/davidmiles/Desktop/AI-Evolution-System/evolution/orchestrator/hybrid_evolution_orchestrator.py
        ;;
    status)
        echo "📊 Evolution Status"
        /Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3 \
            /Users/davidmiles/Desktop/AI-Evolution-System/evolution/orchestrator/metrics_tracker.py
        ;;
    versions)
        echo "📚 Version History"
        ls -la /Users/davidmiles/Desktop/AI-Evolution-System/evolution/history/versions/
        ;;
    *)
        echo "Usage: $0 {once|cycles <n>|continuous|status|versions}"
        echo ""
        echo "  once        - Run one evolution cycle"
        echo "  cycles <n>  - Run n evolution cycles"
        echo "  continuous  - Run continuously with random timing"
        echo "  status      - Show evolution metrics"
        echo "  versions    - List all saved versions"
        ;;
esac
