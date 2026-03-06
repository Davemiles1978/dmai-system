#!/bin/bash
echo "==================================="
echo "DMAI SYSTEM AUDIT - $(date)"
echo "==================================="

# Check evolution directory
echo ""
echo "📁 EVOLUTION DIRECTORY:"
if [ -d "evolution" ]; then
    ls -la evolution/
else
    echo "❌ evolution/ not found"
fi

# Check knowledge directory
echo ""
echo "📁 KNOWLEDGE DIRECTORY:"
if [ -d "knowledge" ]; then
    ls -la knowledge/
else
    echo "❌ knowledge/ not found"
fi

# Check models directory
echo ""
echo "📁 MODELS DIRECTORY:"
if [ -d "models" ]; then
    ls -la models/
else
    echo "❌ models/ not found"
fi

# Check capabilities directory
echo ""
echo "📁 CAPABILITIES DIRECTORY:"
if [ -d "capabilities" ]; then
    ls -la capabilities/
else
    echo "❌ capabilities/ not found"
fi

# Check for duplicate files
echo ""
echo "🔄 CHECKING FOR DUPLICATES:"
find . -name "evolution_engine.py" | xargs ls -la

echo ""
echo "✅ Audit complete"
