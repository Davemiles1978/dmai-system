#!/bin/bash
# Render.com start script for AGI Evolution System
# Repo: https://github.com/Davemiles1978/dmai-system.git

echo "🚀 Starting AGI Evolution System on Render"
echo "=========================================="
echo "📊 System Information:"
echo "  User: Davemiles1978"
echo "  Repo: dmai-system"
echo "  Time: $(date)"
echo "=========================================="

# Set up persistent storage
echo "📁 Setting up persistent storage..."
mkdir -p /var/data/{shared_data,shared_checkpoints,agi}
mkdir -p /var/data/shared_data/agi_evolution/{capabilities,patterns,synthesis,orchestrator_state,evolution_history}
mkdir -p /var/data/agi/{backups,health,models,test_results}

# Create symlinks if they don't exist
echo "🔗 Creating symlinks..."
ln -sfn /var/data/shared_data ./shared_data
ln -sfn /var/data/shared_checkpoints ./shared_checkpoints
ln -sfn /var/data/agi ./agi

# Print system info
echo ""
echo "📊 System Information:"
echo "  Python: $(python --version)"
echo "  Render: $RENDER"
echo "  Generation: ${GENERATION_START:-5}"
echo "  Data path: /var/data"
echo "  Disk space: $(df -h /var/data | tail -1 | awk '{print $4}') free"

# Check if this is first run
if [ ! -f /var/data/deploy_info.json ]; then
    echo ""
    echo "🎉 First time deployment! Starting fresh from Generation ${GENERATION_START:-5}"
else
    echo ""
    echo "🔄 Restarting existing deployment"
    cat /var/data/deploy_info.json
fi

# Start the cloud launcher
echo ""
echo "🎯 Starting AGI Evolution System..."
echo "=========================================="
exec python cloud_launcher.py
