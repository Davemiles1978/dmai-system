#!/bin/bash
# scripts/track_phase0.sh - Foundation Phase Tracker

echo "📊 PHASE 0: FOUNDATION"
echo "=================================="

# Core Architecture Docs
CORE_DOCS=$(ls -la *.docx 2>/dev/null | wc -l)
echo "Core Docs: $CORE_DOCS files"

# Evolution Engine Code
EVOLUTION_FILES=$(ls -la evolution/*.py 2>/dev/null | wc -l)
echo "Evolution Files: $EVOLUTION_FILES"

# Voice System Code
VOICE_FILES=$(ls -la voice/*.py 2>/dev/null | wc -l)
echo "Voice Files: $VOICE_FILES"

# Voice Enrollment
if [ -f "voice_models/voice_profile.json" ]; then
    VOICE_USER=$(jq -r .user_id voice_models/voice_profile.json 2>/dev/null || echo "unknown")
    echo "Voice Enrolled: ✅ ($VOICE_USER)"
else
    echo "Voice Enrolled: ❌"
fi

# Telegram Token
TELEGRAM_TOKEN=$(grep -c "TELEGRAM" .env 2>/dev/null || echo "0")
if [ "$TELEGRAM_TOKEN" -gt "0" ] 2>/dev/null; then
    echo "Telegram Token: ✅ (found)"
else
    echo "Telegram Token: ❌"
fi

# API Harvester
if pgrep -f "harvester.py" > /dev/null; then
    echo "API Harvester: ✅ RUNNING"
else
    echo "API Harvester: ❌ STOPPED"
fi
