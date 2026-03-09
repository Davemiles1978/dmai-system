#!/bin/bash
# DMAI Project Status Report
# Run this to get complete current system status

export DMAI_ROOT="/Users/davidmiles/Desktop/dmai-system"
export HARVESTER_ROOT="/Users/davidmiles/Desktop/api-harvester"

cd "$DMAI_ROOT"
source venv/bin/activate 2>/dev/null

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           DMAI SYSTEM - COMPLETE STATUS REPORT              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Generated: $(date)"
echo ""

# Section 1: System Overview
echo "📊 SYSTEM OVERVIEW"
echo "────────────────────────────────────────────────────────────────"
echo "🔹 DMAI Root: $DMAI_ROOT"
echo "🔹 Harvester Root: $HARVESTER_ROOT"
echo "🔹 Python: $(which python3)"
echo "🔹 Virtual Env: $VIRTUAL_ENV"
echo ""

# Section 2: Core Services Status
echo "🚀 CORE SERVICES STATUS"
echo "────────────────────────────────────────────────────────────────"

# Check Evolution Engine
if pgrep -f "evolution_engine.py.*continuous" > /dev/null; then
    echo "✅ Evolution Engine: RUNNING"
    EVO_PID=$(pgrep -f "evolution_engine.py.*continuous" | head -1)
    echo "   └─ PID: $EVO_PID | Uptime: $(ps -o etime= -p $EVO_PID 2>/dev/null | xargs)"
else
    echo "❌ Evolution Engine: STOPPED"
fi

# Check Book Reader
if pgrep -f "book_reader.py.*continuous" > /dev/null; then
    echo "✅ Book Reader: RUNNING"
    BR_PID=$(pgrep -f "book_reader.py.*continuous" | head -1)
    echo "   └─ PID: $BR_PID | Uptime: $(ps -o etime= -p $BR_PID 2>/dev/null | xargs)"
else
    echo "❌ Book Reader: STOPPED"
fi

# Check Web Researcher
if pgrep -f "web_researcher.py.*continuous" > /dev/null; then
    echo "✅ Web Researcher: RUNNING"
else
    echo "❌ Web Researcher: STOPPED"
fi

# Check Dark Researcher
if pgrep -f "dark_researcher.py.*continuous" > /dev/null; then
    echo "✅ Dark Researcher: RUNNING"
else
    echo "❌ Dark Researcher: STOPPED"
fi

# Check Voice Service
if pgrep -f "dmai_voice_with_learning.py.*continuous" > /dev/null; then
    echo "✅ Voice Service: RUNNING"
else
    echo "❌ Voice Service: STOPPED"
fi

# Check API Harvester
if pgrep -f "harvester.py.*daemon" > /dev/null; then
    echo "✅ API Harvester: RUNNING"
    H_PID=$(pgrep -f "harvester.py.*daemon" | head -1)
    echo "   └─ PID: $H_PID | Uptime: $(ps -o etime= -p $H_PID 2>/dev/null | xargs)"
else
    echo "❌ API Harvester: STOPPED"
fi

# Check Harvester API Server
if pgrep -f "api_server.py" > /dev/null; then
    echo "✅ Harvester API: RUNNING (port 8081)"
    if curl -s http://localhost:8081/status >/dev/null 2>&1; then
        echo "   └─ API responding: YES"
    else
        echo "   └─ API responding: NO"
    fi
else
    echo "❌ Harvester API: STOPPED"
fi

echo ""

# Section 3: Evolution Status
echo "🧬 EVOLUTION STATUS"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$DMAI_ROOT/data/evolution/generation.json" ]; then
    GEN=$(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/generation.json'); print(json.load(f).get('generation', 'unknown'))" 2>/dev/null)
    echo "🔹 Current Generation: $GEN"
else
    echo "🔹 Current Generation: Not started"
fi

if [ -f "$DMAI_ROOT/data/evolution/history.json" ]; then
    EVOS=$(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/history.json'); print(len(json.load(f)))" 2>/dev/null)
    echo "🔹 Total Evolutions: $EVOS"
else
    echo "🔹 Total Evolutions: 0"
fi

if [ -f "$DMAI_ROOT/data/evolution/promotions.json" ]; then
    PROMS=$(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/promotions.json'); print(len(json.load(f)))" 2>/dev/null)
    echo "🔹 Promoted Systems: $PROMS"
else
    echo "🔹 Promoted Systems: 0"
fi

echo ""

# Section 4: Vocabulary Status
echo "📚 VOCABULARY STATUS"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$DMAI_ROOT/language_learning/data/secure/vocabulary_master.json" ]; then
    VOCAB_COUNT=$(python3 -c "import json; f=open('$DMAI_ROOT/language_learning/data/secure/vocabulary_master.json'); print(len(json.load(f)))" 2>/dev/null)
    echo "🔹 Vocabulary Size: $VOCAB_COUNT words"
    echo "🔹 Target: 15,000+ words"
    PERCENT=$((VOCAB_COUNT * 100 / 15000))
    echo "🔹 Progress: $PERCENT%"
else
    echo "❌ Vocabulary file not found"
fi
echo ""

# Section 5: AI Systems Status
echo "🤖 AI SYSTEMS"
echo "────────────────────────────────────────────────────────────────"
if [ -d "$DMAI_ROOT/agents" ]; then
    AGENT_COUNT=$(find "$DMAI_ROOT/agents" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | xargs)
    echo "🔹 Evolved Systems: $AGENT_COUNT"
    if [ $AGENT_COUNT -gt 0 ]; then
        echo "   └─ $(ls -1 "$DMAI_ROOT/agents" 2>/dev/null | head -5 | tr '\n' ', ' | sed 's/, $//')"
    fi
else
    echo "🔹 Evolved Systems: 0 (agents directory not created)"
fi

# External systems from evolution data
if [ -f "$DMAI_ROOT/data/evolution/external_systems.json" ]; then
    EXT_COUNT=$(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/external_systems.json'); print(len(json.load(f)))" 2>/dev/null)
    echo "🔹 External Systems: $EXT_COUNT"
else
    echo "🔹 External Systems: 0"
fi

# Wanted systems
if [ -f "$DMAI_ROOT/data/evolution/wanted_systems.json" ]; then
    WANTED_COUNT=$(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/wanted_systems.json'); print(len(json.load(f)))" 2>/dev/null)
    echo "🔹 Wanted Systems: $WANTED_COUNT"
    if [ $WANTED_COUNT -gt 0 ]; then
        echo "   └─ $(python3 -c "import json; f=open('$DMAI_ROOT/data/evolution/wanted_systems.json'); print(', '.join(list(json.load(f).keys())[:5]))" 2>/dev/null)"
    fi
fi
echo ""

# Section 6: Harvester Status
echo "🔑 API HARVESTER STATUS"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$HARVESTER_ROOT/keys/found_keys.json" ]; then
    KEYS_FOUND=$(python3 -c "import json; f=open('$HARVESTER_ROOT/keys/found_keys.json'); data=json.load(f); print(len(data.get('keys', [])))" 2>/dev/null)
    echo "🔹 API Keys Found: $KEYS_FOUND"
else
    echo "🔹 API Keys Found: 0"
fi

if [ -f "$HARVESTER_ROOT/logs/key_requests.log" ]; then
    REQUESTS=$(wc -l < "$HARVESTER_ROOT/logs/key_requests.log" 2>/dev/null | xargs)
    echo "🔹 Key Requests: $REQUESTS"
fi

# Check GitHub token status
if [ -f "$HARVESTER_ROOT/config.json" ]; then
    TOKEN_STATUS=$(python3 -c "import json; f=open('$HARVESTER_ROOT/config.json'); print('✅ Present' if json.load(f).get('github_token') else '❌ Missing')" 2>/dev/null)
    echo "🔹 GitHub Token: $TOKEN_STATUS"
fi
echo ""

# Section 7: Recent Activity
echo "📝 RECENT ACTIVITY (last 10 log entries)"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$DMAI_ROOT/logs/evolution.log" ]; then
    echo "🔸 Evolution Log:"
    tail -5 "$DMAI_ROOT/logs/evolution.log" 2>/dev/null | sed 's/^/   /'
fi
echo ""
if [ -f "$HARVESTER_ROOT/harvester.log" ]; then
    echo "🔸 Harvester Log:"
    tail -3 "$HARVESTER_ROOT/harvester.log" 2>/dev/null | sed 's/^/   /'
fi
echo ""

# Section 8: System Health
echo "💻 SYSTEM HEALTH"
echo "────────────────────────────────────────────────────────────────"
echo "🔹 CPU Usage: $(ps -A -o %cpu | awk '{s+=$1} END {print s "%"}')"
echo "🔹 Memory Usage: $(vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages free:\s+(\d+)/ and $free=$1; END {printf "%.2f GB\n", $free * $size / 1073741824;}') free"
echo "🔹 Disk Usage: $(df -h . | awk 'NR==2 {print $5}') used"
echo ""

# Section 9: Next Steps
echo "🎯 CURRENT FOCUS / NEXT STEPS"
echo "────────────────────────────────────────────────────────────────"
echo "1. Connect Evolution Engine to Harvester API"
echo "2. Implement API Key verification UI"
echo "3. Create auto-insertion for verified keys"
echo "4. Start continuous evolution with acquired systems"
echo "5. Set up unified daemon for 24/7 operation"
echo ""

echo "╚══════════════════════════════════════════════════════════════╝"
