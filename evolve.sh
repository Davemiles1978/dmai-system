#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

case "$1" in
    now)
        echo -e "${BLUE}Running evolution now...${NC}"
        python ai_core/evolution_scheduler.py --now
        ;;
    status)
        echo -e "${BLUE}Evolution status:${NC}"
        python -c "
import json
try:
    with open('ai_core/evolution/schedule_log.json', 'r') as f:
        s = json.load(f)
    print(f\"Last: {s.get('last_evolution', 'Never')}\")
    print(f\"Next: {s.get('next_evolution', 'Unknown')}\")
    print(f\"Count: {s.get('evolution_count', 0)}\")
except: print('No evolution data yet')
"
        ;;
    history)
        echo -e "${BLUE}Evolution history:${NC}"
        python -c "
import json
try:
    with open('ai_core/evolution/evolution_history.json', 'r') as f:
        h = json.load(f)
    for evo in h[-5:]:  # Last 5
        print(f\"Gen {evo['generation']}: score {evo['score']:.2f}\")
except: print('No history yet')
"
        ;;
    start)
        launchctl load ~/Library/LaunchAgents/com.dmai.evolution.plist
        echo -e "${GREEN}Evolution service started${NC}"
        ;;
    stop)
        launchctl unload ~/Library/LaunchAgents/com.dmai.evolution.plist
        echo -e "${GREEN}Evolution service stopped${NC}"
        ;;
    *)
        echo "Usage: ./evolve.sh {now|status|history|start|stop}"
        ;;
esac
