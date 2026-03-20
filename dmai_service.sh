#!/bin/bash
# DMAI Background Service Control

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SERVICE_NAME="com.dmai.listener"
PLIST_FILE="$HOME/Library/LaunchAgents/$SERVICE_NAME.plist"

echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     DMAI BACKGROUND SERVICE       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"

case "$1" in
    start)
        echo -e "${YELLOW}Starting DMAI background listener...${NC}"
        launchctl load "$PLIST_FILE"
        launchctl start "$SERVICE_NAME"
        echo -e "${GREEN}✅ DMAI is now listening in the background${NC}"
        ;;
    stop)
        echo -e "${YELLOW}Stopping DMAI background listener...${NC}"
        launchctl stop "$SERVICE_NAME"
        launchctl unload "$PLIST_FILE"
        echo -e "${GREEN}✅ DMAI background listener stopped${NC}"
        ;;
    status)
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo -e "${GREEN}✅ DMAI is running in background${NC}"
            # Show last few log lines
            echo -e "\n${BLUE}Recent activity:${NC}"
            tail -n 5 logs/dmai_background.log 2>/dev/null || echo "No logs yet"
        else
            echo -e "${RED}❌ DMAI is not running${NC}"
        fi
        ;;
    logs)
        echo -e "${BLUE}Showing live logs (Ctrl+C to exit):${NC}"
        tail -f logs/dmai_background.log
        ;;
    test)
        echo -e "${YELLOW}Testing wake word detection...${NC}"
        python -c "from voice.wake.wake_detector import WakeWordDetector; d=WakeWordDetector(); d.initialize(); print('Say something...'); d.start(callback=lambda: print('✅ Wake word detected!'))"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|test}"
        echo ""
        echo "  start   - Start background listener"
        echo "  stop    - Stop background listener"
        echo "  status  - Check if running"
        echo "  logs    - View live logs"
        echo "  test    - Test wake word"
        exit 1
        ;;
esac
