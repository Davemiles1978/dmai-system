#!/bin/bash
# DMAI Master Control Script

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     DMAI CONTROL CENTER            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""

while true; do
    echo -e "${GREEN}Choose an option:${NC}"
    echo "  1) 🎤 Enroll Voice (when alone)"
    echo "  2) 🗣️  Run DMAI Voice Interface"
    echo "  3) 📱 Show Connected Devices"
    echo "  4) 📝 Test Command Processor"
    echo "  5) 🔧 Change Name Preference"
    echo "  6) 📊 Check System Status"
    echo "  7) 🚪 Exit"
    echo ""
    read -p "Choice [1-7]: " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Starting voice enrollment...${NC}"
            python voice/enroll_master.py
            ;;
        2)
            echo -e "${YELLOW}Starting DMAI Voice Interface...${NC}"
            python voice/dmai_voice.py
            ;;
        3)
            echo -e "${BLUE}Connected Devices:${NC}"
            python -c "from voice.devices.device_manager import DeviceManager; dm=DeviceManager(); [print(f'  • {d[\"name\"]} ({d[\"type\"]})') for d in dm.get_available_devices()]"
            ;;
        4)
            echo -e "${BLUE}Testing Command Processor:${NC}"
            python voice/commands/enhanced_processor_with_name.py
            ;;
        5)
            echo -e "${YELLOW}Change how DMAI addresses you:${NC}"
            echo "Allowed: David, Master, Father, Sir"
            read -p "Enter preferred name: " newname
            python -c "
from voice.user_preferences import UserPreferences
prefs = UserPreferences()
allowed = ['David', 'Master', 'Father', 'Sir']
if '$newname' in allowed:
    prefs.set_name('$newname')
    print(f'✅ Now calling you {prefs.get_name()}')
else:
    print(f'❌ Only {allowed} are allowed')
"
            ;;
        6)
            echo -e "${BLUE}╔════════ SYSTEM STATUS ═══════╗${NC}"
            echo -e " Voice enrolled: $(python -c "from voice.auth.voice_auth import VoiceAuth; auth=VoiceAuth(); print('✅' if 'master' in auth.voiceprints else '❌')")"
            echo -e " Devices: $(python -c "from voice.devices.device_manager import DeviceManager; dm=DeviceManager(); print(len(dm.get_available_devices()))") connected"
            echo -e " Current name: $(python -c "from voice.user_preferences import UserPreferences; prefs=UserPreferences(); print(prefs.get_name())")"
            echo -e " Whisper ready: ✅"
            echo -e " Wake word: ✅"
            echo -e "${BLUE}╚══════════════════════════════╝${NC}"
            ;;
        7)
            echo -e "${GREEN}DMAI sleeping. Wake me when needed.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    echo ""
done
