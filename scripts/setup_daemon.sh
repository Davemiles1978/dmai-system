#!/bin/bash
# Complete DMAI Daemon Setup - Run this once

echo "🚀 Setting up DMAI 24/7 Daemon System"
echo "====================================="

cd /Users/davidmiles/Desktop/dmai-system

# Step 1: Stop any running services
echo "🛑 Stopping any running services..."
./scripts/stop_all.sh 2>/dev/null
pkill -f "dmai|evolution|web|dark|book|music|voice"

# Step 2: Unload all old launchd services
echo "📦 Unloading old launchd services..."
launchctl unload ~/Library/LaunchAgents/com.dmai.*.plist 2>/dev/null

# Step 3: Make all scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py 2>/dev/null

# Step 4: Ensure dependencies are installed
echo "📦 Checking dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Step 5: Create the launchd service
echo "⚙️ Creating launchd service..."
cat > ~/Library/LaunchAgents/com.dmai.core.plist << 'INNER'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dmai.core</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/davidmiles/Desktop/dmai-system/scripts/start_all_daemons.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/davidmiles/Desktop/dmai-system/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/davidmiles/Desktop/dmai-system/logs/launchd.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>/Users/davidmiles/Desktop/dmai-system</string>
</dict>
</plist>
INNER

# Step 6: Load and start the service
echo "🚀 Loading and starting daemon..."
launchctl load ~/Library/LaunchAgents/com.dmai.core.plist
launchctl start com.dmai.core

# Step 7: Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Step 8: Check status
echo ""
echo "📊 FINAL STATUS:"
./scripts/status.sh

echo ""
echo "✅ Daemon setup complete!"
echo "📝 Services will now run 24/7 and restart automatically"
echo "   - After reboot: They start automatically"
echo "   - If they crash: They restart in 5 seconds"
echo "   - To check status: ./scripts/status.sh"
echo "   - To stop all: ./scripts/stop_all.sh"
echo "   - To view logs: tail -f logs/daemon.log"
