#!/bin/bash
# Voice system starter - protects against corruption

LOG_FILE="/Users/davidmiles/Desktop/AI-Evolution-System/logs/voice_auto.log"
VENV_PATH="/Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3"
VOICE_SCRIPT="/Users/davidmiles/Desktop/AI-Evolution-System/voice/dmai_voice_with_learning.py"
LOCK_FILE="/Users/davidmiles/Desktop/AI-Evolution-System/voice/voice.lock"

echo "$(date): Starting voice system..." >> $LOG_FILE

# Check if already running
if [ -f $LOCK_FILE ]; then
    PID=$(cat $LOCK_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "$(date): Voice already running with PID $PID" >> $LOG_FILE
        exit 0
    else
        echo "$(date): Stale lock file removed" >> $LOG_FILE
        rm $LOCK_FILE
    fi
fi

# Kill any existing voice processes
pkill -f dmai_voice_with_learning.py >> $LOG_FILE 2>&1
sleep 2

# Start voice
cd /Users/davidmiles/Desktop/AI-Evolution-System
nohup $VENV_PATH $VOICE_SCRIPT >> $LOG_FILE 2>&1 &
VOICE_PID=$!
echo $VOICE_PID > $LOCK_FILE

echo "$(date): Voice started with PID $VOICE_PID" >> $LOG_FILE
echo "Voice started with PID $VOICE_PID"
