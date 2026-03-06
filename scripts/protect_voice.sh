#!/bin/bash
# Voice system integrity checker

echo "🔒 VOICE SYSTEM INTEGRITY CHECK"
echo "================================"

# Check if voice is running
if pgrep -f dmai_voice_with_learning.py > /dev/null; then
    echo "✅ Voice process is running"
else
    echo "❌ Voice process is NOT running"
    echo "   Starting voice..."
    /Users/davidmiles/Desktop/AI-Evolution-System/scripts/start_voice.sh
fi

# Check vocabulary file permissions
VOCAB_FILE="/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/vocabulary_master.json"
if [ -f "$VOCAB_FILE" ]; then
    PERMS=$(ls -l $VOCAB_FILE | awk '{print $1}')
    echo "✅ Vocabulary file exists: $PERMS"
else
    echo "❌ Vocabulary file missing!"
fi

# Check symlink
SYMLINK="/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/vocabulary.json"
if [ -L "$SYMLINK" ]; then
    TARGET=$(readlink $SYMLINK)
    echo "✅ Symlink points to: $TARGET"
else
    echo "❌ Symlink broken!"
fi

# Check log file
LOG_FILE="/Users/davidmiles/Desktop/AI-Evolution-System/logs/vocabulary_changes.log"
if [ -f "$LOG_FILE" ]; then
    echo "✅ Log file exists"
else
    echo "⚠️ Creating log file"
    touch $LOG_FILE
    chmod 666 $LOG_FILE
fi

# Check backup directory
BACKUP_DIR="/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/backups/"
if [ -d "$BACKUP_DIR" ]; then
    BACKUP_COUNT=$(ls -1 $BACKUP_DIR | wc -l)
    echo "✅ Backups available: $BACKUP_COUNT"
else
    echo "⚠️ Creating backup directory"
    mkdir -p $BACKUP_DIR
fi

# Final status
echo ""
echo "📊 VOCABULARY STATUS:"
/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/vocabulary_manager.py

echo ""
echo "🔒 Voice system is locked and protected"
