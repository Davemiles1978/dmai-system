#!/bin/bash
echo "=========================================="
echo "DMAI SYSTEM STATUS CAPTURE - $(date)"
echo "=========================================="
echo ""

echo "=== RUNNING PROCESSES ==="
ps aux | grep -E "dmai_core|services/|voice/|evolution" | grep -v grep | grep -v status_capture

echo ""
echo "=== VOCABULARY STATUS ==="
ls -la /Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/vocabulary.json
ls -la /Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/phrases.json
ls -la /Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/secure/vocabulary_master.json

echo ""
echo "=== RECENT WEB RESEARCHER LOGS ==="
tail -20 /Users/davidmiles/Desktop/AI-Evolution-System/logs/web_researcher.log 2>/dev/null

echo ""
echo "=== RECENT VOICE LOGS ==="
tail -20 /Users/davidmiles/Desktop/AI-Evolution-System/logs/voice.log 2>/dev/null
tail -20 /Users/davidmiles/Desktop/AI-Evolution-System/logs/voice_error.log 2>/dev/null

echo ""
echo "=== RECENT EVOLUTION LOGS ==="
tail -20 /Users/davidmiles/Desktop/AI-Evolution-System/logs/evolution.log 2>/dev/null | grep -E "No improvements|Error|WARNING"

echo ""
echo "=== WAKE WORD STATUS ==="
ls -la /Users/davidmiles/Desktop/AI-Evolution-System/voice/wake/keywords/

echo ""
echo "=== CRONTAB STATUS ==="
crontab -l 2>/dev/null

echo ""
echo "=== VOCABULARY SIZE ==="
/Users/davidmiles/Desktop/AI-Evolution-System/venv/bin/python3 -c "
import json
try:
    with open('/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/vocabulary.json', 'r') as f:
        vocab = json.load(f)
        print(f'Current vocabulary: {len(vocab)} words')
except Exception as e:
    print(f'Error reading vocabulary: {e}')
" 2>/dev/null

echo ""
echo "=========================================="
echo "CAPTURE COMPLETE - Copy all output above"
echo "=========================================="
