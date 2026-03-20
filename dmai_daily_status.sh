#!/bin/bash
echo "==================================="
echo "📊 DMAI DAILY STATUS REPORT"
echo "==================================="
echo "Date: $(date)"
echo ""

# Get cloud status
CLOUD_STATUS=$(curl -s https://dmai-final.onrender.com/)
echo "☁️  Cloud DMAI:"
echo "   $CLOUD_STATUS"
echo ""

# Check local voice enrollment
if [ -f "voice/enrollment_data/master_voiceprint.pkl" ]; then
    echo "🎤 Voice: ✅ Enrolled"
else
    echo "🎤 Voice: ❌ Not enrolled"
fi

# Check last evolution locally
if [ -f "ai_core/evolution/current_generation.txt" ]; then
    GEN=$(cat ai_core/evolution/current_generation.txt)
    echo "🧬 Local Generation: $GEN"
fi

echo ""
echo "==================================="

# Language Learning Stats
if [ -f "language_learning/data/stats.json" ]; then
    echo ""
    echo "📚 LANGUAGE LEARNING:"
    python -c "
import json
try:
    with open('language_learning/data/stats.json', 'r') as f:
        s = json.load(f)
    print(f'   Vocabulary: {s.get(\"unique_words\", 0)} words')
    print(f'   Phrases heard: {s.get(\"english_phrases\", 0)}')
    print(f'   Non-English rejected: {s.get(\"non_english_phrases\", 0)}')
except:
    print('   No language data yet')
"
fi
