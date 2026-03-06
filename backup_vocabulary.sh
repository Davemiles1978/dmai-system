#!/bin/bash
SOURCE="/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/vocabulary.json"
BACKUP_DIR="/Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/backups"

mkdir -p "$BACKUP_DIR"

if [ -f "$SOURCE" ]; then
    cp "$SOURCE" "$BACKUP_DIR/vocabulary_$(date +%Y%m%d_%H%M%S).json"
    echo "✅ Vocabulary backed up at $(date)"
    
    # Keep only last 50 backups
    ls -t "$BACKUP_DIR"/vocabulary_*.json | tail -n +51 | xargs rm -f 2>/dev/null
else
    echo "❌ Vocabulary file not found"
fi
