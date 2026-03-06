#!/bin/bash
echo "Testing symlink protection..."
rm /Users/davidmiles/Desktop/AI-Evolution-System/language_learning/data/vocabulary.json 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✅ Symlink is protected!"
else
    echo "❌ Symlink was deleted!"
fi
