#!/bin/bash
echo "Testing symlink protection..."
rm /Users/davidmiles/Desktop/dmai-system/language_learning/data/vocabulary.json 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✅ Symlink is now protected!"
else
    echo "❌ Protection failed - symlink was deleted"
fi

# Show the symlink still exists
ls -la /Users/davidmiles/Desktop/dmai-system/language_learning/data/vocabulary.json
