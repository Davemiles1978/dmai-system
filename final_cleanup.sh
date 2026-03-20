#!/bin/bash
echo "🧹 Final cleanup of documentation and backup files"

OLD_PATH="/Users/davidmiles/Desktop/dmai-system"
NEW_PATH="/Users/davidmiles/Desktop/dmai-system"

# Fix markdown files (they contain commands but don't break functionality)
echo "📝 Fixing markdown files..."
for file in DMAI_MASTER_PLAN_20260306.md TODO.md; do
    if [ -f "$file" ]; then
        sed -i '' "s|$OLD_PATH|$NEW_PATH|g" "$file"
        echo "   ✅ Fixed: $file"
    fi
done

# Fix .command file
echo "🖥️  Fixing command file..."
if [ -f "AI-Evolution-System.command" ]; then
    sed -i '' "s|$OLD_PATH|$NEW_PATH|g" "AI-Evolution-System.command"
    echo "   ✅ Fixed: AI-Evolution-System.command"
fi

# The audit file is a log - we can ignore it or create a note
echo "📋 Note: DMAI_COMPLETE_AUDIT_*.txt contains process info - this is historical data"

# Check for any other text files
echo "🔍 Checking other text files..."
find . -name "*.txt" -o -name "*.md" | while read file; do
    if grep -l "$OLD_PATH" "$file" 2>/dev/null; then
        echo "   ⚠️  Contains old path: $file"
    fi
done

echo ""
echo "✅ Documentation files updated"
echo "⚠️  Note: DMAI_FINAL.zip is a binary file - can't fix automatically"
echo "   This is a backup archive and doesn't affect runtime"
