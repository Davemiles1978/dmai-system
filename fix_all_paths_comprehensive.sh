#!/bin/bash
echo "🔧 DMAI PATH FIXER - Comprehensive Update"
echo "=========================================="
echo "Fixing all hardcoded paths from /Users/davidmiles/Desktop/dmai-system"
echo "to /Users/davidmiles/Desktop/dmai-system"
echo ""

OLD_PATH="/Users/davidmiles/Desktop/dmai-system"
NEW_PATH="/Users/davidmiles/Desktop/dmai-system"
TOTAL_FIXED=0

# Function to fix files
fix_files() {
    local pattern=$1
    local description=$2
    local count=0
    
    echo "📁 Fixing $description..."
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            sed -i '' "s|$OLD_PATH|$NEW_PATH|g" "$file"
            echo "   ✅ Fixed: $file"
            ((count++))
        fi
    done < <(grep -l "$OLD_PATH" $pattern 2>/dev/null)
    
    echo "   ✓ Fixed $count files"
    echo ""
    TOTAL_FIXED=$((TOTAL_FIXED + count))
}

# Fix Python files
fix_files "*.py" "Python files"

# Fix JSON files (excluding checkpoints for now)
fix_files "*.json" "JSON files"

# Fix shell scripts
fix_files "*.sh" "Shell scripts"

# Fix config files
fix_files "*.cfg" "Config files"
fix_files "*.conf" "Config files"

# Special handling for critical files we know about
echo "📁 Fixing specific critical files..."
for file in \
    "language_learning/data/secure/vocabulary_manager.py" \
    "evolution/metrics_tracker.py" \
    "evolution/config/evolution_config.json" \
    "voice/vocab_helper.py" \
    "security/biometric_auth.py" \
    "scripts/start_voice.sh" \
    "scripts/start_evolution.sh" \
    "scripts/start_all_daemons.sh" \
    "scripts/status.sh" \
    "scripts/voice_status.sh" \
    "scripts/protect_voice.sh" \
    "scripts/backup_vocabulary.sh" \
    "scripts/check_dmai.sh"
do
    if [ -f "$file" ]; then
        sed -i '' "s|$OLD_PATH|$NEW_PATH|g" "$file"
        echo "   ✅ Fixed: $file"
        ((TOTAL_FIXED++))
    fi
done

# Fix symlinks
echo ""
echo "📁 Fixing symlinks..."
if [ -L "language_learning/data/vocabulary.json" ]; then
    rm "language_learning/data/vocabulary.json"
    ln -sf "$NEW_PATH/language_learning/data/secure/vocabulary_master.json" "language_learning/data/vocabulary.json"
    echo "   ✅ Recreated vocabulary symlink"
fi

echo ""
echo "=========================================="
echo "✅ Total files fixed: $TOTAL_FIXED"
echo "=========================================="

# Verify fixes
echo ""
echo "🔍 Verifying fixes..."
REMAINING=$(grep -r "$OLD_PATH" --include="*.py" --include="*.json" --include="*.sh" . 2>/dev/null | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All paths fixed! No remaining references to old path."
else
    echo "⚠️  $REMAINING references still found. Running deep clean..."
    
    # Deep clean - fix everything recursively
    find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.sh" -o -name "*.cfg" -o -name "*.conf" \) -exec sed -i '' "s|$OLD_PATH|$NEW_PATH|g" {} \;
    
    REMAINING=$(grep -r "$OLD_PATH" . 2>/dev/null | wc -l)
    if [ "$REMAINING" -eq 0 ]; then
        echo "✅ Deep clean successful! No references remain."
    else
        echo "❌ Some references remain. Manual check needed:"
        grep -r "$OLD_PATH" . 2>/dev/null | head -10
    fi
fi

echo ""
echo "🎯 Next steps:"
echo "1. Restart all services: ./scripts/stop_all_daemons.sh && ./scripts/start_all_daemons.sh"
echo "2. Check status: ./scripts/status.sh"
echo "3. Monitor logs: tail -f logs/*.log | grep -i error"
