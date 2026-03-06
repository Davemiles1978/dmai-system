#!/bin/bash
# Promote a merged "super evolved" version to become the primary evaluator

cd /Users/davidmiles/Desktop/AI-Evolution-System

echo "🚀 DMAI VERSION PROMOTION TOOL"
echo "==============================="
echo ""

# List available merged versions
echo "📋 Available super evolved versions:"
ls -la evolution/merged_versions/ | grep -v total | nl

echo ""
read -p "Enter the number to promote (or 0 to cancel): " choice

if [ "$choice" -eq 0 ]; then
    echo "Cancelled"
    exit 0
fi

# Get the selected file
SELECTED=$(ls -1 evolution/merged_versions/ | sed -n "${choice}p")

if [ -z "$SELECTED" ]; then
    echo "Invalid selection"
    exit 1
fi

MERGED_PATH="evolution/merged_versions/$SELECTED"
echo ""
echo "Selected: $SELECTED"

# Extract evaluator name from filename
EVALUATOR=$(echo $SELECTED | cut -d'_' -f1)
echo "Evaluator: $EVALUATOR"

# Backup current evaluator
BACKUP_PATH="evolution/evaluators/${EVALUATOR}_evaluator.py.backup_$(date +%Y%m%d_%H%M%S)"
cp "evolution/evaluators/${EVALUATOR}_evaluator.py" "$BACKUP_PATH"
echo "✅ Backup created: $BACKUP_PATH"

# Promote merged version
cp "$MERGED_PATH" "evolution/evaluators/${EVALUATOR}_evaluator.py"
echo "✅ Promoted merged version to primary"

# Log the promotion
echo "$(date): Promoted $SELECTED to replace ${EVALUATOR}_evaluator.py" >> logs/promotion_history.log

echo ""
echo "🎉 Version promoted successfully!"
echo "The super evolved version is now the primary evaluator."
