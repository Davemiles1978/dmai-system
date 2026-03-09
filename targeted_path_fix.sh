#!/bin/bash
# Targeted fix for specific files with hardcoded paths

echo "🔧 DMAI TARGETED PATH FIX"
echo "=========================="

# Fix create_icon.py
if [ -f "create_icon.py" ]; then
    echo "📝 Fixing create_icon.py..."
    cp create_icon.py create_icon.py.backup
    sed -i '' 's|AI-Evolution-System|dmai-system|g' create_icon.py
    echo "✅ Fixed create_icon.py"
fi

# Fix final_cleanup.sh
if [ -f "final_cleanup.sh" ]; then
    echo "📝 Fixing final_cleanup.sh..."
    cp final_cleanup.sh final_cleanup.sh.backup
    sed -i '' 's|/Users/davidmiles/Desktop/AI-Evolution-System|/Users/davidmiles/Desktop/dmai-system|g' final_cleanup.sh
    echo "✅ Fixed final_cleanup.sh"
fi

# Fix clone_all_repos.sh
if [ -f "clone_all_repos.sh" ]; then
    echo "📝 Fixing clone_all_repos.sh..."
    cp clone_all_repos.sh clone_all_repos.sh.backup
    sed -i '' 's|AI-Evolution-System|dmai-system|g' clone_all_repos.sh
    echo "✅ Fixed clone_all_repos.sh"
fi

# Fix deploy_to_cloud_247.sh
if [ -f "deploy_to_cloud_247.sh" ]; then
    echo "📝 Fixing deploy_to_cloud_247.sh..."
    cp deploy_to_cloud_247.sh deploy_to_cloud_247.sh.backup
    sed -i '' 's|/Users/davidmiles/Desktop/AI-Evolution-System|/Users/davidmiles/Desktop/dmai-system|g' deploy_to_cloud_247.sh
    echo "✅ Fixed deploy_to_cloud_247.sh"
fi

# Check for any AI-Evolution-System.command file
if [ -f "../AI-Evolution-System.command" ]; then
    echo "📝 Fixing AI-Evolution-System.command..."
    cp "../AI-Evolution-System.command" "../AI-Evolution-System.command.backup"
    sed -i '' 's|AI-Evolution-System|dmai-system|g' "../AI-Evolution-System.command"
    echo "✅ Fixed AI-Evolution-System.command"
fi

echo -e "\n📊 Verification:"
echo "Checking for any remaining 'AI-Evolution-System' references:"
grep -r "AI-Evolution-System" --include="*.py" --include="*.sh" --include="*.json" . 2>/dev/null | grep -v "backup" | grep -v "fix_all_paths.py"

echo -e "\n✅ All targeted fixes complete!"
echo "Backups created with .backup extension"
echo -e "\nNext steps:"
echo "1. Test the voice service: python3 voice/dmai_voice_with_learning.py"
echo "2. Start all daemons: ./scripts/start_all_daemons.sh"
