#!/bin/bash
# DMAI Complete Project Audit - With Pending Task Verification

REPORT_FILE="DMAI_COMPLETE_AUDIT_$(date +%Y%m%d_%H%M%S).txt"

{
echo "================================================================================="
echo "DMAI COMPLETE PROJECT AUDIT WITH TASK VERIFICATION"
echo "Generated: $(date)"
echo "User: $(whoami)"
echo "================================================================================="
echo ""

# SECTION 0: PENDING TASKS VERIFICATION
echo "0. PENDING TASKS VERIFICATION STATUS"
echo "================================================================================="
echo ""

# Task 1: Music learner fix
echo "[ ] Fix music learner (AttributeError in develop_dmai_taste)"
if [ -f "music_learner.py" ]; then
    if grep -q "def develop_dmai_taste" music_learner.py; then
        echo "  ✅ develop_dmai_taste() function exists"
        # Check if it was the fixed version
        if grep -q "def develop_dmai_taste.*try.*except" music_learner.py; then
            echo "  ✅ Function has error handling (FIXED)"
        else
            echo "  ⚠️ Function exists but may still have AttributeError"
        fi
    else
        echo "  ❌ develop_dmai_taste() function NOT FOUND"
    fi
else
    echo "  ❌ music_learner.py NOT FOUND"
fi
echo ""

# Task 2: Safety module fix
echo "[ ] Fix safety module import error ('sys' not defined)"
if [ -f "safety.py" ]; then
    if grep -q "^import sys" safety.py; then
        echo "  ✅ sys import exists"
    else
        echo "  ❌ sys import MISSING"
    fi
    # Test import
    python -c "import safety" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ safety module imports successfully"
    else
        echo "  ❌ safety module import FAILS"
    fi
else
    echo "  ❌ safety.py NOT FOUND"
fi
echo ""

# Task 3: OpenSSL warning
echo "[ ] Resolve urllib3/OpenSSL warning for cloud UI"
if [ -f "cloud_web_ui.py" ] || [ -f "dmai_web_ui.py" ]; then
    # Check urllib3 version
    pip list | grep urllib3 | while read line; do
        if echo "$line" | grep -q "urllib3.*2\.0"; then
            echo "  ⚠️ urllib3 version 2.0+ detected (may cause OpenSSL warning)"
        else
            echo "  ✅ urllib3 version: $line"
        fi
    done
    # Check for fix attempts
    if [ -f "fix_system.py" ] && grep -q "fix_openssl_warning" fix_system.py; then
        echo "  ✅ fix_openssl_warning function exists"
    else
        echo "  ❌ No OpenSSL fix found"
    fi
else
    echo "  ⚠️ No cloud UI files found"
fi
echo ""

# Task 4: Voice enrollment
echo "[ ] Complete comprehensive voice enrollment"
VOICE_ENROLL_FILES=0
if [ -d "voice" ]; then
    ls voice/enroll_master*.py 2>/dev/null | while read file; do
        echo "  ✅ Found: $file"
        VOICE_ENROLL_FILES=$((VOICE_ENROLL_FILES+1))
    done
    if [ $VOICE_ENROLL_FILES -eq 0 ]; then
        echo "  ❌ No enrollment files found"
    fi
    # Check for voice profiles
    if [ -f "data/voice_profile.json" ] || [ -f "voice/auth/voiceprints.json" ]; then
        echo "  ✅ Voice profile exists (enrollment complete)"
    else
        echo "  ⚠️ No voice profile found (enrollment not complete)"
    fi
else
    echo "  ❌ Voice directory not found"
fi
echo ""

# Task 5: Music identification
echo "[ ] Improve music identification (beyond 'Unknown')"
if [ -f "music_identifier.py" ]; then
    echo "  ✅ music_identifier.py exists"
    # Check for API integrations
    for api in spotify acoustid shazam lastfm; do
        if grep -q -i "$api" music_identifier.py 2>/dev/null; then
            echo "  ✅ $api integration found"
        fi
    done
else
    echo "  ❌ music_identifier.py NOT FOUND"
fi
# Check music learner data
if [ -d "data/music" ]; then
    if [ -f "data/music/artists.json" ]; then
        ARTIST_COUNT=$(python -c "import json; print(len(json.load(open('data/music/artists.json')).get('artists', {})))" 2>/dev/null)
        echo "  📊 Artists known: $ARTIST_COUNT"
    fi
fi
echo ""

# Task 6: Cloud UI functional
echo "[ ] Get cloud UI fully functional"
CLOUD_FILES=0
for file in cloud_web_ui.py dmai_web_ui.py app.py; do
    if [ -f "$file" ]; then
        echo "  ✅ Found: $file"
        CLOUD_FILES=$((CLOUD_FILES+1))
        # Check for common issues
        if grep -q "app.run.*debug=True" "$file"; then
            echo "  ⚠️  Debug mode enabled (not for production)"
        fi
        if grep -q "CORS" "$file"; then
            echo "  ✅ CORS handling found"
        fi
    fi
done
if [ $CLOUD_FILES -eq 0 ]; then
    echo "  ❌ No cloud UI files found"
fi
# Check if it's running
if pgrep -f "python.*cloud_web_ui.py" > /dev/null || pgrep -f "python.*dmai_web_ui.py" > /dev/null; then
    echo "  ✅ Cloud UI is RUNNING"
    # Try to get status
    curl -s http://localhost:5000/health 2>/dev/null | grep -q "ok" && echo "  ✅ Health check passed"
else
    echo "  ⚠️ Cloud UI not running"
fi
echo ""

# Task 7: Daily report automation
echo "[ ] Daily report automation"
if [ -f "daily_report.py" ]; then
    echo "  ✅ daily_report.py exists"
    # Check for scheduling
    if crontab -l 2>/dev/null | grep -q "daily_report"; then
        echo "  ✅ Cron job configured"
    else
        echo "  ⚠️ No cron job found for daily report"
    fi
    # Check for email integration
    if grep -q "smtp\|email\|send_mail" daily_report.py; then
        echo "  ✅ Email integration found"
    fi
else
    echo "  ❌ daily_report.py NOT FOUND"
fi
echo ""

# Task 8: Mobile phone integration
echo "[ ] Mobile phone integration"
MOBILE_FILES=0
for dir in mobile ios android; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir directory exists"
        MOBILE_FILES=$((MOBILE_FILES+1))
        ls -la "$dir" 2>/dev/null | head -5
    fi
done
if [ -f "api_server.py" ]; then
    echo "  ✅ api_server.py exists (mobile backend)"
fi
if [ $MOBILE_FILES -eq 0 ]; then
    echo "  ❌ No mobile integration files found"
fi
echo ""

# Task 9: Self-sustaining resources
echo "[ ] Self-sustaining resources"
# Check for auto-scaling/monitoring
if [ -f "monitor.py" ] || [ -f "health_check.py" ]; then
    echo "  ✅ Monitoring files found"
fi
# Check for backup automation
if [ -f "backup.py" ] || crontab -l 2>/dev/null | grep -q "backup"; then
    echo "  ✅ Backup automation found"
fi
# Check for resource optimization
if [ -f "cleanup.py" ] || [ -f "optimize.py" ]; then
    echo "  ✅ Resource optimization found"
fi
# Check for self-healing
if [ -f "self_healer.py" ]; then
    echo "  ✅ Self-healing module found"
fi
echo ""

# SECTION 1: COMPLETE DIRECTORY TREE
echo "1. COMPLETE PROJECT DIRECTORY TREE"
echo "================================================================================="
echo ""
if command -v tree &> /dev/null; then
    tree -a -I 'venv|__pycache__|*.pyc|.git|.idea|.DS_Store' --dirsfirst
else
    find . -type d -not -path "*/venv/*" -not -path "*/.git/*" -not -path "*/__pycache__/*" | sort | sed 's/[^-][^\/]*\//  /g' | sed 's/^/  /'
fi
echo ""

# SECTION 2: ALL PYTHON FILES
echo "2. ALL PYTHON FILES (WITH FIRST 5 LINES)"
echo "================================================================================="
echo ""
find . -name "*.py" -not -path "*/venv/*" -not -path "*/__pycache__/*" | sort | while read file; do
    echo "📄 $file"
    echo "   Size: $(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null) bytes"
    echo "   Modified: $(stat -f%Sm "$file" 2>/dev/null || date -r "$file" "+%Y-%m-%d %H:%M:%S")"
    echo "   First 5 lines:"
    head -5 "$file" | sed 's/^/     /'
    echo "   ---"
done
echo ""

# SECTION 3: TODO.MD COMPLETE
echo "3. TODO.MD COMPLETE CONTENTS"
echo "================================================================================="
echo ""
if [ -f "TODO.md" ]; then
    cat TODO.md
else
    echo "❌ TODO.md not found"
fi
echo ""

# SECTION 4: RUNNING PROCESSES
echo "4. RUNNING DMAI PROCESSES"
echo "================================================================================="
echo ""
ps aux | grep -E "python.*dmai|python.*voice|python.*learner|python.*music|python.*cloud" | grep -v grep || echo "  No DMAI processes running"
echo ""

# SECTION 5: SUMMARY
echo "5. QUICK SUMMARY"
echo "================================================================================="
echo ""
echo "Total Python files: $(find . -name "*.py" -not -path "*/venv/*" | wc -l)"
echo "Total JSON files: $(find . -name "*.json" -not -path "*/venv/*" | wc -l)"
echo ""
echo "PENDING TASKS STATUS:"
echo "---------------------"
echo "Music learner fix: $(grep -q "def develop_dmai_taste" music_learner.py 2>/dev/null && echo "✅" || echo "❌")"
echo "Safety module fix: $(python -c "import safety" 2>/dev/null && echo "✅" || echo "❌")"
echo "OpenSSL warning: $(pip list 2>/dev/null | grep -q "urllib3.*2\.0" && echo "⚠️" || echo "✅")"
echo "Voice enrollment: $(ls voice/enroll_master*.py 2>/dev/null | wc -l | tr -d ' ')/3 files, $(ls data/voice_profile.json 2>/dev/null && echo "profile✅" || echo "profile❌")"
echo "Music identification: $(ls music_identifier.py 2>/dev/null && echo "✅" || echo "❌")"
echo "Cloud UI: $(ls cloud_web_ui.py 2>/dev/null && echo "✅" || echo "❌")"
echo "Daily report: $(ls daily_report.py 2>/dev/null && echo "✅" || echo "❌")"
echo "Mobile integration: $(ls -d mobile ios android 2>/dev/null | wc -l | tr -d ' ') directories"
echo "Self-sustaining: $(ls monitor.py health_check.py self_healer.py 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

echo "================================================================================="
echo "END OF AUDIT"
echo "================================================================================="

} > "$REPORT_FILE"

# Copy to clipboard
cat "$REPORT_FILE" | pbcopy

echo "✅ COMPLETE AUDIT WITH TASK VERIFICATION GENERATED!"
echo "📄 Report saved to: $REPORT_FILE"
echo "📋 Report copied to clipboard"
echo ""
echo "The report includes:"
echo "  • All 9 pending tasks verified with status"
echo "  • Complete directory tree"
echo "  • All Python files with previews"
echo "  • TODO.md contents"
echo "  • Running processes"
echo "  • Quick summary table"
