#!/bin/bash
# DMAI Smart Project Audit - Optimized for size and relevance

REPORT_FILE="DMAI_SMART_AUDIT_$(date +%Y%m%d_%H%M%S).txt"

{
echo "================================================================================="
echo "DMAI SMART PROJECT AUDIT (OPTIMIZED)"
echo "Generated: $(date)"
echo "================================================================================="
echo ""

# SECTION 0: PENDING TASKS VERIFICATION (MOST IMPORTANT)
echo "0. PENDING TASKS VERIFICATION"
echo "================================================================================="
echo ""

# Task 1: Music learner
echo "[ ] Music learner (develop_dmai_taste)"
if [ -f "music_learner.py" ]; then
    if grep -q "def develop_dmai_taste" music_learner.py; then
        echo "  ✅ Function exists"
        # Show just the function signature
        grep -A 2 "def develop_dmai_taste" music_learner.py | head -3
    else
        echo "  ❌ Function MISSING"
    fi
else
    echo "  ❌ File MISSING"
fi
echo ""

# Task 2: Safety module
echo "[ ] Safety module"
if [ -f "safety.py" ]; then
    if grep -q "^import sys" safety.py; then
        echo "  ✅ sys import exists"
    else
        echo "  ❌ sys import MISSING"
    fi
    # Show key functions
    grep -E "def (check_safety|SafetyMonitor)" safety.py | head -5
else
    echo "  ❌ safety.py NOT FOUND"
fi
echo ""

# Task 3: OpenSSL warning
echo "[ ] OpenSSL/urllib3"
pip list | grep urllib3 || echo "  urllib3 not installed"
echo ""

# Task 4: Voice enrollment
echo "[ ] Voice enrollment"
ls -la voice/enroll_master*.py 2>/dev/null | head -5
ls -la data/voice*.json 2>/dev/null || echo "  No voice profile found"
echo ""

# Task 5: Music identification
echo "[ ] Music identification"
ls -la music_identifier.py 2>/dev/null || echo "  No music_identifier.py"
ls -la data/music/*.json 2>/dev/null | head -5
echo ""

# Task 6: Cloud UI
echo "[ ] Cloud UI"
ls -la cloud_web_ui.py dmai_web_ui.py 2>/dev/null
pgrep -f "python.*cloud_web_ui" && echo "  ✅ Running" || echo "  ❌ Not running"
echo ""

# Task 7: Daily report
echo "[ ] Daily report"
ls -la daily_report.py 2>/dev/null
crontab -l 2>/dev/null | grep daily_report || echo "  No cron job"
echo ""

# Task 8: Mobile integration
echo "[ ] Mobile integration"
ls -la mobile/ ios/ android/ 2>/dev/null | head -10
echo ""

# Task 9: Self-sustaining
echo "[ ] Self-sustaining"
ls -la monitor.py health_check.py self_healer.py backup.py 2>/dev/null
echo ""

# SECTION 1: DIRECTORY STRUCTURE (SUMMARY)
echo "1. PROJECT STRUCTURE SUMMARY"
echo "================================================================================="
echo ""
echo "Top-level directories:"
ls -la | grep -E "^d" | awk '{print "  " $9}' | head -20
echo ""
echo "Key Python files:"
find . -maxdepth 2 -name "*.py" -not -path "*/venv/*" | sort | head -20
echo "  ... (showing first 20 only)"
echo ""

# SECTION 2: VOICE SYSTEM OVERVIEW
echo "2. VOICE SYSTEM OVERVIEW"
echo "================================================================================="
echo ""
if [ -d "voice" ]; then
    echo "Voice directory contents:"
    ls -la voice/ | grep -E "\.py$" | awk '{print "  " $9 " (" $5 " bytes)"}'
    echo ""
    echo "Key voice files (first 10 lines each):"
    for file in voice/dmai_voice_with_learning.py voice/enroll_master_comprehensive.py; do
        if [ -f "$file" ]; then
            echo "📄 $(basename $file):"
            head -10 "$file" | grep -v "^$" | sed 's/^/    /'
            echo "    ..."
        fi
    done
else
    echo "  No voice directory"
fi
echo ""

# SECTION 3: MUSIC SYSTEM OVERVIEW
echo "3. MUSIC SYSTEM OVERVIEW"
echo "================================================================================="
echo ""
ls -la music*.py 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes)"}'
if [ -d "data/music" ]; then
    echo ""
    echo "Music data:"
    ls -la data/music/ | grep "\.json" | head -5
fi
echo ""

# SECTION 4: RUNNING PROCESSES
echo "4. RUNNING DMAI PROCESSES"
echo "================================================================================="
echo ""
ps aux | grep -E "python.*(dmai|voice|learner|music|cloud)" | grep -v grep | head -10
echo ""

# SECTION 5: RECENT CHANGES (LAST 24H)
echo "5. RECENT CHANGES (LAST 24 HOURS)"
echo "================================================================================="
echo ""
find . -type f -not -path "*/venv/*" -mtime -1 -ls 2>/dev/null | head -20 | awk '{print "  " $11 " (" $7 " bytes)"}'
echo ""

# SECTION 6: TODO.MD
echo "6. TODO.MD CONTENTS"
echo "================================================================================="
echo ""
if [ -f "TODO.md" ]; then
    grep -E "^##|^- \[[ x]\]" TODO.md | head -30
else
    echo "  TODO.md not found"
fi
echo ""

# SECTION 7: SUMMARY TABLE
echo "7. QUICK STATUS SUMMARY"
echo "================================================================================="
echo ""
printf "%-25s %s\n" "COMPONENT" "STATUS"
printf "%-25s %s\n" "-------------------------" "------"
printf "%-25s %s\n" "Music Learner" "$([ -f music_learner.py ] && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Safety Module" "$(python -c 'import safety' 2>/dev/null && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Voice Enrollment" "$(ls voice/enroll_master*.py 2>/dev/null | wc -l | tr -d ' ') files"
printf "%-25s %s\n" "Voice Profile" "$([ -f data/voice_profile.json ] && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Music ID" "$([ -f music_identifier.py ] && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Cloud UI" "$([ -f cloud_web_ui.py ] && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Daily Report" "$([ -f daily_report.py ] && echo "✅" || echo "❌")"
printf "%-25s %s\n" "Mobile Integration" "$(ls -d mobile ios android 2>/dev/null | wc -l) dirs"
echo ""

echo "================================================================================="
echo "END OF AUDIT"
echo "================================================================================="

} > "$REPORT_FILE"

# Copy to clipboard (macOS)
cat "$REPORT_FILE" | pbcopy

echo "✅ SMART AUDIT COMPLETE!"
echo "📄 Report: $REPORT_FILE"
echo "📋 Size: $(wc -l < "$REPORT_FILE") lines"
echo "📋 Copied to clipboard - ready to paste!"
