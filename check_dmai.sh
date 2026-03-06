#!/bin/bash
echo "==================================="
echo "DMAI Health Check - $(date)"
echo "==================================="

# Check core processes
echo "📊 Running Services:"
ps aux | grep -E "dmai_core|services/|voice/" | grep -v grep | grep -v check_dmai

# Check database stats
echo ""
echo "📚 Learning Progress:"
sqlite3 /Users/davidmiles/Desktop/AI-Evolution-System/data/databases/dmai_research.db "SELECT source_used, COUNT(*) as attempts, SUM(success) as successes, SUM(words_learned) as words FROM research_history GROUP BY source_used;" 2>/dev/null || echo "   No research data yet"

# Check wallet
echo ""
echo "💰 Wallet Balances:"
sqlite3 /Users/davidmiles/Desktop/AI-Evolution-System/data/databases/dmai_research.db "SELECT owner, balance FROM wallet;" 2>/dev/null || echo "   No wallet data yet"

# Check recent activity
echo ""
echo "🔄 Recent Research:"
sqlite3 /Users/davidmiles/Desktop/AI-Evolution-System/data/databases/dmai_research.db "SELECT timestamp, topic, source_used, words_learned FROM research_history ORDER BY timestamp DESC LIMIT 3;" 2>/dev/null || echo "   No recent activity"

# Check crontab
echo ""
echo "⏰ Auto-start Configuration:"
crontab -l | grep -E "dmai_core|web_researcher" || echo "   No crontab entries found"
