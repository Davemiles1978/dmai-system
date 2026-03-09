#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Phase 2 Completion Tracker - Self-Awareness
Run this to see progress and implement recommendations
"""
import json
from pathlib import Path
from datetime import datetime

class Phase2Tracker:
    def __init__(self):
        self.assessment_path = Path("shared_data/agi_evolution/assessment")
        self.status = {
            'learning_metrics_implemented': True,
            'knowledge_gap_identification': False,
            'learning_trend_tracking': False,
            'strategy_optimization': False,
            'recommendations_implemented': []
        }
    
    def check_recent_assessments(self):
        """Check recent assessment reports for recommendations"""
        if not self.assessment_path.exists():
            print("❌ No assessment directory found")
            return
        
        reports = list(self.assessment_path.glob("assessment_*.json"))
        if not reports:
            print("❌ No assessment reports found")
            return
        
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        with open(latest, 'r') as f:
            report = json.load(f)
        
        print(f"\n📋 Latest Assessment: {latest.name}")
        print(f"  Learning Quality: {report['summary']['learning_quality']}")
        print(f"  Knowledge Maturity: {report['summary']['knowledge_maturity']}")
        print(f"  System Health: {report['summary']['system_health']}")
        
        if report.get('recommendations'):
            print("\n💡 Pending Recommendations:")
            for rec in report['recommendations']:
                print(f"  • [{rec['priority'].upper()}] {rec['area']}: {rec['action']}")
                self.status['recommendations_implemented'].append(False)
    
    def show_progress(self):
        """Show Phase 2 completion progress"""
        total_tasks = 4
        completed = sum([
            self.status['learning_metrics_implemented'],
            self.status['knowledge_gap_identification'],
            self.status['learning_trend_tracking'],
            self.status['strategy_optimization']
        ])
        
        progress = (completed / total_tasks) * 100
        print(f"\n{'='*50}")
        print(f"📊 PHASE 2 COMPLETION: {progress:.1f}%")
        print(f"{'='*50}")
        print(f"✅ Learning Metrics: {self.status['learning_metrics_implemented']}")
        print(f"⬜ Knowledge Gap ID: {self.status['knowledge_gap_identification']}")
        print(f"⬜ Learning Trend: {self.status['learning_trend_tracking']}")
        print(f"⬜ Strategy Opt: {self.status['strategy_optimization']}")
        
        if self.status['recommendations_implemented']:
            impl = sum(self.status['recommendations_implemented'])
            total = len(self.status['recommendations_implemented'])
            print(f"\n📋 Recommendations: {impl}/{total} implemented")
    
    def run(self):
        self.check_recent_assessments()
        self.show_progress()
        
        print("\n🎯 Next Steps to Complete Phase 2:")
        if not self.status['knowledge_gap_identification']:
            print("  1. Implement knowledge gap identification in orchestrator")
        if not self.status['learning_trend_tracking']:
            print("  2. Add learning trend visualization")
        if not self.status['strategy_optimization']:
            print("  3. Auto-adjust strategies based on recommendations")
        if self.status['recommendations_implemented'] and False in self.status['recommendations_implemented']:
            print("  4. Implement pending recommendations from assessment")

if __name__ == "__main__":
    tracker = Phase2Tracker()
    tracker.run()
