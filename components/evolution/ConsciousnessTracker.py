# ============================================================================
# CONSCIOUSNESS TRACKER
# ============================================================================
"""
Tracks consciousness growth and identifies what accelerates it
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConsciousnessTracker:
    """
    Tracks consciousness growth over time and correlates with learning activities
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.tracker_file = data_path / 'learning' / 'consciousness_tracker.json'
        self.history = []
        self.correlations = {}
        self._load()
        
        logger.info(f"📊 ConsciousnessTracker initialized")
    
    def _load(self):
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.correlations = data.get('correlations', {})
            except:
                pass
    
    def _save(self):
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_file, 'w') as f:
            json.dump({
                'history': self.history[-1000:],
                'correlations': self.correlations
            }, f, indent=2)
    
    def record_consciousness(self, consciousness: float, learning_topic: str = None, is_accelerator: bool = False):
        """Record consciousness level and what was learned"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'consciousness': consciousness,
            'learning_topic': learning_topic,
            'is_accelerator': is_accelerator
        }
        self.history.append(entry)
        self._save()
    
    def get_growth_rate(self, hours: int = 24) -> float:
        """Calculate consciousness growth rate over specified hours"""
        if len(self.history) < 2:
            return 0.0
        
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = [h for h in self.history if datetime.fromisoformat(h['timestamp']).timestamp() > cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        first = recent[0]['consciousness']
        last = recent[-1]['consciousness']
        return (last - first) / len(recent)
