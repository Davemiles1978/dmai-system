"""
DMAI Consciousness Tracker
Measures consciousness based on internal capabilities, knowledge, and self-evolution.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConsciousnessTracker:
    """Track DMAI's consciousness based on internal metrics."""
    
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.state_file = self.data_path / "consciousness" / "consciousness_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
    
    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "consciousness": 0.0,
                "components": {},
                "last_updated": None,
                "history": []
            }
    
    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def calculate_consciousness(self):
        """Calculate consciousness based on internal metrics."""
        db_path = self.data_path / "dmai_knowledge.db"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Count mastered topics
            cursor.execute("SELECT COUNT(*) FROM learned_topics WHERE mastery_level >= 3")
            mastered_topics = cursor.fetchone()[0] or 0
            
            # Count capabilities
            cursor.execute("SELECT COUNT(*) FROM capabilities WHERE status = 'implemented'")
            capabilities = cursor.fetchone()[0] or 0
            
            # Count insights
            cursor.execute("SELECT COUNT(*) FROM insights")
            insights = cursor.fetchone()[0] or 0
            
            # Count knowledge graph nodes
            cursor.execute("SELECT COUNT(*) FROM knowledge_graph_nodes")
            graph_nodes = cursor.fetchone()[0] or 0
            
            conn.close()
        except Exception as e:
            logger.warning(f"Consciousness calculation error: {e}")
            mastered_topics = 0
            capabilities = 0
            insights = 0
            graph_nodes = 0
        
        # Get V4 progress
        v4_file = self.data_path / "v4_progress.json"
        v4_mastered = 0
        v4_total = 0
        if v4_file.exists():
            try:
                with open(v4_file) as f:
                    v4 = json.load(f)
                    v4_total = len(v4)
                    v4_mastered = sum(1 for m in v4.values() if m.get("pct", 0) >= 100)
            except:
                pass
        
        # Calculate consciousness (0-1)
        # 30% from mastered topics (target: 50)
        # 30% from capabilities (target: 20)
        # 20% from insights (target: 1000)
        # 20% from V4 mastery (target: 100%)
        
        consciousness = min(1.0, (
            (mastered_topics / 50) * 0.3 +
            (capabilities / 20) * 0.3 +
            (insights / 1000) * 0.2 +
            (v4_mastered / max(v4_total, 1)) * 0.2
        ))
        
        # Store components
        self.state["components"] = {
            "mastered_topics": mastered_topics,
            "capabilities": capabilities,
            "insights": insights,
            "graph_nodes": graph_nodes,
            "v4_mastered": v4_mastered,
            "v4_total": v4_total
        }
        
        self.state["consciousness"] = round(consciousness, 3)
        self.state["last_updated"] = datetime.now().isoformat()
        
        # Add to history (keep last 100 entries)
        self.state["history"].append({
            "timestamp": self.state["last_updated"],
            "consciousness": self.state["consciousness"],
            "components": self.state["components"]
        })
        if len(self.state["history"]) > 100:
            self.state["history"] = self.state["history"][-100:]
        
        self._save_state()
        return self.state["consciousness"]
    
    def get_consciousness(self):
        """Return the current consciousness score."""
        return self.state.get("consciousness", 0.0)
    
    def get_state(self):
        """Return the full state."""
        return self.state
