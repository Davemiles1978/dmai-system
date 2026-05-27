"""
Virtual reset tracker for paper trading
Tracks performance from a reset point without actually resetting Alpaca
"""

import json
import time
from pathlib import Path
from datetime import datetime

class ResetTracker:
    def __init__(self):
        self.reset_file = Path("data/reset_state.json")
        self._init_reset()
    
    def _init_reset(self):
        if not self.reset_file.exists():
            self._save_reset({
                "reset_balance": 100000,
                "current_balance": 100000,
                "reset_time": time.time(),
                "trades_since_reset": [],
                "total_pnl": 0
            })
    
    def _save_reset(self, data):
        with open(self.reset_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_status(self):
        with open(self.reset_file, 'r') as f:
            return json.load(f)
    
    def record_trade(self, pnl):
        data = self.get_status()
        data["trades_since_reset"].append({
            "timestamp": time.time(),
            "pnl": pnl
        })
        data["total_pnl"] += pnl
        data["current_balance"] = data["reset_balance"] + data["total_pnl"]
        self._save_reset(data)
    
    def perform_reset(self):
        self._save_reset({
            "reset_balance": 100000,
            "current_balance": 100000,
            "reset_time": time.time(),
            "trades_since_reset": [],
            "total_pnl": 0
        })
        return {"success": True, "message": "Reset to $100,000"}

reset_tracker = ResetTracker()
