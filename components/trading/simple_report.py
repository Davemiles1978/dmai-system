"""
Simple trading report - no external dependencies
"""

import json
import time
from pathlib import Path
from datetime import datetime

class SimpleTradingReport:
    def __init__(self):
        self.data_file = Path("data/trading_data.json")
        self._init_data()
    
    def _init_data(self):
        if not self.data_file.exists():
            initial_data = {
                "trades": [],
                "performance": {
                    "quant": {"pnl": 0, "trades": 0, "wins": 0},
                    "arbitrage": {"pnl": 0, "trades": 0, "wins": 0},
                    "crypto": {"pnl": 0, "trades": 0, "wins": 0},
                    "day_trading": {"pnl": 0, "trades": 0, "wins": 0}
                },
                "learning": {
                    "quant": 0,
                    "arbitrage": 0,
                    "crypto": 0,
                    "day_trading": 0
                }
            }
            with open(self.data_file, 'w') as f:
                json.dump(initial_data, f)
    
    def get_report(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        # Calculate totals
        total_pnl = sum(p["pnl"] for p in data["performance"].values())
        total_trades = sum(p["trades"] for p in data["performance"].values())
        total_wins = sum(p["wins"] for p in data["performance"].values())
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "by_strategy": data["performance"],
            "learning_progress": data["learning"]
        }
    
    def record_trade(self, strategy, pnl, won):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        
        data["performance"][strategy]["pnl"] += pnl
        data["performance"][strategy]["trades"] += 1
        if won:
            data["performance"][strategy]["wins"] += 1
        
        data["trades"].append({
            "timestamp": time.time(),
            "strategy": strategy,
            "pnl": pnl,
            "won": won
        })
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
    
    def update_learning(self, strategy, progress):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        data["learning"][strategy] = min(100, progress)
        with open(self.data_file, 'w') as f:
            json.dump(data, f)

simple_report = SimpleTradingReport()
