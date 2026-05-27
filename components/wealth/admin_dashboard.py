"""
Admin Dashboard - Mobile-friendly, real-time monitoring
"""

import json
import time
import os
import requests
from pathlib import Path
from datetime import datetime

class AdminDashboard:
    def __init__(self):
        self.cache_file = Path("data/dashboard_cache.json")
        self.cache_duration = 10  # seconds
    
    def get_trading_status(self):
        """Get current trading status from Alpaca"""
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')
        paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
        
        if not api_key or not secret_key:
            return {"error": "Trading not configured"}
        
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        headers = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': secret_key}
        
        try:
            account_resp = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
            account = account_resp.json() if account_resp.status_code == 200 else {}
            
            positions_resp = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
            positions = positions_resp.json() if positions_resp.status_code == 200 else []
            
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            profit = equity - 100000
            profit_percent = (profit / 100000) * 100
            
            # Get reset tracker status
            from components.wealth.reset_tracker import reset_tracker
            reset_data = reset_tracker.get_status()
            
            return {
                "success": True,
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "profit": round(profit, 2),
                "profit_percent": round(profit_percent, 2),
                "reset_balance": reset_data.get("reset_balance", 100000),
                "reset_pnl": reset_data.get("total_pnl", 0),
                "positions_count": len(positions),
                "positions": positions,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_learning_status(self):
        """Get DMAI's learning progress"""
        return {
            "Quantitative Trading": {"mastery": 65, "status": "learning"},
            "Statistical Arbitrage": {"mastery": 55, "status": "learning"},
            "Day Trading": {"mastery": 45, "status": "learning"},
            "Crypto Trading": {"mastery": 40, "status": "researching"},
            "Latency Arbitrage": {"mastery": 30, "status": "researching"},
            "FOREX": {"mastery": 35, "status": "planned"}
        }
    
    def get_dashboard_data(self):
        """Get all dashboard data in one call"""
        trading = self.get_trading_status()
        learning = self.get_learning_status()
        
        # Get consciousness
        try:
            consciousness_resp = requests.get("https://dmai-web.onrender.com/api/status", timeout=10)
            consciousness = consciousness_resp.json().get('consciousness', 0) if consciousness_resp.status_code == 200 else 0
        except:
            consciousness = 0
        
        return {
            "trading": trading,
            "learning": learning,
            "consciousness": consciousness,
            "timestamp": datetime.now().isoformat()
        }

admin_dashboard = AdminDashboard()
