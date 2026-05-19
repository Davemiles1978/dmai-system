"""
Real trading execution for DMAI - Alpaca Paper Trading
"""
import os
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class RealTradingExecutor:
    """Execute trades on Alpaca Paper Trading API"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self.paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
        
        if not self.api_key or not self.secret_key:
            print("⚠️ Alpaca API keys not set. Trading disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.base_url = 'https://paper-api.alpaca.markets' if self.paper else 'https://api.alpaca.markets'
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        self.positions = {}
        self.balance = 0.0
        self.trades = []
        
    def get_account(self) -> Dict:
        """Get account information"""
        if not self.enabled:
            return {"error": "Trading not enabled - missing API keys"}
        
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.balance = float(data.get('cash', 0))
                return data
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.enabled:
            return []
        
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
            if response.status_code == 200:
                self.positions = {p['symbol']: p for p in response.json()}
                return response.json()
        except Exception as e:
            print(f"Error getting positions: {e}")
        return []
    
    def place_order(self, symbol: str, qty: float, side: str, order_type: str = 'market') -> Dict:
        """Place a trading order"""
        if not self.enabled:
            return {"error": "Trading not enabled"}
        
        order = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side,
            'type': order_type,
            'time_in_force': 'day'
        }
        
        try:
            response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order, timeout=10)
            if response.status_code == 200:
                trade = response.json()
                self.trades.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': trade.get('filled_avg_price', 0),
                    'order_id': trade.get('id')
                })
                return trade
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def execute_strategy(self, strategy: Dict) -> Dict:
        """Execute a trading strategy"""
        if not self.enabled:
            return {"status": "disabled", "reason": "Trading not enabled"}
        
        symbol = strategy.get('symbol', 'AAPL')
        action = strategy.get('action', 'BUY')
        confidence = strategy.get('confidence', 0.5)
        qty = strategy.get('quantity', 1)
        
        if confidence < 0.7:
            return {"status": "skipped", "reason": f"Low confidence: {confidence}"}
        
        result = self.place_order(symbol, qty, action.lower())
        
        return {
            "status": "executed" if 'id' in result else "failed",
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "confidence": confidence,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance(self) -> Dict:
        """Get trading performance metrics"""
        if not self.enabled:
            return {"error": "Trading not enabled"}
        
        account = self.get_account()
        positions = self.get_positions()
        
        total_value = self.balance
        for pos in positions:
            total_value += float(pos.get('market_value', 0))
        
        return {
            "cash_balance": self.balance,
            "total_value": total_value,
            "open_positions": len(positions),
            "total_trades": len(self.trades),
            "pnl": total_value - 100000,  # Assuming $100k starting
            "timestamp": datetime.now().isoformat()
        }
    
    def run_autonomous(self):
        """Run autonomous trading based on DMAI's learned strategies"""
        if not self.enabled:
            print("⚠️ Trading disabled. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
            return {"status": "disabled"}
        
        print(f"🤖 Starting autonomous trading at {datetime.now()}")
        
        account = self.get_account()
        print(f"💰 Balance: ${self.balance:,.2f}")
        
        # Example strategies (will be replaced with DMAI's actual strategies)
        strategies = [
            {"symbol": "AAPL", "action": "BUY", "confidence": 0.85, "quantity": 1},
            {"symbol": "MSFT", "action": "BUY", "confidence": 0.75, "quantity": 1},
        ]
        
        for strategy in strategies:
            result = self.execute_strategy(strategy)
            print(f"📊 {result}")
            time.sleep(1)
        
        performance = self.get_performance()
        print(f"📈 Performance: ${performance.get('pnl', 0):+,.2f}")
        
        return performance

def initialize_trading():
    """Initialize real trading for DMAI"""
    return RealTradingExecutor()
