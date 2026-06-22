"""
Aggressive trading strategies for DMAI - 80% consciousness optimized
"""

import requests
import time
import json
import random
from typing import Dict, List, Any
from datetime import datetime

class AggressiveTrader:
    """High-performance trading with 70-80% capital utilization"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Aggressive position sizing
        self.max_position_size = 0.70  # 70% of capital max
        self.min_position_size = 0.10   # 10% minimum
        self.risk_per_trade = 0.02      # 2% risk per trade
        
        # Trading parameters
        self.trading_pairs = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL']
        self.conservative_pairs = ['SPY', 'QQQ', 'IVV']  # ETFs for safety
        
    def get_account(self) -> Dict:
        """Get current account status"""
        response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "cash": float(data.get('cash', 0)),
                "equity": float(data.get('equity', 0)),
                "buying_power": float(data.get('buying_power', 0)),
                "profit_loss": float(data.get('equity', 0)) - 100000
            }
        return {"error": response.text}
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    
    def calculate_position_size(self, account_equity: float, confidence: float) -> float:
        """Calculate position size based on confidence and capital"""
        base_size = account_equity * self.max_position_size
        confidence_multiplier = confidence / 0.5  # Normalize around 0.5
        position_size = base_size * min(2.0, confidence_multiplier)
        return min(position_size, account_equity * self.max_position_size)
    
    def execute_buy(self, symbol: str, confidence: float) -> Dict:
        """Execute buy order with dynamic sizing"""
        account = self.get_account()
        if "error" in account:
            return {"error": account["error"]}
        
        # Calculate position size
        position_value = self.calculate_position_size(account["equity"], confidence)
        
        # Get current price
        price_response = requests.get(
            f"{self.base_url}/v2/stocks/{symbol}/quote",
            headers=self.headers,
            timeout=10
        )
        if price_response.status_code != 200:
            return {"error": "Could not get price"}
        
        current_price = float(price_response.json().get('bid_price', 0))
        qty = int(position_value / current_price)
        
        if qty < 1:
            return {"status": "skipped", "reason": f"Quantity too small: {qty}"}
        
        # Place market order
        order = {
            'symbol': symbol,
            'qty': qty,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day'
        }
        
        response = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order,
            timeout=10
        )
        
        if response.status_code == 200:
            order_data = response.json()
            return {
                "status": "executed",
                "symbol": symbol,
                "qty": qty,
                "price": current_price,
                "value": qty * current_price,
                "confidence": confidence,
                "order_id": order_data.get('id')
            }
        return {"error": response.text}
    
    def execute_sell(self, symbol: str) -> Dict:
        """Close position completely"""
        response = requests.delete(
            f"{self.base_url}/v2/positions/{symbol}",
            headers=self.headers,
            timeout=10
        )
        
        if response.status_code in [200, 204]:
            return {"status": "closed", "symbol": symbol}
        return {"error": response.text}
    
    def get_market_sentiment(self) -> float:
        """Get current market sentiment (0-1) from various sources"""
        # Simplified sentiment - in production would use real data
        hour = datetime.now().hour
        is_premarket = 4 <= hour <= 9
        is_after_hours = 16 <= hour <= 20
        
        if is_premarket:
            return 0.6  # Slightly bullish pre-market
        elif is_after_hours:
            return 0.5  # Neutral after hours
        else:
            return 0.7  # Bullish during market hours
    
    def generate_signals(self) -> List[Dict]:
        """Generate trading signals with confidence scores"""
        sentiment = self.get_market_sentiment()
        signals = []
        
        # Strong buy signals for top performers
        strong_buy = ['NVDA', 'MSFT', 'AAPL']
        for symbol in strong_buy:
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'confidence': 0.85 + (sentiment * 0.1),
                'reason': 'Strong technical momentum'
            })
        
        # Medium confidence signals
        medium_buy = ['AMZN', 'META', 'GOOGL']
        for symbol in medium_buy:
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'confidence': 0.70 + (sentiment * 0.1),
                'reason': 'Positive trend'
            })
        
        # ETFs for diversification
        for symbol in self.conservative_pairs:
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'confidence': 0.65,
                'reason': 'Market exposure'
            })
        
        return signals
    
    def execute_aggressive_trades(self) -> Dict:
        """Execute aggressive trading strategy"""
        account = self.get_account()
        if "error" in account:
            return {"error": account["error"]}
        
        # Get current positions
        current_positions = {p['symbol']: p for p in self.get_positions()}
        
        # Generate signals
        signals = self.generate_signals()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "account_value": account["equity"],
            "cash": account["cash"],
            "trades": []
        }
        
        # Execute buys for high-confidence signals
        for signal in signals:
            if signal['action'] == 'BUY' and signal['confidence'] > 0.7:
                # Don't double-allocate
                if signal['symbol'] in current_positions:
                    continue
                
                result = self.execute_buy(signal['symbol'], signal['confidence'])
                results["trades"].append(result)
                time.sleep(0.5)  # Rate limiting
        
        # Re-evaluate positions after trades
        results["positions"] = self.get_positions()
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        account = self.get_account()
        positions = self.get_positions()
        
        total_value = account.get("equity", 0)
        initial_capital = 100000
        profit_loss = total_value - initial_capital
        roi = (profit_loss / initial_capital) * 100
        
        return {
            "total_value": total_value,
            "profit_loss": profit_loss,
            "roi_percent": roi,
            "positions_count": len(positions),
            "capital_utilized": (total_value - account.get("cash", total_value)) / total_value * 100,
            "timestamp": datetime.now().isoformat()
        }

def get_aggressive_trader(api_key: str, secret_key: str, paper: bool = True):
    """Initialize aggressive trader"""
    return AggressiveTrader(api_key, secret_key, paper)
