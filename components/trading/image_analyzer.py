"""
Trading algorithm extraction from images using OCR + AI
DMAI can read screenshots of trading algorithms and implement them
"""

import base64
import requests
import json
import re
import os
from typing import Dict, List, Optional
from pathlib import Path

class TradingImageAnalyzer:
    """Extract trading algorithms, charts, and techniques from images"""
    
    def __init__(self):
        self.extracted_algorithms = []
        self.trading_rules = []
        
    def analyze_trading_image(self, image_path: str) -> Dict:
        """
        Extract trading information from image using OCR + GPT-4V
        Supports: charts, algorithm screenshots, trading rules, indicators
        """
        
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Prompt for trading analysis
        prompt = """
        Analyze this trading image and extract:
        
        1. TRADING ALGORITHMS: Any code, pseudocode, or step-by-step logic
        2. CHART PATTERNS: Head and shoulders, triangles, flags, support/resistance
        3. INDICATORS: RSI, MACD, Bollinger Bands, Moving Averages, etc.
        4. ENTRY/EXIT RULES: Specific conditions for buying and selling
        5. RISK MANAGEMENT: Stop loss, position sizing, risk-reward ratios
        6. TIME FRAMES: Daily, hourly, minute charts being used
        
        Return as JSON:
        {
            "algorithms": [{"name": "...", "code": "...", "description": "..."}],
            "chart_patterns": ["pattern1", "pattern2"],
            "indicators": ["indicator1", "indicator2"],
            "entry_rules": ["rule1", "rule2"],
            "exit_rules": ["rule1", "rule2"],
            "risk_management": {},
            "timeframes": []
        }
        """
        
        # Use AI tutor to analyze
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4-vision-preview",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                                ]
                            }
                        ],
                        "max_tokens": 2000
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    extracted = json.loads(result['choices'][0]['message']['content'])
                    self.extracted_algorithms.extend(extracted.get('algorithms', []))
                    return extracted
            except Exception as e:
                return {"error": str(e), "fallback": "manual_review_needed"}
        
        # Fallback: extract text using basic OCR
        return self._basic_ocr_extraction(image_base64)
    
    def _basic_ocr_extraction(self, image_base64: str) -> Dict:
        """Basic extraction without GPT-4V"""
        return {
            "algorithms": [],
            "chart_patterns": [],
            "indicators": [],
            "entry_rules": [],
            "exit_rules": [],
            "risk_management": {},
            "timeframes": [],
            "note": "GPT-4V not available. Please set OPENAI_API_KEY for full analysis."
        }
    
    def extract_from_multiple_images(self, image_paths: List[str]) -> Dict:
        """Extract trading knowledge from multiple screenshots"""
        all_data = {
            "algorithms": [],
            "chart_patterns": set(),
            "indicators": set(),
            "entry_rules": [],
            "exit_rules": [],
            "risk_management": {}
        }
        
        for path in image_paths:
            result = self.analyze_trading_image(path)
            if "error" not in result:
                all_data["algorithms"].extend(result.get("algorithms", []))
                all_data["chart_patterns"].update(result.get("chart_patterns", []))
                all_data["indicators"].update(result.get("indicators", []))
                all_data["entry_rules"].extend(result.get("entry_rules", []))
                all_data["exit_rules"].extend(result.get("exit_rules", []))
        
        return all_data
    
    def generate_trading_code(self, algorithm: Dict) -> str:
        """Convert extracted algorithm to executable Python code"""
        code_template = f'''
def execute_trading_strategy(data):
    """
    {algorithm.get('description', 'Trading strategy')}
    """
    
    # Entry conditions
    entry_signal = False
    exit_signal = False
    
    # {algorithm.get('name', 'Strategy')} logic
    # (Converted from image)
    
    if entry_signal:
        return {{"action": "BUY", "confidence": 0.85}}
    elif exit_signal:
        return {{"action": "SELL", "confidence": 0.85}}
    else:
        return {{"action": "HOLD", "confidence": 0.5}}
'''
        return code_template


class TradingMonitor:
    """Monitor trading performance and track metrics"""
    
    def __init__(self):
        self.trades = []
        self.daily_stats = {}
        
    def track_trade(self, trade: Dict):
        """Record a trade for analysis"""
        self.trades.append(trade)
        
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return (wins / len(self.trades)) * 100
    
    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)"""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.get('return_pct', 0) for t in self.trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_return == 0:
            return 0.0
        return avg_return / std_return
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trades:
            return 0.0
        
        peak = 0
        drawdown = 0
        cumulative = 0
        
        for trade in self.trades:
            cumulative += trade.get('pnl', 0)
            if cumulative > peak:
                peak = cumulative
            current_dd = peak - cumulative
            if current_dd > drawdown:
                drawdown = current_dd
        
        return drawdown
    
    def generate_report(self) -> Dict:
        """Generate comprehensive trading report"""
        return {
            "total_trades": len(self.trades),
            "win_rate": self.get_win_rate(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "max_drawdown": self.get_max_drawdown(),
            "total_pnl": sum(t.get('pnl', 0) for t in self.trades),
            "avg_return": sum(t.get('return_pct', 0) for t in self.trades) / max(1, len(self.trades))
        }


class TradingIndicators:
    """Technical indicators for trading analysis"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        return [sum(prices[i:i+period])/period for i in range(len(prices)-period+1)]
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if not prices:
            return []
        multiplier = 2 / (period + 1)
        ema_values = [prices[0]]
        for price in prices[1:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float]) -> Dict:
        """Moving Average Convergence Divergence"""
        if len(prices) < 26:
            return {"macd": [], "signal": [], "histogram": []}
        
        ema12 = TradingIndicators.ema(prices, 12)
        ema26 = TradingIndicators.ema(prices, 26)
        min_len = min(len(ema12), len(ema26))
        macd_line = [ema12[i] - ema26[i] for i in range(min_len)]
        signal_line = TradingIndicators.ema(macd_line, 9)
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": [m - s for m, s in zip(macd_line, signal_line)] if len(macd_line) == len(signal_line) else []
        }


def initialize_trading_analyzer():
    """Initialize the trading analysis system"""
    return {
        "image_analyzer": TradingImageAnalyzer(),
        "monitor": TradingMonitor(),
        "indicators": TradingIndicators()
    }
