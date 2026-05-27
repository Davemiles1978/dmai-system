"""
Trading Mastery System - DMAI studies and masters all trading types
Tracks algorithms, strategies, and performance metrics
"""

import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

class TradingType(Enum):
    QUANT = "Quantitative Trading"
    LATENCY_ARB = "Latency Arbitrage"
    STAT_ARB = "Statistical Arbitrage"
    DAY_TRADING = "Day Trading"
    CRYPTO = "Crypto Trading"
    FOREX = "FOREX Trading"
    MOMENTUM = "Momentum Trading"
    MEAN_REVERSION = "Mean Reversion"

@dataclass
class TradeRecord:
    """Record of a single trade"""
    id: str
    timestamp: float
    symbol: str
    trading_type: str
    algorithm: str
    action: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    confidence: float
    reasoning: str

@dataclass
class Algorithm:
    """Trading algorithm definition"""
    name: str
    trading_type: str
    description: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict
    performance: Dict
    status: str  # learning, testing, active, mastered

class TradingMasterySystem:
    """DMAI learns and masters all trading types"""
    
    def __init__(self):
        self.db_path = Path("data/trading_mastery.db")
        self.algorithms = {}
        self.trade_history = []
        self.performance_metrics = {}
        self._init_db()
        self._load_algorithms()
    
    def _init_db(self):
        """Initialize trading mastery database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                symbol TEXT,
                trading_type TEXT,
                algorithm TEXT,
                action TEXT,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_percent REAL,
                confidence REAL,
                reasoning TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_type TEXT,
                algorithm TEXT,
                date TEXT,
                daily_pnl REAL,
                win_rate REAL,
                sharpe REAL,
                max_drawdown REAL,
                trades_count INTEGER
            )
        ''')
        
        # Learning progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                trading_type TEXT PRIMARY KEY,
                mastery_level REAL,
                papers_studied INTEGER,
                strategies_implemented INTEGER,
                backtests_run INTEGER,
                last_update REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_algorithms(self):
        """Load all trading algorithms"""
        self.algorithms = {
            TradingType.QUANT.value: [
                Algorithm(
                    name="Factor Investing",
                    trading_type=TradingType.QUANT.value,
                    description="Multi-factor model using value, momentum, quality factors",
                    entry_conditions=["Z-score < -2", "Factor composite > threshold"],
                    exit_conditions=["Z-score > 0", "Stop loss -5%"],
                    risk_management={"position_size": "2%", "max_correlation": 0.7},
                    performance={"sharpe": 1.2, "win_rate": 0.55},
                    status="active"
                ),
                Algorithm(
                    name="Statistical Arbitrage (Pairs)",
                    trading_type=TradingType.STAT_ARB.value,
                    description="Pairs trading on cointegrated assets",
                    entry_conditions=["Spread > 2 std dev", "Cointegration p-value < 0.05"],
                    exit_conditions=["Spread < 0.5 std dev", "Mean reversion confirmed"],
                    risk_management={"hedge_ratio": "calculated", "max_exposure": "5%"},
                    performance={"sharpe": 1.5, "win_rate": 0.65},
                    status="learning"
                )
            ]
        }
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade for analysis"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id, trade.timestamp, trade.symbol, trade.trading_type,
            trade.algorithm, trade.action, trade.quantity, trade.entry_price,
            trade.exit_price, trade.pnl, trade.pnl_percent, trade.confidence,
            trade.reasoning
        ))
        conn.commit()
        conn.close()
        self.trade_history.append(trade)
    
    def update_performance(self, trading_type: str, algorithm: str, daily_data: Dict):
        """Update performance metrics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance (trading_type, algorithm, date, daily_pnl, win_rate, sharpe, max_drawdown, trades_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trading_type, algorithm, datetime.now().date().isoformat(),
            daily_data.get('daily_pnl', 0), daily_data.get('win_rate', 0),
            daily_data.get('sharpe', 0), daily_data.get('max_drawdown', 0),
            daily_data.get('trades_count', 0)
        ))
        conn.commit()
        conn.close()
    
    def generate_report(self, trading_type: str = None) -> Dict:
        """Generate comprehensive performance report"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "trading_types": {},
            "overall": {}
        }
        
        if trading_type:
            cursor.execute("SELECT * FROM trades WHERE trading_type = ?", (trading_type,))
        else:
            cursor.execute("SELECT * FROM trades")
        
        trades = cursor.fetchall()
        
        if trades:
            total_pnl = sum(t[10] for t in trades)
            winning_trades = [t for t in trades if t[10] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            report["overall"] = {
                "total_trades": len(trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate * 100,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
                "best_trade": max(trades, key=lambda x: x[10])[10] if trades else 0,
                "worst_trade": min(trades, key=lambda x: x[10])[10] if trades else 0
            }
        
        conn.close()
        return report
    
    def backtest_algorithm(self, algorithm: Algorithm, historical_data: List[float]) -> Dict:
        """Run backtest on an algorithm"""
        results = {
            "algorithm": algorithm.name,
            "trading_type": algorithm.trading_type,
            "total_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "trades": []
        }
        
        # Simulate trading
        position = False
        entry_price = 0
        trades = []
        
        for i, price in enumerate(historical_data):
            if not position and self._check_entry(price, algorithm.entry_conditions):
                position = True
                entry_price = price
                trades.append({"action": "BUY", "price": price, "index": i})
            elif position and self._check_exit(price, algorithm.exit_conditions):
                position = False
                pnl = (price - entry_price) / entry_price
                trades.append({"action": "SELL", "price": price, "pnl": pnl})
                results["trades"].append({"entry": entry_price, "exit": price, "pnl": pnl})
        
        if trades:
            pnls = [t.get('pnl', 0) for t in results["trades"]]
            results["total_return"] = sum(pnls)
            results["win_rate"] = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
            results["sharpe_ratio"] = (sum(pnls) / len(pnls)) / (sum((p - (sum(pnls)/len(pnls)))**2 for p in pnls) / len(pnls))**0.5 if pnls else 0
        
        return results
    
    def _check_entry(self, price: float, conditions: List[str]) -> bool:
        """Check if entry conditions are met"""
        # Simplified - real implementation would have actual calculations
        return False
    
    def _check_exit(self, price: float, conditions: List[str]) -> bool:
        """Check if exit conditions are met"""
        return False
    
    def get_learning_status(self) -> Dict:
        """Get DMAI's learning progress for each trading type"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM learning_progress")
        rows = cursor.fetchall()
        conn.close()
        
        status = {}
        for row in rows:
            status[row[0]] = {
                "mastery_level": row[1] * 100,
                "papers_studied": row[2],
                "strategies_implemented": row[3],
                "backtests_run": row[4]
            }
        
        # Add default status for missing types
        for t in TradingType:
            if t.value not in status:
                status[t.value] = {
                    "mastery_level": 0,
                    "papers_studied": 0,
                    "strategies_implemented": 0,
                    "backtests_run": 0,
                    "status": "not_started"
                }
        
        return status

class ChartGenerator:
    """Generate P&L charts for trading performance"""
    
    @staticmethod
    def generate_pnl_chart(trades: List[TradeRecord], save_path: str):
        """Generate P&L chart from trade history"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not trades:
                return {"error": "No trades to chart"}
            
            # Calculate cumulative P&L
            trades_sorted = sorted(trades, key=lambda x: x.timestamp)
            cumulative = []
            running = 0
            dates = []
            
            for trade in trades_sorted:
                running += trade.pnl
                cumulative.append(running)
                dates.append(datetime.fromtimestamp(trade.timestamp))
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, cumulative, 'b-', linewidth=2)
            plt.fill_between(dates, 0, cumulative, where=[c > 0 for c in cumulative], color='green', alpha=0.3)
            plt.fill_between(dates, 0, cumulative, where=[c < 0 for c in cumulative], color='red', alpha=0.3)
            plt.title('DMAI Trading Performance (Paper Trading)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative P&L ($)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add performance annotations
            total_pnl = cumulative[-1] if cumulative else 0
            win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) * 100 if trades else 0
            
            plt.text(0.02, 0.98, f'Total P&L: ${total_pnl:,.2f}\nWin Rate: {win_rate:.1f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()
            
            return {"success": True, "chart_path": save_path}
        except Exception as e:
            return {"error": str(e)}

def initialize_trading_mastery():
    return TradingMasterySystem()
