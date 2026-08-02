#!/usr/bin/env python3
"""
Inject the multi-asset algorithmic trading knowledge and repository references
into DMAI's knowledge graph and parallel learner queue.
"""

import json
import sqlite3
import requests
from pathlib import Path

# ----------------------------------------------------------------------
# 1. The full trading knowledge text (copy‑pasted from the user message)
# ----------------------------------------------------------------------
TRADING_KNOWLEDGE = """
feed the below info into DMAI for her to utilise the repos for resources, analyse code snippets provided to see if they will improve our trading system, and implement any useful knowledge, whether it be code to create or repo to ingest (I have starred the repos in GitHub already);

To instantly download GitHub repositories filled with algorithmic trading strategies, historical trading rules, and quantitative models, you can clone the repositories directly via your command line or download them as a ZIP file.
Top Repositories to Download
Here are some of the best open-source repositories of trading rules and strategies:
Awesome Systematic Trading on GitHub: A massive curated list of academic papers, quantitative trading rules, and backtested strategies categorized by asset class. [1]
Vibe-Trading Agent on GitHub: An agent that extracts, backtests, and evaluates rule-based shadow strategies from your personal trading records.[1]
Turtle Trading Rules on GitHub: A direct download of the historically famous, original Turtle trend-following system. [1]
Awesome Stock Trading on GitHub: A structured collection of open-source tools, scripts, and strategy resources for retail and algorithmic traders. [1]
How to Download
You can download any of these repositories using one of the following methods:
Method 1: Clone via Command Line
Open your terminal (macOS/Linux) or Command Prompt (Windows) and enter:
git clone https://github.com/[username]/[repo_name].git
Method 2: Direct Download as ZIP
Navigate to the specific repository link above.
Click the green Code button at the top right of the page.
Select Download ZIP.
Extract the folder to your local machine.
Could you tell me what kind of trading style or asset class you are looking for?
Algorithmic/Python-based rules
Systematic trend following
Machine learning models
Let me know so I can help you target the most relevant codebase.
AI responses may include mistakes. For financial advice, consult a professional. Learn more




I am looking to trade forex, stocks, cryptocurrencies, etc… so need a solid foundation to provide my ai so she can utilise the algorithms and assets to build a profitable automated trading system


12 sites

To provide your AI with a multi-asset foundation across Forex, stocks, and crypto, you must feed it standardized algorithmic structures, unified math frameworks, and robust backtesting engines.
The complete Python blueprint and structural framework below will give your AI agent the exact logic it needs to ingest historical data and execute multi-asset rules.
Multi-Asset Algorithmic Foundation Block
This framework standardizes inputs across disparate asset classes. It maps high-frequency crypto, highly leveraged Forex, and traditional equities into a singular mathematical array that your AI can optimize. [1]
python
import numpy as np
import pandas as pd

class MultiAssetTradingEngine:
    def __init__(self, asset_type, ticker, leverage=1.0):
        self.asset_type = asset_type  # 'FOREX', 'STOCK', or 'CRYPTO'
        self.ticker = ticker
        self.leverage = leverage
        self.position = 0          # Current holdings
        self.cash = 100000.0       # Starting capital ($100k)
        
    def calculate_technical_indicators(self, data):
        \"\"\"Standardizes core math features for the AI's inputs.\"\"\"
        # Log Returns: Standardizes variance across volatile Crypto and stable Forex
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Exponential Moving Average (EMA) for Trend-Following Rules
        data['ema_fast'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # Average True Range (ATR) for Asset-Specific Risk Sizing
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        data['atr'] = ranges.max(axis=1).ewm(span=14, adjust=False).mean()
        
        return data

    def calculate_risk_adjusted_position(self, current_price, atr, account_risk=0.01):
        \"\"\"
        Applies the Turtle Trading Rule for position sizing based on ATR volatility.
        Ensures the AI risks exactly 1% of the portfolio per trade regardless of asset class.
        \"\"\"
        if pd.isna(atr) or atr == 0:
            return 0
        
        # Dollar value of risk (1% of $100k = $1000)
        risk_amount = self.cash * account_risk
        
        # Unit sizing: How many units match the allowed risk distance
        position_size = risk_amount / atr
        
        # Apply leverage modifications depending on the asset class rules
        if self.asset_type == 'FOREX':
            position_size *= self.leverage
            
        return int(position_size)

    def execute_trading_rule(self, data):
        \"\"\"Executes a systematic dual-indicator breakout rule.\"\"\"
        signals = []
        for i in range(len(data)):
            # Trend Check (Fast EMA > Slow EMA) & Momentum Confirmation
            if data['ema_fast'].iloc[i] > data['ema_slow'].iloc[i] and data['log_return'].iloc[i] > 0:
                signals.append(1)  # AI Buy Signal
            elif data['ema_fast'].iloc[i] < data['ema_slow'].iloc[i] and data['log_return'].iloc[i] < 0:
                signals.append(-1) # AI Short/Sell Signal
            else:
                signals.append(0)  # Hold
        data['signal'] = signals
        return data
Use code with caution.

Crucial Strategy Repositories for Your AI
Feed these open-source codebases directly to your AI's context window. They provide production-ready mechanics for multi-asset automated trading:
Vectorbt on GitHub: The highest-performance backtesting tool available in Python. It uses NumPy and Numba to evaluate thousands of trading rules across multi-asset data matrices simultaneously. This is essential for AI optimization loops. [1]
Freqtrade on GitHub: A highly modular, open-source crypto and multi-market algorithmic trading bot. It provides structural examples of how to format configurations, manage live order books, and protect capital via hardcoded stop-losses. [1, 2]
Qlib by Microsoft on GitHub: Microsoft's AI-oriented quantitative investment platform. It bridges data-driven AI models with traditional market rules, making it a perfect foundation blueprint for training trading agents. [1, 2, 3]
Core Structural Rules Your AI Must Follow
To prevent your AI from draining your capital, force it to respect the rigid mathematical guardrails built into the architecture above:
1. Volatility-Based Position Sizing (ATR) [1]
Crypto can drop 20% in an hour; Forex rarely moves 1% a day. Your AI must use the Average True Range (ATR) function from the code above to dynamic-size positions. A high ATR means smaller position units, maintaining equal dollar risk across Bitcoin, Apple stock, or EUR/USD. [1, 2]
2. Log-Return Standardization
Do not allow your AI to look at raw price movements. A $500 move in Bitcoin is normal; a $500 move in a stock is catastrophic. Feeding the AI Log Returns (\\(\\ln(P_t / P_{t-1})\\)) standardizes all data streams into percentage variances. [1]
3. Strict Execution Isolation
Separate your AI's signal generation logic from the order execution engine. Let your AI predict the direction and asset weight, but use hard-coded, non-AI Python rules (like the code snippet above) to handle stop-losses, margin checks, and take-profit targets. [1]
To help you hook this up seamlessly, what environment are you building this in?
An AI Coding Agent workspace (e.g., Cursor, Replit Agent, or Claude Engineer)
A specialized quant sandbox (e.g., QuantConnect, Jupyter Notebooks)
A custom Python backend hosted on a cloud server
Let me know so I can provide the exact code connectors or API integrationsyou need next!
AI responses may include mistakes. For financial advice, consult a professional. Learn more




Custom python backend hosted on cloud server for now but will later be migrated to a personal cloud server


1 site

To deploy your multi-asset algorithmic foundation on a custom Python backend that easily migrates from commercial cloud (AWS, DigitalOcean) to a personal private cloud (e.g., Umbrello, Proxmox, local Linux server), you must build with a headless, decoupled architecture using Docker containerization.
Here is the production-ready infrastructure blueprint and boilerplate code to establish this foundation for your AI.
Step 1: The Multi-Asset Production Engine (engine.py)
This file runs as an isolated execution microservice. It uses ccxt for Crypto, ib_insync (Interactive Brokers) for Forex/Stocks, and vectorbt for backtesting.
python
import os
import ccxt
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, Forex

class ProductionTradingBackend:
    def __init__(self):
        # Environment variables ensure seamless commercial-to-personal cloud migration
        self.db_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/trading")
        self.environment = os.getenv("APP_ENV", "development")
        
        # Initialize Exchanges
        self.binance = ccxt.binance({'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET')})
        self.ib = IB()

    def connect_brokers(self):
        \"\"\"Connects to live execution pipelines based on asset class requirements.\"\"\"
        if self.environment == "production":
            try:
                # Interactive Brokers gateway port (Default 7496 for live, 7497 for paper)
                self.ib.connect('ib-gateway', 7497, clientId=1)
            except Exception as e:
                print(f"Broker connection failed: {e}")

    def fetch_live_data(self, asset_class, symbol, timeframe='1h'):
        \"\"\"Fetches and standardizes live market data arrays for your AI.\"\"\"
        if asset_class == 'CRYPTO':
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')
            
        elif asset_class in ['STOCK', 'FOREX']:
            contract = Stock(symbol, 'SMART', 'USD') if asset_class == 'STOCK' else Forex(symbol)
            bars = self.ib.reqHistoricalData(contract, endDateTime='', durationStr='30 D',
                                            barSizeSetting=timeframe, whatToShow='MIDPOINT', useRTH=True)
            df = pd.DataFrame([(b.date, b.open, b.high, b.low, b.close, b.volume) for b in bars],
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df.set_index('timestamp')

    def execute_ai_order(self, asset_class, symbol, side, dollar_amount):
        \"\"\"Executes rigid order logic, safeguarding your raw funds from AI errors.\"\"\"
        # Calculate current market price to evaluate order sizing safely
        df = self.fetch_live_data(asset_class, symbol)
        current_price = df['close'].iloc[-1]
        units = dollar_amount / current_price

        if asset_class == 'CRYPTO':
            # Safe execution logic for volatile crypto assets
            return self.binance.create_market_order(symbol, side, units)
            
        elif asset_class in ['STOCK', 'FOREX']:
            contract = Stock(symbol, 'SMART', 'USD') if asset_class == 'STOCK' else Forex(symbol)
            order = MarketOrder(side, int(units))
            trade = self.ib.placeOrder(contract, order)
            return trade
Use code with caution.

Step 2: Containerize for Cloud Mobility (Dockerfile)
To make sure your code runs exactly the same on an AWS EC2 instance today and a personal HomeLab Linux server tomorrow, wrap the entire application in Docker. [1]
dockerfile
# Use a lightweight, high-performance math-ready Python image
FROM python:3.11-slim

# Install system dependencies needed for compiling heavy quant libraries (Numba/TA-Lib)
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set up project workspace
WORKDIR /app

# Copy dependency mappings
COPY requirements.txt .

# Install scientific stack and broker interfaces
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase into container
COPY . .

# Expose local API port for your AI agent to communicate with
EXPOSE 8000

# Fire up the engine server
CMD ["python", "main.py"]
Use code with caution.

Step 3: Define Dependencies (requirements.txt)
Pin these specific versions to ensure your math stacks (numpy, pandas) don't break when migrated across server CPU architectures:
text
numpy==1.26.4
pandas==2.2.2
vectorbt==0.26.0
ccxt==4.2.82
ib-insync==0.9.86
pydantic==2.7.1
fastapi==0.111.0
uvicorn==0.29.0
psycopg2-binary==2.9.9
Use code with caution.

Key Infrastructure Rules for AI Backends
Decouple Data Fetching from Trade Execution: Let your AI access historical and current state data via a local REST API endpoint. Never allow the AI agent to connect directly to broker APIs; force it to make an API call to your backend wrapper instead.
State Persistence via PostgreSQL: Ensure your server spins up an internal PostgreSQL DB instance. Log every single feature vector, signal, and decision score generated by your AI. This lets you debug exactly why a trade was placed if the system goes offline during a server migration.
The "Kill-Switch" Environment Variable: Hardcode an EMERGENCY_STOPvariable inside your .env configuration file. If your AI begins behaving unexpectedly, changing this to true should force your Python backend script to instantly drop all active connections and exit loop executions safely.
To help prepare the communication bridge for your AI agent, how do you want the AI to send its trading triggers?
A REST API mapping (e.g., FastAPI endpoints that your AI sends JSON payloads to)
A message broker connection (e.g., RabbitMQ or Redis Pub/Sub for sub-second streaming)
Direct Python script execution orchestration (e.g., Celery cron tasks)
"""

# ----------------------------------------------------------------------
# 2. List of relevant GitHub repository URLs
# ----------------------------------------------------------------------
REPO_URLS = [
    "https://github.com/vectorbt/vectorbt",
    "https://github.com/freqtrade/freqtrade",
    "https://github.com/microsoft/qlib",
    # Add more from the "Top Repositories to Download" list as needed
    # "https://github.com/username/awesome-systematic-trading",
    # "https://github.com/username/vibe-trading-agent",
    # "https://github.com/username/turtle-trading-rules",
    # "https://github.com/username/awesome-stock-trading",
]

# ----------------------------------------------------------------------
# 3. Inject into DMAI's SQLite knowledge base
# ----------------------------------------------------------------------
def inject_knowledge(db_path="data/dmai_knowledge.db"):
    """Insert the trading text as an insight."""
    db = Path(db_path)
    if not db.exists():
        print(f"⚠️ Database {db_path} not found. Ensure you're in the dmai-system directory.")
        return

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_text TEXT,
            entity_type TEXT,
            entities TEXT,
            relationship TEXT,
            source_topic TEXT,
            target_topic TEXT,
            confidence REAL,
            source_title TEXT,
            source_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Insert the full knowledge
    cursor.execute('''
        INSERT INTO insights (
            insight_text, entity_type, entities, relationship,
            source_topic, target_topic, confidence, source_title
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        TRADING_KNOWLEDGE,
        "trading_knowledge",
        "multi-asset,algorithms,backtesting,forex,stocks,crypto",
        "provided_by_user",
        "trading_system",
        "general_knowledge",
        0.95,
        "User Provided Trading Foundation (full text)"
    ))

    # Also insert each repo URL as a separate insight for easy reference
    for url in REPO_URLS:
        cursor.execute('''
            INSERT INTO insights (
                insight_text, entity_type, entities, relationship,
                source_topic, target_topic, confidence, source_title, source_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"GitHub repository for algorithmic trading: {url}",
            "repository",
            "github,trading,algorithm",
            "provided_by_user",
            "trading_system",
            "repository_ingest",
            0.95,
            "User Provided Repo",
            url
        ))

    conn.commit()
    conn.close()
    print(f"✅ Injected trading knowledge and {len(REPO_URLS)} repo references into {db_path}")

# ----------------------------------------------------------------------
# 4. Optionally, send repos to the parallel learner via API
# ----------------------------------------------------------------------
def add_repos_to_parallel_learner(base_url="https://dmai-web.onrender.com", password="Talula.78"):
    """Call /api/knowledge/add-url for each repo."""
    headers = {
        "X-Master-Password": password,
        "Content-Type": "application/json"
    }
    success_count = 0
    for url in REPO_URLS:
        payload = {
            "url": url,
            "reason": "Algorithmic trading repository for DMAI to analyze and integrate"
        }
        try:
            resp = requests.post(
                f"{base_url}/api/knowledge/add-url",
                headers=headers,
                json=payload,
                timeout=10
            )
            if resp.status_code == 200:
                print(f"✅ Added repo to learner: {url}")
                success_count += 1
            else:
                print(f"⚠️ Failed to add {url}: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"⚠️ Error adding {url}: {e}")
    print(f"✅ Successfully added {success_count}/{len(REPO_URLS)} repos to parallel learner")

# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Inject into local SQLite
    inject_knowledge()

    # Step 2: Also inject the full text and repos into the live deployment
    # (Uncomment the line below after verifying credentials and network)
    # add_repos_to_parallel_learner()

    print("\n🎯 Next steps:")
    print("1. Commit and push data/dmai_knowledge.db to sync the knowledge to Render (if you want it there).")
    print("2. Run the parallel learner endpoint manually using the commands in the comments.")
    print("3. Check /api/kaizen to see if DMAI has generated proposals.")
