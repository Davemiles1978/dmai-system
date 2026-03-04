#!/usr/bin/env python3
"""DMAI Cloud Evolution - Runs 24/7/365"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from flask import Flask, jsonify
import threading

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.evolution_scheduler import EvolutionScheduler

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global evolution state
scheduler = EvolutionScheduler()
evolution_active = True
last_evolution = None

def evolution_worker():
    """Background thread that runs evolution continuously"""
    global last_evolution
    while evolution_active:
        try:
            if scheduler.should_evolve_now():
                logger.info("🧬 Running scheduled evolution in cloud")
                result = scheduler.evolve()
                last_evolution = {
                    "timestamp": datetime.now().isoformat(),
                    "generation": scheduler.engine.generation,
                    "score": result.get("score", 0)
                }
                logger.info(f"✅ Evolution complete: Gen {scheduler.engine.generation}")
            
            # Check every 30 minutes
            time.sleep(1800)
        except Exception as e:
            logger.error(f"Evolution error: {e}")
            time.sleep(300)

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI CLOUD EVOLUTION - 24/7/365",
        "current_generation": scheduler.engine.generation,
        "best_score": scheduler.engine.best_score,
        "last_evolution": last_evolution,
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    })

@app.route('/evolve')
def evolve_now():
    """Manually trigger evolution"""
    result = scheduler.evolve()
    return jsonify({
        "success": True,
        "generation": scheduler.engine.generation,
        "result": result
    })

@app.route('/status')
def status():
    return jsonify({
        "generation": scheduler.engine.generation,
        "best_score": scheduler.engine.best_score,
        "evolution_count": len(scheduler.engine.evolution_history),
        "last_evolution": last_evolution,
        "schedule": scheduler.schedule
    })

@app.route('/history')
def history():
    return jsonify({
        "history": scheduler.engine.evolution_history[-10:]
    })

# Start background evolution thread
start_time = datetime.now()
thread = threading.Thread(target=evolution_worker, daemon=True)
thread.start()
logger.info("🚀 Cloud evolution thread started")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
