#!/usr/bin/env python3
"""DMAI Cloud Evolution - Status reporter (simplified)"""
import os
import time
import logging
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory state
start_time = datetime.now()
generation = 103  # Match your actual generation
best_score = 1.144
evolution_history = []

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Cloud Evolution",
        "current_generation": generation,
        "best_score": best_score,
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "service": "cloud-evolution"
    })

@app.route('/health')
def health():
    """Health check for Render"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    return jsonify({
        "generation": generation,
        "best_score": best_score,
        "evolution_count": len(evolution_history),
        "last_evolution": evolution_history[-1] if evolution_history else None,
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    })

@app.route('/evolve')
def evolve():
    """Record an evolution event"""
    global generation
    generation += 1
    evolution_history.append({
        "timestamp": datetime.now().isoformat(),
        "new_generation": generation
    })
    return jsonify({
        "success": True,
        "generation": generation
    })

@app.route('/sync/<int:gen>')
def sync(gen):
    """Sync with actual generation from your local system"""
    global generation
    generation = gen
    return jsonify({"synced": True, "generation": generation})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting cloud evolution reporter on port {port}")
    app.run(host='0.0.0.0', port=port)
