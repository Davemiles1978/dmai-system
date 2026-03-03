#!/usr/bin/env python3
"""DMAI Evolution Service - Clean deployment version"""
import os
import time
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# Simple in-memory state
start_time = datetime.now()
generation = 23
best_score = 1.1444530004071987

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Evolution Online",
        "generation": generation,
        "best_score": best_score,
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/evolve')
def evolve():
    global generation
    generation += 1
    return jsonify({
        "success": True,
        "new_generation": generation,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
