#!/usr/bin/env python3
"""Simple DMAI Evolution - with status endpoint"""
from flask import Flask, jsonify
import os
from datetime import datetime
import json

app = Flask(__name__)

start_time = datetime.now()
generation = 26
evolution_history = []

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Evolution Online",
        "generation": generation,
        "service": "dmai-final",
        "uptime": str(datetime.now() - start_time)
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/evolve')
def evolve():
    global generation
    generation += 1
    evolution_history.append({
        "timestamp": datetime.now().isoformat(),
        "old_generation": generation - 1,
        "new_generation": generation
    })
    return jsonify({
        "success": True,
        "generation": generation
    })

@app.route('/status')
def status():
    """Detailed status report"""
    return jsonify({
        "current_generation": generation,
        "total_evolutions": len(evolution_history),
        "last_evolution": evolution_history[-1] if evolution_history else None,
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "service": "dmai-final",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/history')
def history():
    """View evolution history"""
    return jsonify({
        "history": evolution_history[-10:]  # Last 10 evolutions
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
