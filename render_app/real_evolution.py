#!/usr/bin/env python3
"""Cloud service that triggers REAL DMAI evolution"""
from flask import Flask, jsonify
import os
import subprocess
import json
from datetime import datetime
import threading
import time

app = Flask(__name__)

start_time = datetime.now()
evolution_log = []

def run_evolution():
    """Actually run the evolution engine"""
    try:
        # This would trigger your real evolution
        result = subprocess.run(
            ['python', 'ai_core/evolution_scheduler.py', '--now'],
            capture_output=True,
            text=True,
            cwd='/Users/davidmiles/Desktop/AI-Evolution-System'  # Adjust path
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout[-200:],  # Last 200 chars
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Real Evolution Service",
        "generation": get_current_gen(),
        "service": "dmai-final",
        "uptime": str(datetime.now() - start_time)
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/evolve')
def evolve():
    """Trigger REAL evolution"""
    result = run_evolution()
    evolution_log.append(result)
    return jsonify({
        "success": result.get("success", False),
        "generation": get_current_gen(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    return jsonify({
        "current_generation": get_current_gen(),
        "total_evolutions": len(evolution_log),
        "last_evolution": evolution_log[-1] if evolution_log else None,
        "uptime_seconds": (datetime.now() - start_time).total_seconds()
    })

def get_current_gen():
    """Read actual generation from file"""
    try:
        with open('ai_core/evolution/current_generation.txt', 'r') as f:
            return int(f.read().strip())
    except:
        return 3  # From your file

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
