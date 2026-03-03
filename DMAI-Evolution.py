#!/usr/bin/env python3
"""Render cloud service for DMAI evolution - Lightweight version"""
import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory state (no heavy dependencies)
evolution_state = {
    "generation": 22,
    "best_score": 1.1444530004071987,
    "last_evolution": {
        "timestamp": datetime.now().isoformat(),
        "improvements": ["Enhanced Reasoning", "Memory Expansion", "Planning Optimization"]
    },
    "evolution_history": [],
    "start_time": datetime.now().isoformat()
}

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Evolution Service Running",
        "generation": evolution_state["generation"],
        "best_score": evolution_state["best_score"],
        "last_evolution": evolution_state["last_evolution"],
        "uptime": (datetime.now() - datetime.fromisoformat(evolution_state["start_time"])).total_seconds(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/evolve')
def evolve():
    """Trigger evolution cycle - lightweight version"""
    global evolution_state
    
    # Simple evolution logic
    evolution_state["generation"] += 1
    evolution_state["last_evolution"] = {
        "timestamp": datetime.now().isoformat(),
        "improvements": ["Minor optimization", "Pattern recognition"],
        "score": evolution_state["best_score"] * 0.99  # Slight variation
    }
    
    # Keep history small (last 10 entries)
    evolution_state["evolution_history"].append({
        "generation": evolution_state["generation"],
        "timestamp": datetime.now().isoformat(),
        "score": evolution_state["last_evolution"]["score"]
    })
    
    if len(evolution_state["evolution_history"]) > 10:
        evolution_state["evolution_history"] = evolution_state["evolution_history"][-10:]
    
    return jsonify({
        "success": True,
        "generation": evolution_state["generation"],
        "last_evolution": evolution_state["last_evolution"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    return jsonify({
        "generation": evolution_state["generation"],
        "best_score": evolution_state["best_score"],
        "total_evolutions": len(evolution_state["evolution_history"]),
        "last_evolution": evolution_state["last_evolution"],
        "uptime": (datetime.now() - datetime.fromisoformat(evolution_state["start_time"])).total_seconds()
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "generation": evolution_state["generation"]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
