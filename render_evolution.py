#!/usr/bin/env python3
"""Render cloud service for DMAI evolution"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from flask import Flask, jsonify

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_core.core_brain import DMAIBrain
from ai_core.evolution_engine import EvolutionEngine

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DMAI
brain = DMAIBrain()
engine = EvolutionEngine()

@app.route('/')
def home():
    return jsonify({
        "status": "DMAI Evolution Service Running",
        "generation": engine.generation,
        "best_score": engine.best_score,
        "last_evolution": engine.evolution_history[-1] if engine.evolution_history else None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/evolve')
def evolve():
    """Trigger evolution cycle"""
    try:
        # Get metrics
        metrics = {
            "accuracy": brain.calculate_accuracy(),
            "speed": brain.calculate_speed(),
            "knowledge_gaps": len(brain.knowledge_base.get("learned", [])),
        }
        
        brain_state = {
            "capabilities": brain.capabilities,
            "knowledge": brain.knowledge_base
        }
        
        # Run evolution
        result = engine.evolve(brain_state, metrics)
        
        return jsonify({
            "success": True,
            "generation": engine.generation,
            "result": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "generation": engine.generation,
        "best_score": engine.best_score,
        "total_evolutions": len(engine.evolution_history),
        "capabilities": {k: v.get("level", 0) for k, v in brain.capabilities.items()}
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
