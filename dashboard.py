#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Simple Flask dashboard to monitor evolution progress
"""
from flask import Flask, jsonify
from pathlib import Path
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "AGI Evolution Dashboard - Coming Soon"

@app.route('/api/progress')
def progress():
    # Read assessment files and return progress
    return jsonify({"phase2": 25, "phase3": 0})

if __name__ == '__main__':
    app.run(port=5000)
