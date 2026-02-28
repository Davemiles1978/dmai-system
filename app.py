#!/usr/bin/env python3
"""
DMAI Web Application
"""
from flask import Flask, send_from_directory, jsonify
import os
from pathlib import Path

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_ui():
    """Serve the main UI"""
    ui_path = Path('ui/ai_ui.html')
    if ui_path.exists():
        return send_from_directory('ui', 'ai_ui.html')
    return "UI not found", 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'generation': 5,
        'service': 'dmai-final'
    })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
