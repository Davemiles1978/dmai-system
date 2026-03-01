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
    """Health check endpoint with real generation"""
    try:
        # Try to get real generation from checkpoints
        checkpoint_dir = Path("shared_checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("gen_*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                gen = int(latest.name.split('_')[1])
            else:
                gen = 5
        else:
            gen = 5
    except:
        gen = 5
    
    return jsonify({
        'status': 'healthy',
        'generation': gen,
        'service': 'dmai-final',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
