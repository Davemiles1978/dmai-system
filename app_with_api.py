from flask import Flask, send_from_directory, jsonify
import os
import json
from pathlib import Path

app = Flask(__name__, static_folder='ui')

# API endpoint for evolution stats
@app.route('/api/evolution-stats')
def evolution_stats():
    stats = {
        'generation': 5,
        'bestScore': 1.26,
        'status': 'running'
    }
    # Read actual evolution data
    checkpoints = Path('checkpoints')
    if checkpoints.exists():
        gen_file = checkpoints / 'current_generation.txt'
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                stats['generation'] = int(f.read().strip())
    return jsonify(stats)

# Serve the main UI
@app.route('/')
def serve_ui():
    return send_from_directory('ui', 'ai_ui.html')

# Serve all other static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
