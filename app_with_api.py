from flask import Flask, send_from_directory, jsonify
import os
import json
from pathlib import Path

app = Flask(__name__)  # Removed static_folder='ui' for flexibility

# API endpoint for evolution stats
@app.route('/api/evolution-stats')
def evolution_stats():
    stats = {
        'generation': 5,
        'bestScore': 1.26,
        'status': 'running',
        'totalFiles': 5661,
        'activeRepos': 22,
        'healthScore': 95.2
    }
    
    # Read actual evolution data
    checkpoints = Path('checkpoints')
    if checkpoints.exists():
        # Get current generation
        gen_file = checkpoints / 'current_generation.txt'
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                stats['generation'] = int(f.read().strip())
        
        # Get best scores
        best_scores_file = checkpoints / 'best_scores.json'
        if best_scores_file.exists():
            with open(best_scores_file, 'r') as f:
                stats['bestScores'] = json.load(f)
        
        # Get latest generation summary
        latest_gen = stats['generation']
        gen_summary = checkpoints / f'generation_{latest_gen}_summary.json'
        if gen_summary.exists():
            with open(gen_summary, 'r') as f:
                summary = json.load(f)
                stats['lastCycleSummary'] = summary
    
    return jsonify(stats)

# Serve the main UI - try multiple possible locations
@app.route('/')
def serve_ui():
    # Try ui folder first (your current structure)
    if os.path.exists('ui/ai_ui.html'):
        return send_from_directory('ui', 'ai_ui.html')
    # Fall back to root directory
    elif os.path.exists('ai_ui.html'):
        return send_from_directory('.', 'ai_ui.html')
    # Check AI-Evolution-System specific path
    elif os.path.exists('AI-Evolution-System/ui/ai_ui.html'):
        return send_from_directory('AI-Evolution-System/ui', 'ai_ui.html')
    else:
        return """
        <html>
            <head><title>DMAI System</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h1>🚀 DMAI Evolution System</h1>
                <p>System is running but UI file not found. Checked locations:</p>
                <ul>
                    <li>ui/ai_ui.html</li>
                    <li>ai_ui.html</li>
                    <li>AI-Evolution-System/ui/ai_ui.html</li>
                </ul>
                <p>API endpoint: <a href="/api/evolution-stats">/api/evolution-stats</a> is working</p>
                <p>Health check: <a href="/health">/health</a></p>
            </body>
        </html>
        """, 200  # Return 200 so Render doesn't show error page

# Serve static files (CSS, JS, images, etc.)
@app.route('/<path:path>')
def serve_static(path):
    # Skip API routes
    if path.startswith('api/'):
        return jsonify({"error": "API endpoint not found"}), 404
    
    # Try ui folder first
    if os.path.exists(os.path.join('ui', path)):
        return send_from_directory('ui', path)
    # Then try root
    elif os.path.exists(path):
        return send_from_directory('.', path)
    # Then try AI-Evolution-System/ui
    elif os.path.exists(os.path.join('AI-Evolution-System/ui', path)):
        return send_from_directory('AI-Evolution-System/ui', path)
    else:
        return jsonify({"error": f"File {path} not found"}), 404

# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "generation": evolution_stats().json.get('generation'),
        "timestamp": str(Path('checkpoints/current_generation.txt').stat().st_mtime) if Path('checkpoints/current_generation.txt').exists() else None
    })

# Root API info
@app.route('/api')
def api_info():
    return jsonify({
        "name": "DMAI Evolution System API",
        "endpoints": [
            "/api/evolution-stats - Evolution statistics",
            "/api - This info",
            "/health - Health check"
        ],
        "version": "2.0",
        "status": "running"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 DMAI System starting on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📄 Files in current directory: {os.listdir('.')}")
    if os.path.exists('ui'):
        print(f"📁 ui folder contents: {os.listdir('ui')}")
    app.run(host='0.0.0.0', port=port, debug=False)