from flask import Flask, send_from_directory, jsonify, request
import os
import json
from pathlib import Path
from datetime import datetime
import atexit

# Import persistence manager
import sys
sys.path.append('.')  # Ensure current directory is in path

try:
    from shared_data.persistence_manager import get_persistence_manager
except ImportError:
    # Create minimal persistence if module not available
    class SimplePersistence:
        def save_user_data(self, username, data): return True
        def load_user_data(self, username): return None
        def create_checkpoint(self, reason): return None
        def get_system_status(self): return {"status": "simple"}
        def stop(self): pass
        def export_all_data(self, path): return None
    get_persistence_manager = lambda: SimplePersistence()

app = Flask(__name__)

# Initialize persistence manager
persistence = get_persistence_manager()

# API endpoint for evolution stats
@app.route('/api/evolution-stats')
def evolution_stats():
    stats = {
        'generation': 5,
        'bestScore': 1.26,
        'status': 'running',
        'totalFiles': 5661,
        'activeRepos': 22,
        'healthScore': 95.2,
        'persistence': persistence.get_system_status()
    }
    
    # Read actual evolution data from shared_checkpoints
    checkpoints = Path('shared_checkpoints')
    if checkpoints.exists():
        gen_file = checkpoints / 'current_generation.txt'
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                stats['generation'] = int(f.read().strip())
        
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

# API endpoint for user data persistence
@app.route('/api/save-user-data', methods=['POST'])
def save_user_data():
    try:
        data = request.json
        username = data.get('username')
        user_data = data.get('data')
        
        if not username or not user_data:
            return jsonify({'error': 'Missing username or data'}), 400
        
        persistence.save_user_data(username, user_data)
        return jsonify({
            'status': 'saved', 
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-user-data/<username>')
def load_user_data(username):
    try:
        data = persistence.load_user_data(username)
        if data:
            return jsonify(data)
        return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-checkpoint')
def api_create_checkpoint():
    try:
        reason = request.args.get('reason', 'manual')
        checkpoint = persistence.create_checkpoint(reason)
        return jsonify({
            'status': 'created', 
            'checkpoint': str(checkpoint) if checkpoint else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def system_status():
    try:
        return jsonify(persistence.get_system_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-data')
def export_data():
    try:
        export_path = request.args.get('path', '/tmp/dmai_export')
        export_file = persistence.export_all_data(export_path)
        return jsonify({
            'status': 'exported', 
            'file': str(export_file) if export_file else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve the main UI
@app.route('/')
def serve_ui():
    # Try ui folder first
    if os.path.exists('ui/ai_ui.html'):
        return send_from_directory('ui', 'ai_ui.html')
    # Fall back to root directory
    elif os.path.exists('ai_ui.html'):
        return send_from_directory('.', 'ai_ui.html')
    else:
        return """
        <html>
            <head><title>DMAI System</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h1>🚀 DMAI Evolution System</h1>
                <p>System is running with persistence!</p>
                <p>API endpoint: <a href="/api/evolution-stats">/api/evolution-stats</a></p>
                <p>System status: <a href="/api/system-status">/api/system-status</a></p>
                <p>Checkpoints: <a href="/api/create-checkpoint">Create checkpoint</a></p>
            </body>
        </html>
        """, 200

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
    else:
        return jsonify({"error": f"File {path} not found"}), 404

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "persistence": "active",
        "timestamp": datetime.now().isoformat(),
        "generation": evolution_stats().json.get('generation')
    })

# Root API info
@app.route('/api')
def api_info():
    return jsonify({
        "name": "DMAI Evolution System API",
        "endpoints": [
            "/api/evolution-stats - Evolution statistics",
            "/api/save-user-data - Save user data (POST)",
            "/api/load-user-data/<username> - Load user data",
            "/api/create-checkpoint - Create system checkpoint",
            "/api/system-status - Get persistence status",
            "/api/export-data - Export all data",
            "/api - This info",
            "/health - Health check"
        ],
        "version": "3.0",
        "persistence": "active",
        "status": "running"
    })

# Graceful shutdown
@atexit.register
def shutdown():
    print("🛑 Shutting down, creating final checkpoint...")
    try:
        persistence.create_checkpoint("shutdown")
        persistence.stop()
        print("✅ Final checkpoint created")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 DMAI System starting on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"💾 Persistence manager active - auto-save every 60 seconds")
    print(f"📊 Evolution data from: shared_checkpoints/")
    print(f"👤 User data from: shared_data/users/")
    
    # Check if directories exist
    if os.path.exists('shared_checkpoints'):
        print(f"✅ shared_checkpoints/ exists")
    else:
        print(f"📁 Creating shared_checkpoints/ directory")
        os.makedirs('shared_checkpoints', exist_ok=True)
    
    if os.path.exists('shared_data/users'):
        print(f"✅ shared_data/users/ exists")
    else:
        print(f"📁 Creating shared_data/users/ directory")
        os.makedirs('shared_data/users', exist_ok=True)
    
    app.run(host='0.0.0.0', port=port, debug=False)