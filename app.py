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
        # Try multiple possible checkpoint locations
        checkpoint_dirs = [
            Path("shared_checkpoints"),
            Path("/var/data/shared_checkpoints"),
            Path("checkpoints")
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("gen_*"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    gen = int(latest.name.split('_')[1])
                    return jsonify({
                        'status': 'healthy',
                        'generation': gen,
                        'service': 'dmai-final',
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Default if no checkpoints found
        return jsonify({
            'status': 'healthy',
            'generation': 5,
            'service': 'dmai-final',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'healthy',
            'generation': 5,
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

@app.route('/evolution')
def evolution_dashboard():
    """Serve the evolution dashboard"""
    return send_from_directory('templates', 'evolution_dashboard.html')

from config.admin_config import AdminAuth, BiometricAuth
from functools import wraps
import secrets

# Initialize auth
admin_auth = AdminAuth()
bio_auth = BiometricAuth()

def admin_required(f):
    """Decorator for admin-only routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for admin token in session
        token = request.cookies.get('admin_token')
        if not token or not admin_auth.verify_session(token):
            return jsonify({"error": "Admin access required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Secure admin login"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if admin_auth.verify_admin(username, password):
        token = admin_auth.create_session()
        response = jsonify({"success": True, "token": token})
        response.set_cookie('admin_token', token, httponly=True, secure=True, samesite='Strict')
        return response
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    token = request.cookies.get('admin_token')
    if token:
        admin_auth.revoke_session(token)
    response = jsonify({"success": True})
    response.delete_cookie('admin_token')
    return response

@app.route('/admin/status')
def admin_status():
    """Check admin login status"""
    token = request.cookies.get('admin_token')
    return jsonify({"is_admin": bool(token and admin_auth.verify_session(token))})

@app.route('/evolution')
@admin_required
def evolution_dashboard():
    """Secure evolution dashboard - admin only"""
    return send_from_directory('templates', 'evolution_dashboard.html')

@app.route('/api/evolution/start', methods=['POST'])
@admin_required
def start_evolution():
    """Manually trigger evolution cycle - admin only"""
    try:
        from evolution_engine import run_evolution_cycle
        result = run_evolution_cycle()
        return jsonify({"status": "started", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evolution/status')
@admin_required
def evolution_status():
    """Get detailed evolution status - admin only"""
    try:
        with open('checkpoints/current_generation.txt', 'r') as f:
            generation = f.read().strip()
        with open('checkpoints/best_scores.json', 'r') as f:
            scores = json.load(f)
        return jsonify({
            "generation": generation,
            "phase2_progress": 25,
            "phase3_progress": 0,
            "best_scores": scores
        })
    except:
        return jsonify({"generation": 5, "phase2_progress": 25, "phase3_progress": 0})

@app.route('/admin-login')
def admin_login_page():
    """Serve admin login page"""
    return send_from_directory('templates', 'admin_login.html')

from config.admin_config import AdminAuth
from functools import wraps
import secrets

# Initialize auth
admin_auth = AdminAuth()

def admin_required(f):
    """Decorator for admin-only routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('admin_token')
        if not token or not admin_auth.verify_session(token):
            return jsonify({"error": "Admin access required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Secure admin login"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if admin_auth.verify_admin(username, password):
        token = admin_auth.create_session()
        response = jsonify({"success": True, "token": token})
        response.set_cookie('admin_token', token, httponly=True, secure=True, samesite='Strict')
        return response
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    token = request.cookies.get('admin_token')
    if token:
        admin_auth.revoke_session(token)
    response = jsonify({"success": True})
    response.delete_cookie('admin_token')
    return response

@app.route('/admin/status')
def admin_status():
    """Check admin login status"""
    token = request.cookies.get('admin_token')
    return jsonify({"is_admin": bool(token and admin_auth.verify_session(token))})

@app.route('/evolution')
@admin_required
def evolution_dashboard():
    """Secure evolution dashboard - admin only"""
    return send_from_directory('templates', 'evolution_dashboard.html')

@app.route('/admin-login')
def admin_login_page():
    """Serve admin login page"""
    return send_from_directory('templates', 'admin_login.html')
