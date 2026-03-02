from flask import Flask, send_from_directory, jsonify, request, send_file
from datetime import datetime, timedelta
import os
import json
import hashlib
import hmac
import secrets
from functools import wraps

app = Flask(__name__, static_folder='static')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_NAME'] = 'dmai_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

@app.route('/')
def serve_ui():
    return send_from_directory('ui', 'ai_ui.html')

@app.route('/health')
def health():
    try:
        with open('checkpoints/current_generation.txt', 'r') as f:
            gen = f.read().strip()
        return jsonify({
            "status": "healthy",
            "generation": gen,
            "service": "dmai-final",
            "timestamp": datetime.now().isoformat()
        })
    except:
        return jsonify({
            "status": "healthy",
            "generation": "unknown",
            "service": "dmai-final",
            "timestamp": datetime.now().isoformat()
        })

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Admin authentication
class AdminAuth:
    def __init__(self):
        self.admin_id = "david"
        self.master_hash = hashlib.sha256("dmai2026".encode()).hexdigest()
        self.active_sessions = {}
        
    def verify_admin(self, username, password):
        if username != self.admin_id:
            return False
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(password_hash, self.master_hash)
    
    def create_session(self):
        token = secrets.token_hex(32)
        self.active_sessions[token] = datetime.now() + timedelta(hours=1)
        return token
    
    def verify_session(self, token):
        if token in self.active_sessions:
            if datetime.now() < self.active_sessions[token]:
                return True
            else:
                del self.active_sessions[token]
        return False
    
    def revoke_session(self, token):
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False

admin_auth = AdminAuth()

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('admin_token')
        if not token or not admin_auth.verify_session(token):
            return jsonify({"error": "Admin access required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['POST'])
def admin_login():
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
    token = request.cookies.get('admin_token')
    if token:
        admin_auth.revoke_session(token)
    response = jsonify({"success": True})
    response.delete_cookie('admin_token')
    return response

@app.route('/admin/status')
def admin_status():
    token = request.cookies.get('admin_token')
    return jsonify({"is_admin": bool(token and admin_auth.verify_session(token))})

@app.route('/evolution')
@admin_required
def evolution_dashboard():
    return send_from_directory('templates', 'evolution_dashboard.html')

@app.route('/api/evolution/start', methods=['POST'])
@admin_required
def start_evolution():
    try:
        from evolution_engine import run_evolution_cycle
        result = run_evolution_cycle()
        return jsonify({"status": "started", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evolution/status')
def evolution_status():
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
    import os
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'admin_login.html')
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            return f.read()
    return "Admin login page not found", 404
    except Exception as e:
        return f"Error loading admin page: {str(e)}", 500

@app.route('/test')
def test():
    return "Test route works!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=False)
