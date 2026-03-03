from flask import Flask, send_from_directory, jsonify, request, session, redirect, url_for, render_template
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import atexit
import secrets
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import base64

# Import persistence manager
import sys
sys.path.append('.')  # Ensure current directory is in path

try:
    from shared_data.persistence_manager import get_persistence_manager
except ImportError:
    class SimplePersistence:
        def save_user_data(self, username, data): return True
        def load_user_data(self, username): return None
        def create_checkpoint(self, reason): return None
        def get_system_status(self): return {"status": "simple"}
        def stop(self): pass
        def export_all_data(self, path): return None
    get_persistence_manager = lambda: SimplePersistence()

app = Flask(__name__)

# Configuration for sessions and security
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_PERMANENT'] = True

# Admin credentials from environment variables
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'changeme'))
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

# Initialize persistence manager
persistence = get_persistence_manager()

# ========== PERSISTENT BIOMETRIC STORAGE ==========
BIOMETRIC_DB_PATH = Path('shared_data/biometric_credentials.json')

def load_biometric_credentials():
    """Load credentials from disk"""
    if BIOMETRIC_DB_PATH.exists():
        try:
            with open(BIOMETRIC_DB_PATH, 'r') as f:
                data = json.load(f)
                # Convert base64 strings back to bytes for the public_key
                for user_id, cred in data.items():
                    if 'public_key' in cred and isinstance(cred['public_key'], str):
                        cred['public_key'] = base64.b64decode(cred['public_key'])
                return data
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return {}
    return {}

def save_biometric_credentials(credentials):
    """Save credentials to disk with proper encoding"""
    try:
        BIOMETRIC_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Make a copy for serialization
        serializable = {}
        for user_id, cred in credentials.items():
            serializable[user_id] = cred.copy()
            # Convert bytes to base64 string for JSON
            if 'public_key' in serializable[user_id] and isinstance(serializable[user_id]['public_key'], bytes):
                serializable[user_id]['public_key'] = base64.b64encode(serializable[user_id]['public_key']).decode('ascii')
        
        with open(BIOMETRIC_DB_PATH, 'w') as f:
            json.dump(serializable, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False

# Load credentials at startup
user_credentials = load_biometric_credentials()
webauthn_challenges = {}  # Challenges can stay in memory (they're short-lived)

print(f"🔐 Loaded {len(user_credentials)} stored biometric credentials")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

# ========== WEBAUTHN SETUP ==========
from webauthn import generate_registration_options, verify_registration_response
from webauthn import generate_authentication_options, verify_authentication_response
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    UserVerificationRequirement,
    AuthenticatorAttachment,
)
from typing import Dict, Any

class WebAuthnManager:
    def __init__(self):
        self.rp_id = os.environ.get('RP_ID', 'localhost')
        self.rp_name = os.environ.get('RP_NAME', 'DMAI Evolution System')
        self.origin = os.environ.get('ORIGIN', 'http://localhost:10000')
    
    def generate_registration_options(self, username: str, user_id: str):
        options = generate_registration_options(
            rp_id=self.rp_id,
            rp_name=self.rp_name,
            user_id=user_id.encode('utf-8'),
            user_name=username,
            user_display_name=username,
            authenticator_selection=AuthenticatorSelectionCriteria(
                authenticator_attachment=AuthenticatorAttachment.PLATFORM,
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
        )
        webauthn_challenges[user_id] = options.challenge
        return options
    
    def verify_registration(self, user_id: str, response: Dict[str, Any]):
        challenge = webauthn_challenges.pop(user_id, None)
        if not challenge:
            return False, "Challenge not found"
        try:
            verification = verify_registration_response(
                credential=response,
                expected_challenge=challenge,
                expected_rp_id=self.rp_id,
                expected_origin=self.origin,
            )
            
            # Store in persistent storage
            user_credentials[user_id] = {
                'credential_id': response['id'],
                'public_key': verification.credential_public_key,  # This is bytes
                'sign_count': verification.sign_count,
                'transports': response.get('transports', []),
                'username': user_id,
                'created_at': datetime.now().isoformat()
            }
            # Save to disk immediately
            if save_biometric_credentials(user_credentials):
                return True, "Registration successful"
            else:
                return False, "Failed to save credentials"
        except Exception as e:
            print(f"Verification error: {e}")
            return False, str(e)
    
    def generate_authentication_options(self, user_id: str):
        credential = user_credentials.get(user_id)
        if not credential:
            return None
        options = generate_authentication_options(
            rp_id=self.rp_id,
            allow_credentials=[credential['credential_id']],
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        webauthn_challenges[user_id] = options.challenge
        return options
    
    def verify_authentication(self, user_id: str, response: Dict[str, Any]):
        challenge = webauthn_challenges.pop(user_id, None)
        credential = user_credentials.get(user_id)
        if not challenge or not credential:
            return False, "Invalid session"
        try:
            verification = verify_authentication_response(
                credential=response,
                expected_challenge=challenge,
                expected_rp_id=self.rp_id,
                expected_origin=self.origin,
                credential_public_key=credential['public_key'],
                credential_current_sign_count=credential['sign_count'],
            )
            credential['sign_count'] = verification.new_sign_count
            # Update stored credential with new sign count
            save_biometric_credentials(user_credentials)
            return True, "Authentication successful"
        except Exception as e:
            print(f"Auth error: {e}")
            return False, str(e)

webauthn_manager = WebAuthnManager()

# ========== API ENDPOINTS ==========
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
        
        latest_gen = stats['generation']
        gen_summary = checkpoints / f'generation_{latest_gen}_summary.json'
        if gen_summary.exists():
            with open(gen_summary, 'r') as f:
                summary = json.load(f)
                stats['lastCycleSummary'] = summary
    
    return jsonify(stats)

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
            "/admin-login - Admin login page",
            "/forgot-password - Password reset",
            "/api - This info",
            "/health - Health check"
        ],
        "version": "3.1",
        "persistence": "active",
        "status": "running"
    })

# ========== AUTH ROUTES ==========
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session.permanent = True
            session['admin_logged_in'] = True
            session['username'] = username
            return redirect('/admin-dashboard')
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    
    return render_template('admin_login.html', error=None)

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    return jsonify({
        'status': 'authenticated',
        'message': 'Welcome to admin dashboard',
        'username': session.get('username'),
        'stats': evolution_stats().json
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/admin-login')

# ========== BIOMETRIC API ENDPOINTS ==========
@app.route('/api/biometric/register/begin', methods=['POST'])
def biometric_register_begin():
    try:
        data = request.get_json()
        username = data.get('username')
        if not username:
            return jsonify({'error': 'Username required'}), 400
        if username != ADMIN_USERNAME:
            return jsonify({'error': 'Unauthorized'}), 401
        user_id = base64.b64encode(username.encode()).decode()
        options = webauthn_manager.generate_registration_options(username, user_id)
        return jsonify({
            'challenge': base64.b64encode(options.challenge).decode(),
            'rp': {'id': options.rp.id, 'name': options.rp.name},
            'user': {
                'id': base64.b64encode(options.user.id).decode(),
                'name': options.user.name,
                'displayName': options.user.display_name,
            },
            'pubKeyCredParams': [{'alg': param.alg.value, 'type': param.type} for param in options.pub_key_cred_params],
            'authenticatorSelection': {'authenticatorAttachment': 'platform', 'userVerification': 'preferred'},
            'timeout': options.timeout,
            'attestation': options.attestation.value,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/register/complete', methods=['POST'])
def biometric_register_complete():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        credential = data.get('credential')
        if not user_id or not credential:
            return jsonify({'error': 'Missing data'}), 400
        success, message = webauthn_manager.verify_registration(user_id, credential)
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/login/begin', methods=['POST'])
def biometric_login_begin():
    try:
        data = request.get_json()
        username = data.get('username')
        if not username:
            return jsonify({'error': 'Username required'}), 400
        user_id = base64.b64encode(username.encode()).decode()
        options = webauthn_manager.generate_authentication_options(user_id)
        if not options:
            return jsonify({'error': 'No biometric credential found'}), 404
        return jsonify({
            'challenge': base64.b64encode(options.challenge).decode(),
            'allowCredentials': [{'id': cred.id, 'type': cred.type, 'transports': cred.transports} for cred in options.allow_credentials],
            'timeout': options.timeout,
            'userVerification': options.user_verification.value,
            'rpId': options.rp_id,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/login/complete', methods=['POST'])
def biometric_login_complete():
    try:
        data = request.get_json()
        username = data.get('username')
        credential = data.get('credential')
        if not username or not credential:
            return jsonify({'error': 'Missing data'}), 400
        user_id = base64.b64encode(username.encode()).decode()
        success, message = webauthn_manager.verify_authentication(user_id, credential)
        if success:
            session.permanent = True
            session['admin_logged_in'] = True
            session['username'] = username
            session['auth_method'] = 'biometric'
            return jsonify({'success': True, 'message': message, 'redirect': '/'})
        else:
            return jsonify({'success': False, 'message': message}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== BIOMETRIC UI PAGES ==========
@app.route('/biometric-setup')
def biometric_setup():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biometric Setup - DMAI System</title>
        <style>
            body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); width: 400px; }
            h2 { color: #333; margin-bottom: 10px; }
            .subtitle { color: #666; margin-bottom: 30px; }
            button { width: 100%; padding: 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 10px 0; }
            .message { padding: 12px; border-radius: 8px; margin: 10px 0; display: none; }
            .success { background: #efe; color: #3c3; border: 1px solid #cfc; }
            .error { background: #fee; color: #c33; border: 1px solid #fcc; }
            input { width: 100%; padding: 12px; margin: 10px 0; border: 2px solid #e0e0e0; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔐 Biometric Setup</h2>
            <div id="message" class="message"></div>
            <input type="text" id="username" value="Baphomet_78">
            <button onclick="startBiometricSetup()">Register Biometric</button>
            <p style="margin-top:20px; font-size:12px; color:#666;">Your biometric will be saved permanently</p>
        </div>
        <script>
        async function startBiometricSetup() {
            const msg = document.getElementById('message');
            msg.style.display = 'none';
            
            const username = document.getElementById('username').value;
            try {
                const r1 = await fetch('/api/biometric/register/begin', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({username:username})
                });
                const o = await r1.json();
                
                o.challenge = Uint8Array.from(atob(o.challenge), c=>c.charCodeAt(0));
                o.user.id = Uint8Array.from(atob(o.user.id), c=>c.charCodeAt(0));
                
                const c = await navigator.credentials.create({publicKey:o});
                
                const cr = {
                    id: c.id,
                    rawId: btoa(String.fromCharCode(...new Uint8Array(c.rawId))),
                    type: c.type,
                    transports: c.response.getTransports ? c.response.getTransports() : [],
                    response: {
                        attestationObject: btoa(String.fromCharCode(...new Uint8Array(c.response.attestationObject))),
                        clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(c.response.clientDataJSON)))
                    }
                };
                
                const r2 = await fetch('/api/biometric/register/complete', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({userId: btoa(username), credential: cr})
                });
                
                const result = await r2.json();
                
                if(result.success) {
                    msg.textContent = 'Success! Biometric saved permanently.';
                    msg.className = 'message success';
                    msg.style.display = 'block';
                } else {
                    msg.textContent = result.message || 'Registration failed';
                    msg.className = 'message error';
                    msg.style.display = 'block';
                }
            } catch(e) { 
                msg.textContent = 'Error: ' + e.message;
                msg.className = 'message error';
                msg.style.display = 'block';
                console.error(e);
            }
        }
        </script>
    </body>
    </html>
    '''

@app.route('/biometric-login')
def biometric_login_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biometric Login - DMAI System</title>
        <style>
            body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); width: 350px; text-align: center; }
            h2 { color: #333; margin-bottom: 10px; }
            button { width: 100%; padding: 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; }
            .message { padding: 12px; border-radius: 8px; margin: 10px 0; display: none; }
            .success { background: #efe; color: #3c3; }
            .error { background: #fee; color: #c33; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Biometric Login</h2>
            <div id="message" class="message"></div>
            <button onclick="biometricLogin()">Authenticate</button>
        </div>
        <script>
        async function biometricLogin() {
            const msg = document.getElementById('message');
            msg.style.display = 'none';
            
            const username = 'Baphomet_78';
            try {
                const r1 = await fetch('/api/biometric/login/begin', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({username:username})
                });
                const o = await r1.json();
                
                if(!r1.ok) {
                    msg.textContent = o.error || 'Login failed';
                    msg.className = 'message error';
                    msg.style.display = 'block';
                    return;
                }
                
                o.challenge = Uint8Array.from(atob(o.challenge), c=>c.charCodeAt(0));
                o.allowCredentials.forEach(cred => {
                    cred.id = Uint8Array.from(atob(cred.id), c=>c.charCodeAt(0));
                });
                
                const c = await navigator.credentials.get({publicKey:o});
                
                const cr = {
                    id: c.id,
                    rawId: btoa(String.fromCharCode(...new Uint8Array(c.rawId))),
                    type: c.type,
                    response: {
                        authenticatorData: btoa(String.fromCharCode(...new Uint8Array(c.response.authenticatorData))),
                        clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(c.response.clientDataJSON))),
                        signature: btoa(String.fromCharCode(...new Uint8Array(c.response.signature))),
                        userHandle: c.response.userHandle ? btoa(String.fromCharCode(...new Uint8Array(c.response.userHandle))) : null
                    }
                };
                
                const r2 = await fetch('/api/biometric/login/complete', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({username:username, credential:cr})
                });
                
                const result = await r2.json();
                
                if(result.success) {
                    msg.textContent = 'Success! Redirecting...';
                    msg.className = 'message success';
                    msg.style.display = 'block';
                    setTimeout(() => { window.location.href = result.redirect; }, 1000);
                } else {
                    msg.textContent = result.message || 'Login failed';
                    msg.className = 'message error';
                    msg.style.display = 'block';
                }
            } catch(e) { 
                msg.textContent = 'Error: ' + e.message;
                msg.className = 'message error';
                msg.style.display = 'block';
                console.error(e);
            }
        }
        </script>
    </body>
    </html>
    '''

# ========== DEBUG BIOMETRIC ENDPOINTS ==========
@app.route('/api/biometric/debug-login', methods=['POST'])
def debug_login():
    """Debug endpoint to test login flow"""
    try:
        data = request.get_json()
        username = data.get('username')
        print(f"🔍 DEBUG - Login attempt for: {username}")
        
        if not username:
            return jsonify({'error': 'Username required'}), 400
            
        user_id = base64.b64encode(username.encode()).decode()
        print(f"🔍 DEBUG - User ID: {user_id}")
        print(f"🔍 DEBUG - Stored users: {list(user_credentials.keys())}")
        
        if user_id in user_credentials:
            return jsonify({
                'found': True, 
                'user_id': user_id,
                'has_credential': True
            })
        else:
            print(f"🔍 DEBUG - No credential for {username}")
            return jsonify({'found': False, 'user_id': user_id}), 404
            
    except Exception as e:
        print(f"🔍 DEBUG - Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def fix_base64_padding(s):
    """Add proper base64 padding"""
    s = s.rstrip('=')
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += '=' * padding
    return s

@app.route('/api/biometric/login/begin-fixed', methods=['POST'])
def biometric_login_begin_fixed():
    """Fixed login begin with proper padding handling"""
    try:
        data = request.get_json()
        username = data.get('username')
        print(f"🔐 FIXED - Login begin for: {username}")
        
        if not username:
            return jsonify({'error': 'Username required'}), 400
            
        user_id = base64.b64encode(username.encode()).decode()
        print(f"🔐 FIXED - User ID: {user_id}")
        print(f"🔐 FIXED - Stored users: {list(user_credentials.keys())}")
        
        if user_id not in user_credentials:
            print(f"🔐 FIXED - No credential found")
            return jsonify({'error': 'No biometric credential found'}), 404
            
        credential = user_credentials[user_id]
        print(f"🔐 FIXED - Raw credential_id: {credential['credential_id']}")
        
        # Fix padding
        fixed_cred_id = fix_base64_padding(credential['credential_id'])
        print(f"🔐 FIXED - Fixed credential_id: {fixed_cred_id}")
        
        # Decode to bytes
        try:
            cred_id_bytes = base64.b64decode(fixed_cred_id)
            print(f"🔐 FIXED - Successfully decoded to {len(cred_id_bytes)} bytes")
        except Exception as e:
            print(f"🔐 FIXED - Decode error: {e}")
            return jsonify({'error': f'Failed to decode credential: {e}'}), 500
        
        # Generate options
        options = generate_authentication_options(
            rp_id=webauthn_manager.rp_id,
            allow_credentials=[{
                'type': 'public-key',
                'id': cred_id_bytes,
                'transports': credential.get('transports', [])
            }],
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        
        # Store challenge
        webauthn_challenges[user_id] = options.challenge
        
        response = {
            'challenge': base64.b64encode(options.challenge).decode(),
            'allowCredentials': [
                {
                    'id': fixed_cred_id,
                    'type': 'public-key',
                    'transports': credential.get('transports', [])
                }
            ],
            'timeout': options.timeout,
            'userVerification': options.user_verification.value,
            'rpId': options.rp_id,
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"🔐 FIXED - Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/biometric-login-fixed')
def biometric_login_fixed():
    """Fixed login page using the fixed endpoint"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biometric Login - DMAI System</title>
        <style>
            body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); width: 350px; text-align: center; }
            h2 { color: #333; margin-bottom: 10px; }
            button { width: 100%; padding: 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; margin: 10px 0; }
            .message { padding: 12px; border-radius: 8px; margin: 10px 0; display: none; }
            .success { background: #efe; color: #3c3; }
            .error { background: #fee; color: #c33; }
            .debug { font-family: monospace; font-size: 12px; background: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px; display: none; max-height: 200px; overflow-y: auto; text-align: left; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Biometric Login (Fixed)</h2>
            <div id="message" class="message"></div>
            <button onclick="biometricLogin()">Authenticate</button>
            <div id="debug" class="debug"></div>
        </div>
        <script>
        function debug(t) { 
            const d = document.getElementById('debug'); 
            d.innerHTML += '<div>' + new Date().toLocaleTimeString() + ': ' + t + '</div>'; 
            d.style.display = 'block'; 
        }
        
        async function biometricLogin() {
            const msg = document.getElementById('message');
            msg.style.display = 'none';
            
            const username = 'Baphomet_78';
            debug('Starting login for ' + username);
            
            try {
                debug('Calling /api/biometric/login/begin-fixed');
                const r1 = await fetch('/api/biometric/login/begin-fixed', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({username:username})
                });
                
                const o = await r1.json();
                debug('Status: ' + r1.status);
                
                if (!r1.ok) {
                    msg.textContent = o.error || 'Login failed';
                    msg.className = 'message error';
                    msg.style.display = 'block';
                    debug('Error: ' + JSON.stringify(o));
                    return;
                }
                
                debug('Options received, converting challenge...');
                o.challenge = Uint8Array.from(atob(o.challenge), c => c.charCodeAt(0));
                o.allowCredentials.forEach(cred => {
                    cred.id = Uint8Array.from(atob(cred.id), c => c.charCodeAt(0));
                });
                
                debug('Requesting biometric authentication...');
                const c = await navigator.credentials.get({publicKey: o});
                debug('Biometric credential received');
                
                const cr = {
                    id: c.id,
                    rawId: btoa(String.fromCharCode(...new Uint8Array(c.rawId))),
                    type: c.type,
                    response: {
                        authenticatorData: btoa(String.fromCharCode(...new Uint8Array(c.response.authenticatorData))),
                        clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(c.response.clientDataJSON))),
                        signature: btoa(String.fromCharCode(...new Uint8Array(c.response.signature))),
                        userHandle: c.response.userHandle ? btoa(String.fromCharCode(...new Uint8Array(c.response.userHandle))) : null
                    }
                };
                
                debug('Calling /api/biometric/login/complete');
                const r2 = await fetch('/api/biometric/login/complete', {
                    method:'POST', 
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({username:username, credential:cr})
                });
                
                const result = await r2.json();
                debug('Response: ' + JSON.stringify(result));
                
                if (result.success) {
                    msg.textContent = 'Success! Redirecting...';
                    msg.className = 'message success';
                    msg.style.display = 'block';
                    setTimeout(() => { window.location.href = result.redirect; }, 1000);
                } else {
                    msg.textContent = result.message || 'Login failed';
                    msg.className = 'message error';
                    msg.style.display = 'block';
                }
            } catch (e) {
                debug('Error: ' + e.toString());
                msg.textContent = 'Error: ' + e.message;
                msg.className = 'message error';
                msg.style.display = 'block';
                console.error(e);
            }
        }
        </script>
    </body>
    </html>
    '''

# ========== DEBUG ENDPOINTS ==========
@app.route('/debug-credential-details')
def debug_credential_details():
    """Show raw credential data for debugging"""
    if BIOMETRIC_DB_PATH.exists():
        with open(BIOMETRIC_DB_PATH, 'r') as f:
            raw_data = json.load(f)
        return jsonify({
            'raw_data': raw_data,
            'file_exists': True,
            'file_path': str(BIOMETRIC_DB_PATH)
        })
    return jsonify({'error': 'No credential file'}), 404

@app.route('/debug-env')
def debug_env():
    return jsonify({
        'admin_username': ADMIN_USERNAME,
        'admin_email': ADMIN_EMAIL,
        'admin_username_from_env': os.environ.get('ADMIN_USERNAME', 'not set'),
        'admin_password_hash_exists': ADMIN_PASSWORD_HASH is not None,
        'session_exists': session.get('admin_logged_in', False)
    })

@app.route('/debug-password', methods=['POST'])
def debug_password():
    try:
        data = request.get_json()
        test_password = data.get('password', '')
        is_correct = check_password_hash(ADMIN_PASSWORD_HASH, test_password)
        return jsonify({'password_matches': is_correct, 'username': ADMIN_USERNAME})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list-routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            routes.append({
                'endpoint': rule.endpoint,
                'methods': [m for m in rule.methods if m not in ['HEAD', 'OPTIONS']],
                'path': str(rule)
            })
    return jsonify(routes)

@app.route('/debug-credentials')
def debug_credentials():
    return jsonify({
        'stored_users': list(user_credentials.keys()),
        'credential_count': len(user_credentials),
        'storage_path': str(BIOMETRIC_DB_PATH),
        'storage_exists': BIOMETRIC_DB_PATH.exists()
    })

# ========== GRACEFUL SHUTDOWN ==========
@atexit.register
def shutdown():
    print("🛑 Shutting down, saving biometric credentials...")
    save_biometric_credentials(user_credentials)
    print(f"✅ Saved {len(user_credentials)} biometric credentials")
    try:
        persistence.create_checkpoint("shutdown")
        persistence.stop()
        print("✅ Final checkpoint created")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")

# ========== MAIN ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 DMAI System starting on port {port}")
    print(f"🔐 Admin login: /admin-login")
    print(f"🔑 Biometric login: /biometric-login")
    print(f"🔑 Fixed biometric: /biometric-login-fixed")
    print(f"🔍 Debug: /debug-credentials, /debug-credential-details, /list-routes")
    print(f"💾 Biometric storage: {BIOMETRIC_DB_PATH}")
    print(f"📊 Loaded {len(user_credentials)} stored credentials")
    
    os.makedirs('shared_checkpoints', exist_ok=True)
    os.makedirs('shared_data/users', exist_ok=True)
    os.makedirs('shared_data', exist_ok=True)
    
    app.run(host='0.0.0.0', port=port, debug=False)
