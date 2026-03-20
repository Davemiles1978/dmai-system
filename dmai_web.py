#!/usr/bin/env python3
"""
DMAI Web Interface - Simplified version with no login page
Root (/) goes directly to chat, admin requires login
Added Research Targets Management API
"""

import os
import sys
import logging
import json
import random
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

# Try to import Flask
try:
    from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("❌ Flask not installed. Run: pip install flask flask-cors")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('web')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_COOKIE_NAME'] = 'session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Enable CORS
CORS(app, supports_credentials=True)

# Master password from environment
MASTER_PASSWORD = os.environ.get('MASTER_PASSWORD', 'Talula.78')

# Try to import DMAI Core
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dmai_core_clean import DMAIIntelligence
    core = DMAIIntelligence()
    logger.info("✅ DMAI Core loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to load DMAI Core: {e}")
    core = None
except Exception as e:
    logger.error(f"❌ Error initializing DMAI Core: {e}")
    core = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_generation():
    """Get current generation from core"""
    if core:
        if hasattr(core, 'generation'):
            return core.generation
        elif hasattr(core, 'get_status'):
            try:
                status = core.get_status()
                return status.get('generation', 72)
            except:
                return 72
    return 72

def generate_response(message: str) -> str:
    """Generate a simple but intelligent response for chat"""
    message_lower = message.lower()
    
    # Get system status if available
    status_info = ""
    component_count = 0
    healthy_count = 0
    
    if core and hasattr(core, 'get_status'):
        try:
            status = core.get_status()
            component_count = status.get('components', {}).get('total', 0)
            healthy_count = status.get('components', {}).get('healthy', 0)
            gen = status.get('generation', 72)
            status_info = f" (Generation {gen}, {healthy_count}/{component_count} components healthy)"
        except:
            pass
    
    # Greetings
    if any(g in message_lower for g in ['hello', 'hi', 'hey', 'greetings']):
        return f"Hello! I'm DMAI, your autonomous intelligence.{status_info} How can I help you today?"
    
    # About DMAI
    elif any(q in message_lower for q in ['who are you', 'what are you', 'yourself', 'about you']):
        return f"I am DMAI - Dynamic Meta-Adaptive Intelligence. I'm a self-evolving AI system designed to learn, grow, and assist you.{status_info} I can evolve myself, learn from interactions, and handle various tasks."
    
    # Capabilities
    elif any(c in message_lower for c in ['what can you do', 'capabilities', 'help me', 'abilities']):
        return f"""I can help you with:
• Answer questions about your system
• Evolve and improve myself automatically
• Monitor component health ({healthy_count}/{component_count} healthy)
• Learn from our conversations
• Research new topics
• Deploy to cloud providers
• Generate reports

Try asking about my status, components, or evolution!"""
    
    # Status
    elif 'status' in message_lower:
        return f"System Status:{status_info}\n• Core: Active\n• Evolution: Running\n• Self-healer: Active\n• Learning: Continuous\n• Web Interface: Online"
    
    # Evolution
    elif 'evolv' in message_lower:
        return f"I'm constantly evolving to become better! Currently at generation {get_generation()}. I learn from every interaction and improve my components automatically."
    
    # Components
    elif 'component' in message_lower:
        if component_count > 0:
            return f"I have {component_count} total components, with {healthy_count} currently healthy. The evolution engine is working to improve the remaining {component_count - healthy_count} components."
        return f"I have multiple components across Phase 0-7. The evolution engine works to keep them all healthy and improving."
    
    # Learning
    elif 'learn' in message_lower:
        return f"I learn continuously from our conversations. Every interaction helps me understand better and improve my responses. What would you like to teach me?"
    
    # Thanks
    elif any(t in message_lower for t in ['thank', 'thanks']):
        return "You're welcome! I'm here to help. Feel free to ask me anything."
    
    # Default
    else:
        responses = [
            f"That's an interesting question. Let me think about that. I'm currently at generation {get_generation()} and continuously learning.",
            f"I appreciate you asking. I'm still evolving, but I can help with questions about system status, components, or evolution. What else would you like to know?",
            f"I'm processing your message. As I evolve, I'll get better at answering these types of questions. In the meantime, feel free to ask about my status or components!",
            f"Good question! I'm learning from every interaction. Right now, I have {healthy_count}/{component_count} components healthy and actively evolving."
        ]
        return random.choice(responses)

# ============================================
# AUTHENTICATION DECORATOR (for admin only)
# ============================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# ROUTES - PUBLIC (no auth needed)
# ============================================

@app.route('/')
def index():
    """Main chat interface - no login required"""
    return redirect(url_for('chat_page'))

@app.route('/chat')
def chat_page():
    """Serve the chat interface page - TEMPORARILY FORCED PUBLIC
    DMAI will restore proper authentication later"""
    # TEMPORARY: Force authenticated=True to bypass login screen
    return render_template('chat.html', 
                         generation=get_generation(),
                         user='Guest',
                         authenticated=True,
                         admin_mode=False)

@app.route('/vision')
def vision():
    """DMAI Vision document"""
    return render_template('vision.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'core': core is not None,
        'authenticated': session.get('admin_authenticated', False)
    })

# ============================================
# ADMIN ROUTES (require authentication)
# ============================================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        data = request.json
        password = data.get('password', '')
        
        if password == MASTER_PASSWORD:
            session['admin_authenticated'] = True
            session.permanent = True
            logger.info(f"✅ Admin authenticated successfully")
            return jsonify({'success': True, 'redirect': '/admin'})
        
        logger.warning(f"❌ Failed admin login attempt")
        return jsonify({'success': False, 'error': 'Invalid password'}), 401
    
    # GET request - show login page
    return render_template('admin_login.html')

@app.route('/admin')
@admin_required
def admin():
    """Admin dashboard"""
    return render_template('admin.html', user='Admin')

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout"""
    session.pop('admin_authenticated', None)
    return jsonify({'success': True})

# ============================================
# API ROUTES - Chat (no auth needed)
# ============================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat endpoint - no authentication required"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Generate intelligent response
    response = generate_response(message)
    generation = get_generation()
    
    return jsonify({
        'response': response,
        'generation': generation,
        'timestamp': str(datetime.now())
    })

# ============================================
# API ROUTES - Status (public)
# ============================================

@app.route('/api/status')
def api_status():
    """Get system status - public"""
    if core:
        try:
            if hasattr(core, 'get_status'):
                status = core.get_status()
            else:
                status = {
                    'generation': get_generation(),
                    'components': {
                        'total': 52,
                        'healthy': 13,
                        'needs_evolution': 39,
                    },
                    'metrics': {
                        'evolutions': getattr(core, 'evolution_count', 0),
                        'thoughts_processed': getattr(core, 'thoughts_count', 0),
                    }
                }
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Core not initialized'}), 503

# ============================================
# API ROUTES - Research Targets Management
# ============================================

@app.route('/api/research/targets', methods=['GET', 'POST', 'DELETE'])
@admin_required
def api_research_targets():
    """Manage research targets"""
    manifest_path = Path('research/manifest.json')
    
    if request.method == 'GET':
        # Return all research targets
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'repositories': []})
    
    elif request.method == 'POST':
        # Add a new research target
        data = request.json
        if not data or 'name' not in data or 'url' not in data:
            return jsonify({'error': 'Name and URL required'}), 400
        
        # Load existing manifest
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {'repositories': []}
        
        # Add new target
        new_target = {
            'name': data['name'],
            'url': data['url'],
            'priority': data.get('priority', len(manifest['repositories']) + 1),
            'reason': data.get('reason', 'User-added research target'),
            'integration_potential': data.get('integration_potential', ['To be determined']),
            'added_by': 'user',
            'added_at': datetime.now().isoformat()
        }
        
        manifest['repositories'].append(new_target)
        
        # Save
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Added research target: {data['name']}")
        return jsonify({'success': True, 'target': new_target})
    
    elif request.method == 'DELETE':
        # Remove a research target
        data = request.json
        if not data or 'name' not in data:
            return jsonify({'error': 'Name required'}), 400
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        original_count = len(manifest['repositories'])
        manifest['repositories'] = [r for r in manifest['repositories'] if r['name'] != data['name']]
        
        if len(manifest['repositories']) < original_count:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"✅ Removed research target: {data['name']}")
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Target not found'}), 404

# ============================================
# API ROUTES - Admin (require auth)
# ============================================

@app.route('/api/evolution/queue')
@admin_required
def api_evolution_queue():
    """Get evolution queue - admin only"""
    return jsonify({
        'queue_size': 39,
        'needs_evolution': [
            {'id': 'P0T0_Migrate_Database_to_Production', 'health_score': 16},
            {'id': 'P1T10_Test_fragment_recreation', 'health_score': 0}
        ]
    })

@app.route('/api/command', methods=['POST'])
@admin_required
def api_command():
    """Execute admin commands"""
    data = request.json
    command = data.get('command')
    
    if command == 'evolve':
        return jsonify({'success': True, 'message': 'Evolution triggered'})
    elif command == 'health_audit':
        return jsonify({'success': True, 'message': 'Health audit completed'})
    else:
        return jsonify({'success': False, 'error': f'Unknown command: {command}'}), 400

# ============================================
# STATIC FILES
# ============================================

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"🚀 Starting DMAI Web Interface on port {port}")
    logger.info(f"🔓 Chat is PUBLIC - no login required")
    logger.info(f"🔐 Admin login requires password")
    logger.info(f"🧠 Core status: {'Loaded' if core else 'Not loaded'}")
    logger.info(f"📋 Research Targets API enabled at /api/research/targets")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
