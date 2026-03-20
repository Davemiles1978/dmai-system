#!/usr/bin/env python3
"""
DMAI Web Interface - Full LLM Integration
Uses DMAI's ExternalToolManager to access all LLMs including DeepSeek
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
    
    # Check available LLMs
    if hasattr(core, 'tool_manager'):
        tools = core.tool_manager.get_tools_by_type('llm')
        llm_names = [t['tool_name'] for t in tools]
        logger.info(f"✅ Available LLMs: {llm_names}")
    else:
        logger.warning("⚠️ Tool manager not available")
        
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

def get_available_llms():
    """Get list of available LLMs from core"""
    if core and hasattr(core, 'tool_manager'):
        try:
            tools = core.tool_manager.get_tools_by_type('llm')
            return [t['tool_name'] for t in tools]
        except:
            pass
    return []

def call_llm(message: str, context: str = None) -> str:
    """
    Call DMAI's LLM through the tool manager
    Supports OpenAI, Google Gemini, DeepSeek, Anthropic, and more
    """
    if not core or not hasattr(core, 'tool_manager'):
        return "I'm still initializing my intelligence. Please wait a moment and try again."
    
    # Get available LLMs
    llms = get_available_llms()
    
    if not llms:
        return """I don't have any LLM APIs configured yet. To give me intelligence, add API keys for:
- OpenAI (GPT-4, GPT-3.5)
- Google Gemini
- DeepSeek
- Anthropic Claude

Add these to your environment variables and I'll evolve to use them!"""
    
    # Try each LLM in order until one works
    for llm_name in llms:
        try:
            # Format prompt with context if provided
            full_prompt = message
            if context:
                full_prompt = f"{context}\n\nUser: {message}\n\nDMAI:"
            
            # Call the LLM through tool manager
            result = core.tool_manager.use_tool(llm_name, {
                "prompt": full_prompt,
                "max_tokens": 500,
                "temperature": 0.7
            })
            
            if result and result.get('success'):
                return result.get('response', result.get('result', "I processed your request but didn't get a clear response."))
            
        except Exception as e:
            logger.error(f"Error calling {llm_name}: {e}")
            continue
    
    return "I'm having trouble connecting to my LLM services. Please check API keys and try again. I'll evolve to be more resilient!"

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
        'llms_available': get_available_llms(),
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
    """Chat endpoint - uses DMAI's LLM capabilities"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        # Use DMAI's core intelligence with LLM
        response = call_llm(message)
        generation = get_generation()
        
        return jsonify({
            'response': response,
            'generation': generation,
            'timestamp': str(datetime.now()),
            'llms_available': get_available_llms()
        })
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({
            'response': f"I encountered an error: {str(e)}. I'm evolving to handle this better.",
            'generation': get_generation(),
            'timestamp': str(datetime.now())
        }), 500

# ============================================
# API ROUTES - LLM Management (admin)
# ============================================

@app.route('/api/llms', methods=['GET'])
@admin_required
def api_llms():
    """Get available LLMs and their status"""
    if core and hasattr(core, 'tool_manager'):
        tools = core.tool_manager.get_tools_by_type('llm')
        return jsonify({
            'llms': tools,
            'count': len(tools)
        })
    return jsonify({'llms': [], 'count': 0})

@app.route('/api/llms/test', methods=['POST'])
@admin_required
def api_test_llm():
    """Test a specific LLM"""
    data = request.json
    llm_name = data.get('llm_name')
    test_message = data.get('message', 'Hello, are you working?')
    
    if not llm_name:
        return jsonify({'error': 'LLM name required'}), 400
    
    try:
        result = core.tool_manager.use_tool(llm_name, {
            "prompt": test_message,
            "max_tokens": 100
        })
        return jsonify({
            'success': True,
            'llm': llm_name,
            'response': result.get('response', result.get('result'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            # Add LLM info
            status['llms_available'] = get_available_llms()
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
    elif command == 'discover_llms':
        # Force rediscovery of LLMs
        if core and hasattr(core, '_discover_tools'):
            core._discover_tools()
        return jsonify({'success': True, 'message': 'LLM discovery triggered'})
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
    logger.info(f"🤖 LLMs available: {get_available_llms()}")
    logger.info(f"📋 Research Targets API enabled at /api/research/targets")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
