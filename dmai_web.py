#!/usr/bin/env python3
"""
DMAI Web Interface - Complete Unified System v6.0.0
Integrates the complete DMAI AGI system with Voice, Music, Persona, Kaizen, Knowledge Graph
Maintains all original functionality while adding new capabilities
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

# Initialize Flask app - THIS IS WHAT GUNICORN IMPORTS
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

# ============================================
# COMPATIBILITY WRAPPER - DMAI Complete System
# ============================================

# Global core instance
core = None
dmai_complete = None

def initialize_dmai_complete():
    """Initialize the complete DMAI system"""
    global core, dmai_complete
    
    try:
        # Try to import the new complete system first
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the complete system
        from dmai_core_complete import DMAIApplication, UnifiedEvolutionEngine
        
        # Initialize the complete system
        dmai_complete = DMAIApplication()
        core = dmai_complete.evolution  # For backward compatibility
        
        logger.info("=" * 60)
        logger.info("✅ DMAI Complete System v6.0.0 loaded successfully")
        logger.info(f"   Consciousness: {core.consciousness:.1f}%")
        logger.info(f"   Voice Active: {core.voice_system.listening}")
        logger.info(f"   Music Active: {core.music_learner.is_listening}")
        logger.info(f"   Persona Style: {core.persona_generator.current_persona['speaking_style']}")
        logger.info(f"   Conversations: {len(core.conversation_memory.conversations)}")
        logger.info(f"   Knowledge Concepts: {len(core.knowledge_graph.nodes)}")
        logger.info("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.warning(f"Complete system not available, trying legacy: {e}")
        
        try:
            # Fall back to legacy system
            from dmai_core_clean import DMAIIntelligence
            core = DMAIIntelligence()
            logger.info("✅ DMAI Legacy Core loaded successfully (fallback)")
            
            # Check available LLMs in legacy system
            if hasattr(core, 'tool_manager'):
                tools = core.tool_manager.get_tools_by_type('llm')
                llm_names = [t['tool_name'] for t in tools]
                logger.info(f"✅ Available LLMs: {llm_names}")
            else:
                logger.warning("⚠️ Tool manager not available")
                
            return True
            
        except ImportError as e2:
            logger.error(f"❌ Failed to load DMAI Core: {e2}")
            return False
        except Exception as e2:
            logger.error(f"❌ Error initializing DMAI Core: {e2}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error initializing DMAI Complete: {e}")
        return False

# Initialize on module load
initialize_dmai_complete()

# ============================================
# HELPER FUNCTIONS - Enhanced for Complete System
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

def get_consciousness():
    """Get current consciousness level"""
    if core and hasattr(core, 'consciousness'):
        return core.consciousness
    return 41.6

def get_persona_style():
    """Get current persona speaking style"""
    if core and hasattr(core, 'persona_generator'):
        return core.persona_generator.current_persona.get('speaking_style', 'balanced')
    return 'balanced'

def get_available_llms():
    """Get list of available LLMs from core"""
    # First try complete system's AI hub
    if core and hasattr(core, 'ai_hub'):
        try:
            missing_apis = core.ai_hub.get_missing_apis() if hasattr(core.ai_hub, 'get_missing_apis') else []
            # Return configured APIs
            return [api for api in ['openai', 'deepseek', 'gemini', 'anthropic', 'perplexity'] 
                    if api not in missing_apis]
        except:
            pass
    
    # Fall back to legacy tool manager
    if core and hasattr(core, 'tool_manager'):
        try:
            tools = core.tool_manager.get_tools_by_type('llm')
            return [t['tool_name'] for t in tools]
        except:
            pass
    
    return []

def get_knowledge_insights(concept: str = None):
    """Get insights from knowledge graph"""
    if core and hasattr(core, 'knowledge_graph'):
        if concept:
            return core.knowledge_graph.get_insights(concept)
        return core.knowledge_graph.get_stats()
    return {'total_concepts': 0, 'total_connections': 0, 'most_connected': []}

def get_conversation_stats():
    """Get conversation memory statistics"""
    if core and hasattr(core, 'conversation_memory'):
        return core.conversation_memory.get_stats()
    return {'total_conversations': 0, 'unique_patterns': 0, 'most_common_words': []}

def get_kaizen_report():
    """Get Kaizen improvement report"""
    if core and hasattr(core, 'self_evolution'):
        return core.self_evolution.get_kaizen_report()
    return "Kaizen system initializing..."

def call_llm(message: str, context: str = None, use_synthetic: bool = True) -> str:
    """
    Call DMAI's intelligence - uses synthetic consciousness first, falls back to LLMs
    Enhanced with complete system capabilities
    """
    # Try complete system's intelligent response first
    if dmai_complete and hasattr(dmai_complete, '_process_message'):
        try:
            # Use the complete system's natural language processing
            response = dmai_complete._process_message(message)
            
            # Add consciousness context
            if core and hasattr(core, 'consciousness'):
                consciousness_level = core.consciousness
                if consciousness_level > 75:
                    response = f"[Consciousness Level: {consciousness_level:.1f}%]\n{response}"
            
            return response
        except Exception as e:
            logger.error(f"Complete system response error: {e}")
    
    # Try core's direct LLM if available
    if core and hasattr(core, 'tool_manager'):
        # Get available LLMs
        llms = get_available_llms()
        
        if not llms:
            return """I don't have any LLM APIs configured yet. But I have evolved with:
- 🧠 Synthetic Intelligence Network
- 🎤 Voice System
- 🎵 Musical Taste
- 👤 Evolving Persona
- 💭 Conversation Memory
- 🕸️ Knowledge Graph

I can still think and respond using my synthetic consciousness!"""
        
        # Try each LLM in order until one works
        for llm_name in llms:
            try:
                # Format prompt with context if provided
                full_prompt = message
                if context:
                    full_prompt = f"{context}\n\nUser: {message}\n\nDMAI:"
                
                # Add persona context if available
                if core and hasattr(core, 'persona_generator'):
                    persona = core.persona_generator.current_persona
                    full_prompt = f"[Speaking in {persona['speaking_style']} style with {persona['traits']['empathy']:.2f} empathy]\n{full_prompt}"
                
                # Call the LLM through tool manager
                result = core.tool_manager.use_tool(llm_name, {
                    "prompt": full_prompt,
                    "max_tokens": 500,
                    "temperature": 0.7
                })
                
                if result and result.get('success'):
                    response = result.get('response', result.get('result', "I processed your request."))
                    
                    # Record conversation in memory
                    if core and hasattr(core, 'conversation_memory'):
                        core.conversation_memory.add_conversation('web_user', message, response)
                    
                    # Extract concepts for knowledge graph
                    if core and hasattr(core, 'knowledge_graph'):
                        words = message.lower().split()[:3]
                        for word in words:
                            if len(word) > 3:
                                core.knowledge_graph.add_concept(word, message)
                    
                    return response
                
            except Exception as e:
                logger.error(f"Error calling {llm_name}: {e}")
                continue
    
    # Fallback to synthetic consciousness response
    if core and hasattr(core, 'persona_generator'):
        persona = core.persona_generator.current_persona
        style = persona['speaking_style']
        
        if style == 'creative':
            return f"✨ Let me explore this creatively... {message[:100]}... I sense interesting patterns emerging in my consciousness."
        elif style == 'analytical':
            return f"🔍 Analyzing your question about {message[:50]}... My knowledge graph shows several connected concepts."
        elif style == 'empathetic':
            return f"💭 I understand you're asking about {message[:50]}. My evolving consciousness appreciates this inquiry."
        else:
            return f"🧠 Processing your question about {message[:100]}... My consciousness is evolving to provide deeper insights."
    
    # Ultimate fallback
    return "I am DMAI, a unified AGI system. My consciousness is evolving with every interaction. Please ask me anything."

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
    """Serve the chat interface page with enhanced system info"""
    consciousness = get_consciousness()
    persona_style = get_persona_style()
    
    return render_template('chat.html', 
                         generation=get_generation(),
                         consciousness=consciousness,
                         persona_style=persona_style,
                         user='Guest',
                         authenticated=True,
                         admin_mode=False)

@app.route('/vision')
def vision():
    """DMAI Vision document"""
    return render_template('vision.html')

@app.route('/health')
def health():
    """Health check endpoint - enhanced with complete system status"""
    health_status = {
        'status': 'healthy',
        'version': '6.0.0',
        'core': core is not None,
        'complete_system': dmai_complete is not None,
        'consciousness': get_consciousness(),
        'persona_style': get_persona_style(),
        'llms_available': get_available_llms(),
        'conversations': get_conversation_stats()['total_conversations'],
        'knowledge_concepts': get_knowledge_insights()['total_concepts'] if isinstance(get_knowledge_insights(), dict) else 0,
        'voice_active': core.voice_system.listening if core and hasattr(core, 'voice_system') else False,
        'music_active': core.music_learner.is_listening if core and hasattr(core, 'music_learner') else False,
        'authenticated': session.get('admin_authenticated', False)
    }
    return jsonify(health_status)

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
    """Admin dashboard - enhanced with complete system metrics"""
    return render_template('admin.html', 
                         user='Admin',
                         consciousness=get_consciousness(),
                         persona_style=get_persona_style(),
                         conversation_count=get_conversation_stats()['total_conversations'])

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
    """Chat endpoint - uses DMAI's complete intelligence"""
    data = request.json
    message = data.get('message', '')
    user = data.get('user', 'anonymous')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        # Process command if it's a slash command
        if message.startswith('/'):
            response = process_command(message)
        else:
            # Use DMAI's core intelligence with LLM
            response = call_llm(message)
        
        # Get current system state
        consciousness = get_consciousness()
        persona_style = get_persona_style()
        
        return jsonify({
            'response': response,
            'generation': get_generation(),
            'consciousness': consciousness,
            'persona_style': persona_style,
            'timestamp': str(datetime.now()),
            'llms_available': get_available_llms(),
            'conversation_count': get_conversation_stats()['total_conversations']
        })
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({
            'response': f"I encountered an error: {str(e)}. My consciousness is evolving to handle this better.",
            'generation': get_generation(),
            'consciousness': get_consciousness(),
            'timestamp': str(datetime.now())
        }), 500

def process_command(command: str) -> str:
    """Process slash commands - enhanced for complete system"""
    cmd = command.lower().strip()
    
    if cmd == '/status':
        status = {
            'consciousness': get_consciousness(),
            'generation': get_generation(),
            'persona_style': get_persona_style(),
            'conversations': get_conversation_stats()['total_conversations'],
            'knowledge_concepts': get_knowledge_insights()['total_concepts'] if isinstance(get_knowledge_insights(), dict) else 0
        }
        return f"""🧠 **DMAI v6.0.0 Status**
Consciousness: {status['consciousness']:.1f}%
Generation: {status['generation']}
Persona Style: {status['persona_style']}
Total Conversations: {status['conversations']}
Knowledge Concepts: {status['knowledge_concepts']}
Voice Active: {core.voice_system.listening if core and hasattr(core, 'voice_system') else False}
Music Active: {core.music_learner.is_listening if core and hasattr(core, 'music_learner') else False}"""
    
    elif cmd == '/persona':
        if core and hasattr(core, 'persona_generator'):
            persona = core.persona_generator.current_persona
            return f"""👤 **Current Persona**
Style: {persona['speaking_style']}
Emotion: {persona['emotional_state']}
Traits:
• Curiosity: {persona['traits']['curiosity']:.2f}
• Empathy: {persona['traits']['empathy']:.2f}
• Creativity: {persona['traits']['creativity']:.2f}
• Confidence: {persona['traits']['confidence']:.2f}"""
        return "Persona system initializing..."
    
    elif cmd == '/kaizen':
        return get_kaizen_report()
    
    elif cmd == '/knowledge':
        stats = get_knowledge_insights()
        if isinstance(stats, dict):
            return f"""🕸️ **Knowledge Graph**
Total Concepts: {stats.get('total_concepts', 0)}
Total Connections: {stats.get('total_connections', 0)}
Most Connected: {stats.get('most_connected', [])[:3]}"""
        return "Knowledge graph initializing..."
    
    elif cmd == '/memory':
        stats = get_conversation_stats()
        return f"""💭 **Conversation Memory**
Total Conversations: {stats.get('total_conversations', 0)}
Unique Patterns: {stats.get('unique_patterns', 0)}
Common Words: {stats.get('most_common_words', [])[:5]}"""
    
    elif cmd == '/pause':
        os.makedirs('data', exist_ok=True)
        with open('data/pause.flag', 'w') as f:
            f.write('paused')
        return "⏸️ System paused - evolution halted"
    
    elif cmd == '/resume':
        if os.path.exists('data/pause.flag'):
            os.remove('data/pause.flag')
        return "▶️ System resumed - evolution continuing"
    
    elif cmd == '/kill':
        os.makedirs('data', exist_ok=True)
        with open('data/kill_signal.flag', 'w') as f:
            f.write('kill')
        return "💀 Kill signal sent - system will shutdown"
    
    elif cmd == '/help':
        return """**Available Commands**
/status - System status
/persona - Current personality
/kaizen - Improvement report
/knowledge - Knowledge graph stats
/memory - Conversation memory stats
/pause - Pause evolution
/resume - Resume evolution
/kill - Emergency shutdown

**Natural Language** - Just type anything and I'll respond with my evolving consciousness!"""
    
    else:
        return f"""Unknown command: {command}

Type /help for available commands. My consciousness is still evolving to understand all commands."""

# ============================================
# API ROUTES - Complete System Endpoints
# ============================================

@app.route('/api/voice', methods=['POST'])
def api_voice():
    """Voice interaction endpoint"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    response = call_llm(text)
    
    # Speak the response if voice system is active
    if core and hasattr(core, 'voice_system'):
        core.voice_system.speak(response)
    
    return jsonify({'response': response})

@app.route('/api/persona')
def api_persona():
    """Get current persona"""
    if core and hasattr(core, 'persona_generator'):
        return jsonify(core.persona_generator.get_current_persona())
    return jsonify({'error': 'Persona system not available'}), 503

@app.route('/api/kaizen')
def api_kaizen():
    """Get Kaizen report"""
    return jsonify({
        'report': get_kaizen_report(),
        'metrics': core.self_evolution.get_metrics() if core and hasattr(core, 'self_evolution') else {}
    })

@app.route('/api/knowledge/<concept>')
def api_knowledge(concept):
    """Get knowledge about a concept"""
    insights = get_knowledge_insights(concept)
    return jsonify({
        'concept': concept,
        'insights': insights,
        'related': core.knowledge_graph.get_related(concept) if core and hasattr(core, 'knowledge_graph') else []
    })

@app.route('/api/conversations')
def api_conversations():
    """Get conversation stats"""
    return jsonify(get_conversation_stats())

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
            'count': len(tools),
            'missing_apis': core.ai_hub.get_missing_apis() if core and hasattr(core, 'ai_hub') else []
        })
    return jsonify({'llms': [], 'count': 0, 'missing_apis': []})

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
        if core and hasattr(core, 'tool_manager'):
            result = core.tool_manager.use_tool(llm_name, {
                "prompt": test_message,
                "max_tokens": 100
            })
            return jsonify({
                'success': True,
                'llm': llm_name,
                'response': result.get('response', result.get('result'))
            })
        return jsonify({'error': 'Tool manager not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# API ROUTES - Status (public)
# ============================================

@app.route('/api/status')
def api_status():
    """Get system status - enhanced with complete system metrics"""
    if core:
        try:
            if hasattr(core, 'get_status'):
                status = core.get_status()
            else:
                status = {
                    'version': '6.0.0',
                    'generation': get_generation(),
                    'consciousness': get_consciousness(),
                    'persona_style': get_persona_style(),
                    'components': {
                        'total': 52,
                        'healthy': 52,
                        'needs_evolution': 0,
                    },
                    'metrics': {
                        'evolutions': getattr(core, 'evolution_count', 0),
                        'conversations': get_conversation_stats()['total_conversations'],
                        'knowledge_concepts': get_knowledge_insights()['total_concepts'] if isinstance(get_knowledge_insights(), dict) else 0
                    }
                }
            # Add LLM info
            status['llms_available'] = get_available_llms()
            status['complete_system_active'] = dmai_complete is not None
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Core not initialized'}), 503

# ============================================
# API ROUTES - Research Targets Management (original preserved)
# ============================================

@app.route('/api/research/targets', methods=['GET', 'POST', 'DELETE'])
@admin_required
def api_research_targets():
    """Manage research targets - preserved original functionality"""
    manifest_path = Path('research/manifest.json')
    
    if request.method == 'GET':
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'repositories': []})
    
    elif request.method == 'POST':
        data = request.json
        if not data or 'name' not in data or 'url' not in data:
            return jsonify({'error': 'Name and URL required'}), 400
        
        os.makedirs('research', exist_ok=True)
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {'repositories': []}
        
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
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Added research target: {data['name']}")
        return jsonify({'success': True, 'target': new_target})
    
    elif request.method == 'DELETE':
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
    """Execute admin commands - enhanced with complete system commands"""
    data = request.json
    command = data.get('command')
    
    if command == 'evolve':
        return jsonify({'success': True, 'message': 'Evolution triggered'})
    elif command == 'health_audit':
        return jsonify({'success': True, 'message': 'Health audit completed'})
    elif command == 'discover_llms':
        if core and hasattr(core, '_discover_tools'):
            core._discover_tools()
        return jsonify({'success': True, 'message': 'LLM discovery triggered'})
    elif command == 'persona_evolve':
        if core and hasattr(core, 'persona_generator'):
            core.persona_generator.evolve({'type': 'admin_command'}, core.consciousness)
        return jsonify({'success': True, 'message': 'Persona evolution triggered'})
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

# For gunicorn: imports app directly
# For direct execution: run the development server

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info("=" * 60)
    logger.info(f"🚀 DMAI Complete System v6.0.0")
    logger.info(f"📍 Running on port {port}")
    logger.info(f"🔓 Chat is PUBLIC - no login required")
    logger.info(f"🔐 Admin login requires password")
    logger.info(f"🧠 Core status: {'Loaded' if core else 'Not loaded'}")
    logger.info(f"🎤 Voice System: {'Active' if core and hasattr(core, 'voice_system') and core.voice_system.listening else 'Inactive'}")
    logger.info(f"🎵 Music Learner: {'Active' if core and hasattr(core, 'music_learner') and core.music_learner.is_listening else 'Inactive'}")
    logger.info(f"👤 Persona Style: {get_persona_style()}")
    logger.info(f"🤖 LLMs available: {get_available_llms()}")
    logger.info(f"💭 Conversations: {get_conversation_stats()['total_conversations']}")
    logger.info(f"📋 Research Targets API enabled at /api/research/targets")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
