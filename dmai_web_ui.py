#!/usr/bin/env python3
"""DMAI Web Control UI - Access via browser"""
from flask import Flask, render_template_string, request, jsonify
import sys
import os
import json
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.safety_switch import safety
from voice.auth.voice_auth import VoiceAuth
from language_learning.processor.language_learner import LanguageLearner

app = Flask(__name__)

# Global state
safety_switch = safety
voice_auth = VoiceAuth()
learner = LanguageLearner()
master_id = "david"
backup_codes = ["DMAI-OVERRIDE-2026", "MASTER-KEY-789"]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧬 DMAI Master Control</title>
    <style>
        body { background: #1a1a2f; color: #e0e0e0; font-family: Arial; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; padding: 20px; }
        .header h1 { color: #6e8efb; font-size: 2em; }
        .status { background: #16213e; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .authenticated { color: #00ff00; font-weight: bold; }
        .not-auth { color: #ff4444; font-weight: bold; }
        .auth-section { background: #0a0a1a; padding: 20px; border-radius: 5px; margin: 10px 0; }
        .command-section { background: #16213e; padding: 20px; border-radius: 5px; margin: 10px 0; }
        button { 
            background: #6e8efb; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            margin: 5px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 1.1em;
        }
        button:hover { background: #5a7dda; }
        button.danger { background: #ff4444; }
        button.danger:hover { background: #cc0000; }
        button.warning { background: #ffaa00; color: black; }
        input, select { 
            width: 100%; 
            padding: 10px; 
            margin: 5px 0; 
            background: #1a1a2f; 
            border: 1px solid #3a3a4f; 
            color: white; 
            border-radius: 3px;
        }
        .output { 
            background: #0a0a1a; 
            padding: 15px; 
            border-radius: 5px; 
            height: 200px; 
            overflow-y: scroll;
            font-family: monospace;
            margin: 10px 0;
        }
        .info-bar {
            display: flex;
            justify-content: space-between;
            background: #16213e;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .info-item {
            text-align: center;
        }
        .info-label {
            font-size: 0.8em;
            color: #888;
        }
        .info-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #a777e3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 DMAI MASTER CONTROL</h1>
        </div>
        
        <div class="status" id="authStatus">
            <span id="authText">⚠️ NOT AUTHENTICATED</span>
        </div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="info-label">Vocabulary</div>
                <div class="info-value" id="vocabCount">{{ vocab }}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Generation</div>
                <div class="info-value" id="generation">{{ generation }}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Paused</div>
                <div class="info-value" id="paused">{{ paused }}</div>
            </div>
        </div>
        
        <div class="auth-section">
            <h2>🔐 Authentication</h2>
            <select id="authMethod">
                <option value="backup">Backup Code</option>
                <option value="password">Master Password</option>
            </select>
            <input type="password" id="authCode" placeholder="Enter code/password">
            <button onclick="authenticate()">Authenticate</button>
            <div id="authMessage" style="margin-top: 10px;"></div>
        </div>
        
        <div class="command-section">
            <h2>⚡ Emergency Controls</h2>
            <div style="text-align: center;">
                <button class="warning" onclick="sendCommand('pause')">⏸️ Pause</button>
                <button onclick="sendCommand('resume')">▶️ Resume</button>
                <button class="danger" onclick="sendCommand('kill')">🛑 Kill</button>
            </div>
        </div>
        
        <div class="command-section">
            <h2>📝 Command Input</h2>
            <input type="text" id="commandInput" placeholder="Enter command..." disabled>
            <button onclick="sendCommand()" id="sendButton" disabled>Send Command</button>
        </div>
        
        <div class="output" id="output">
            [System Ready]
        </div>
    </div>
    
    <script>
        let authenticated = false;
        
        function log(message) {
            const output = document.getElementById('output');
            const timestamp = new Date().toLocaleTimeString();
            output.innerHTML += `\\n[${timestamp}] ${message}`;
            output.scrollTop = output.scrollHeight;
        }
        
        async function authenticate() {
            const method = document.getElementById('authMethod').value;
            const code = document.getElementById('authCode').value;
            
            const response = await fetch('/authenticate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({method: method, code: code})
            });
            
            const data = await response.json();
            if (data.success) {
                authenticated = true;
                document.getElementById('authStatus').innerHTML = '<span class="authenticated">✅ AUTHENTICATED</span>';
                document.getElementById('commandInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                log('✅ Authentication successful');
                updateStatus();
            } else {
                log('❌ Authentication failed');
            }
        }
        
        async function sendCommand(cmd = null) {
            if (!authenticated) {
                log('❌ Please authenticate first');
                return;
            }
            
            const command = cmd || document.getElementById('commandInput').value;
            if (!command) return;
            
            document.getElementById('commandInput').value = '';
            log(`>>> ${command}`);
            
            const response = await fetch('/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            });
            
            const data = await response.json();
            log(`🤖 DMAI: ${data.response}`);
            
            if (command === 'pause' || command === 'resume' || command === 'kill') {
                updateStatus();
            }
        }
        
        async function updateStatus() {
            const response = await fetch('/status');
            const data = await response.json();
            document.getElementById('vocabCount').innerText = data.vocabulary;
            document.getElementById('generation').innerText = data.generation;
            document.getElementById('paused').innerText = data.paused ? 'YES' : 'NO';
        }
        
        // Update status every 30 seconds
        setInterval(updateStatus, 30000);
        updateStatus();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    vocab = len(learner.vocabulary)
    try:
        with open('ai_core/evolution/current_generation.txt', 'r') as f:
            gen = f.read().strip()
    except:
        gen = '?'
    return render_template_string(HTML_TEMPLATE, vocab=vocab, generation=gen, paused=safety_switch.check_paused())

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    method = data.get('method')
    code = data.get('code')
    
    if method == 'backup':
        success = code in backup_codes
    elif method == 'password':
        success = code == "Master2026!"
    else:
        success = False
    
    return jsonify({'success': success})

@app.route('/command', methods=['POST'])
def command():
    data = request.json
    cmd = data.get('command', '').lower().strip()
    
    if cmd == 'pause':
        success, msg = safety_switch.pause(master_id)
        return jsonify({'response': msg})
    elif cmd == 'resume':
        success, msg = safety_switch.resume(master_id)
        return jsonify({'response': msg})
    elif cmd == 'kill':
        success, msg = safety_switch.kill(master_id, "DMAI terminate authority override")
        return jsonify({'response': msg})
    elif cmd == 'status':
        return jsonify({'response': f"Paused: {safety_switch.check_paused()}, Vocab: {len(learner.vocabulary)}"})
    else:
        return jsonify({'response': f"Command received: {cmd}"})

@app.route('/status')
def status():
    return jsonify({
        'vocabulary': len(learner.vocabulary),
        'generation': open('ai_core/evolution/current_generation.txt').read().strip() if os.path.exists('ai_core/evolution/current_generation.txt') else '?',
        'paused': safety_switch.check_paused()
    })

if __name__ == '__main__':
    print("\n🚀 DMAI Web Control UI")
    print("="*50)
    print("Open your browser to: http://localhost:5000")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False)
