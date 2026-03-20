#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""DMAI Cloud Web Control UI - Accessible from anywhere"""
from flask import Flask, render_template_string, request, jsonify
import os
import json
import requests
from datetime import datetime

app = Flask(__name__)

# Configuration - Update these with your actual service URLs
DMAI_VOICE_URL = "https://dmai-final.onrender.com"  # Your voice service
DMAI_EVOLUTION_URL = "https://dmai-cloud-evolution.onrender.com"  # Evolution service

# Backup codes (same as local)
BACKUP_CODES = ["DMAI-OVERRIDE-2026", "MASTER-KEY-789"]
MASTER_PASSWORD = "Master2026!"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧬 DMAI Cloud Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
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
            width: 100%;
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
            flex: 1;
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
        .device-status {
            background: #16213e;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .online { color: #00ff00; }
        .offline { color: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 DMAI CLOUD CONTROL</h1>
            <p>Accessible from anywhere • 24/7/365</p>
        </div>
        
        <div class="status" id="authStatus">
            <span id="authText">⚠️ NOT AUTHENTICATED</span>
        </div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="info-label">Generation</div>
                <div class="info-value" id="generation">?</div>
            </div>
            <div class="info-item">
                <div class="info-label">Voice Status</div>
                <div class="info-value" id="voiceStatus">⏳</div>
            </div>
            <div class="info-item">
                <div class="info-label">Evolution</div>
                <div class="info-value" id="evolutionStatus">⏳</div>
            </div>
        </div>
        
        <div class="device-status" id="deviceStatus">
            Checking devices...
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
            <button class="warning" onclick="sendCommand('pause')">⏸️ Pause DMAI</button>
            <button onclick="sendCommand('resume')">▶️ Resume DMAI</button>
            <button class="danger" onclick="sendCommand('kill')">🛑 Emergency Kill</button>
        </div>
        
        <div class="command-section">
            <h2>📝 Command Input</h2>
            <input type="text" id="commandInput" placeholder="Enter command..." disabled>
            <button onclick="sendCommand()" id="sendButton" disabled>Send Command</button>
        </div>
        
        <div class="output" id="output">
            [System Ready - Waiting for authentication]
        </div>
    </div>
    
    <script>
        let authenticated = false;
        let authToken = null;
        
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
                authToken = data.token;
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
                headers: {
                    'Content-Type': 'application/json',
                    'X-Auth-Token': authToken
                },
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
            
            document.getElementById('generation').innerText = data.generation || '?';
            
            // Update device status
            let deviceHtml = '📱 Connected Devices:<br>';
            for (let device in data.devices) {
                deviceHtml += `• ${device}: <span class="${data.devices[device] ? 'online' : 'offline'}">${data.devices[device] ? '✅ Online' : '❌ Offline'}</span><br>`;
            }
            document.getElementById('deviceStatus').innerHTML = deviceHtml;
            
            // Update individual service status
            document.getElementById('voiceStatus').innerHTML = data.voice_online ? '✅' : '❌';
            document.getElementById('evolutionStatus').innerHTML = data.evolution_online ? '✅' : '❌';
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
    return render_template_string(HTML_TEMPLATE)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    data = request.json
    method = data.get('method')
    code = data.get('code')
    
    if method == 'backup':
        success = code in BACKUP_CODES
    elif method == 'password':
        success = code == MASTER_PASSWORD
    else:
        success = False
    
    if success:
        # Generate simple token (in production, use proper JWT)
        token = os.urandom(16).hex()
        return jsonify({'success': True, 'token': token})
    return jsonify({'success': False})

@app.route('/command', methods=['POST'])
def command():
    token = request.headers.get('X-Auth-Token')
    if not token:  # In production, validate token properly
        return jsonify({'response': 'Authentication required'}), 401
    
    data = request.json
    cmd = data.get('command', '').lower().strip()
    
    # Forward commands to the appropriate services
    try:
        if cmd == 'pause':
            # Call your voice service pause endpoint
            response = requests.post(f"{DMAI_VOICE_URL}/pause", timeout=10)
            return jsonify({'response': 'Pause command sent to DMAI'})
        elif cmd == 'resume':
            response = requests.post(f"{DMAI_VOICE_URL}/resume", timeout=10)
            return jsonify({'response': 'Resume command sent to DMAI'})
        elif cmd == 'kill':
            response = requests.post(f"{DMAI_VOICE_URL}/kill", timeout=10)
            return jsonify({'response': 'Kill command sent to DMAI'})
        else:
            return jsonify({'response': f'Command "{cmd}" forwarded to DMAI'})
    except Exception as e:
        return jsonify({'response': f'Error communicating with DMAI: {str(e)}'})

@app.route('/status')
def status():
    # Check status of all DMAI services
    devices = {
        'MacBook (Local)': False,
        'Voice Service': False,
        'Evolution Engine': False,
        'Cloud UI': True
    }
    
    # Check voice service
    try:
        r = requests.get(f"{DMAI_VOICE_URL}/health", timeout=5)
        devices['Voice Service'] = r.status_code == 200
    except:
        pass
    
    # Check evolution service
    try:
        r = requests.get(f"{DMAI_EVOLUTION_URL}/health", timeout=5)
        devices['Evolution Engine'] = r.status_code == 200
    except:
        pass
    
    # Try to get generation
    generation = '?'
    try:
        r = requests.get(f"{DMAI_EVOLUTION_URL}/status", timeout=5)
        if r.status_code == 200:
            generation = r.json().get('generation', '?')
    except:
        pass
    
    return jsonify({
        'generation': generation,
        'devices': devices,
        'voice_online': devices['Voice Service'],
        'evolution_online': devices['Evolution Engine']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 DMAI Cloud Control UI")
    print("="*50)
    print(f"Running on port {port}")
    print("Access from anywhere once deployed to Render")
    print("="*50)
    app.run(host='0.0.0.0', port=port, debug=False)
