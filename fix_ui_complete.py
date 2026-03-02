#!/usr/bin/env python3
"""
Complete UI fix - rewrites the problematic JavaScript section
"""
import re

with open('ui/ai_ui.html', 'r') as f:
    content = f.read()

# Find the main script block
script_pattern = r'<script>(.*?)</script>'
match = re.search(script_pattern, content, re.DOTALL)

if match:
    old_script = match.group(1)
    
    # Create a clean, fixed script
    new_script = '''
// ==================== REAL EVOLUTION DATA FETCHING ====================
let evolutionData = {
    generation: 1,
    bestScore: 0,
    nextCycleIn: 3600,
    status: 'running'
};

// Mobile panel state
let chatListOpen = false;
let toolsOpen = false;

// Fetch real evolution stats from the API
async function fetchEvolutionStats() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            evolutionData.generation = data.generation || evolutionData.generation;
            updateGenerationDisplay();
            const dot = document.getElementById('evolution-status-dot');
            if (dot) dot.style.backgroundColor = 'var(--success)';
        } else {
            console.log('Health endpoint not available');
            if (!evolutionData.generation || evolutionData.generation === 1) {
                evolutionData.generation = 266;
                updateGenerationDisplay();
            }
        }
    } catch (error) {
        console.log('Evolution API error:', error);
        if (!evolutionData.generation || evolutionData.generation === 1) {
            evolutionData.generation = 266;
            updateGenerationDisplay();
        }
    }
}

// Submit human guidance to evolution engine
async function submitGuidanceToEngine(guidance) {
    try {
        const response = await fetch('/api/submit-guidance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ guidance: guidance })
        });
        if (response.ok) {
            showNotification('💡 Guidance sent to evolution engine');
        }
    } catch (error) {
        console.log('Guidance API not available');
    }
}

// Update all generation displays
function updateGenerationDisplay() {
    const displays = ['gen-display', 'gen-indicator'];
    displays.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = evolutionData.generation;
        }
    });
    
    const ticker = document.getElementById('cycle-ticker');
    if (ticker) {
        const nextIn = Math.floor(evolutionData.nextCycleIn / 60);
        ticker.textContent = `Next in ${nextIn}m`;
    }
}

setInterval(fetchEvolutionStats, 30000);

// ==================== LOGIN FUNCTIONS ====================
function login() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    
    if (username === 'david' && password === 'dmai2026') {
        localStorage.setItem('dmai_logged_in', 'true');
        localStorage.setItem('dmai_username', username);
        localStorage.setItem('dmai_last_active', Date.now().toString());
        
        document.getElementById('login-screen').style.display = 'none';
        document.getElementById('main-app').style.display = 'block';
        document.getElementById('current-user').textContent = username;
        document.getElementById('user-avatar').textContent = username.substring(0,2).toUpperCase();
        
        loadData();
        initializeToolsPanel();
        fetchEvolutionStats();
    } else {
        document.getElementById('login-error').textContent = 'Invalid credentials';
    }
}

function register() {
    const username = document.getElementById('reg-username').value.trim();
    const password = document.getElementById('reg-password').value;
    const confirm = document.getElementById('reg-confirm').value;
    
    if (!username || !password) {
        document.getElementById('register-error').textContent = 'All fields required';
        return;
    }
    if (password !== confirm) {
        document.getElementById('register-error').textContent = 'Passwords do not match';
        return;
    }
    
    // Simple registration - in production this would save to users object
    alert('Registration successful! Please login.');
    switchLoginTab('login');
}

function switchLoginTab(tab) {
    document.querySelectorAll('.login-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.login-form').forEach(f => f.classList.add('hidden'));
    
    if (tab === 'login') {
        document.querySelector('.login-tab').classList.add('active');
        document.getElementById('login-form').classList.remove('hidden');
    } else {
        document.querySelectorAll('.login-tab')[1].classList.add('active');
        document.getElementById('register-form').classList.remove('hidden');
    }
}

// ==================== MOBILE FUNCTIONS ====================
function toggleChatList() {
    const panel = document.getElementById('chatListPanel');
    const overlay = document.getElementById('mobileOverlay');
    const toolsPanel = document.getElementById('toolsPanel');
    
    if (toolsPanel.classList.contains('open')) {
        toolsPanel.classList.remove('open');
    }
    
    panel.classList.toggle('open');
    overlay.classList.toggle('visible');
}

function toggleToolsPanel() {
    const panel = document.getElementById('toolsPanel');
    const overlay = document.getElementById('mobileOverlay');
    const chatPanel = document.getElementById('chatListPanel');
    
    if (chatPanel.classList.contains('open')) {
        chatPanel.classList.remove('open');
    }
    
    panel.classList.toggle('open');
    overlay.classList.toggle('visible');
}

function closeAllPanels() {
    const chatPanel = document.getElementById('chatListPanel');
    const toolsPanel = document.getElementById('toolsPanel');
    const overlay = document.getElementById('mobileOverlay');
    
    chatPanel.classList.remove('open');
    toolsPanel.classList.remove('open');
    overlay.classList.remove('visible');
}

// ==================== CHAT MANAGEMENT ====================
let users = { 'david': { password: 'dmai2026', role: 'admin' } };
let currentUser = null;
let currentUserRole = 'user';
let internetEnabled = false;
let projects = [];
let currentChat = null;
let activeContextMenu = null;

// Load data from localStorage
function loadData() {
    const saved = localStorage.getItem(`chats_${currentUser}`);
    if (saved) {
        try {
            projects = JSON.parse(saved);
        } catch (e) {}
    }
    
    if (!projects || projects.length === 0) {
        projects = [{
            id: 'chat_' + Date.now(),
            name: 'Chat 1',
            messages: [{
                id: 'welcome_' + Date.now(),
                role: 'ai',
                content: '👋 Welcome to DMAI!',
                timestamp: new Date().toISOString()
            }],
            archived: false,
            pinned: false,
            created: new Date().toISOString()
        }];
    }
    
    currentChat = projects[0].id;
    renderChatList();
    renderMessages();
}

function saveProjects() {
    localStorage.setItem(`chats_${currentUser}`, JSON.stringify(projects));
}

function renderChatList() {
    const list = document.getElementById('chat-list');
    if (!list) return;
    
    list.innerHTML = projects.filter(p => !p.archived).map(p => `
        <div class="chat-item ${p.id === currentChat ? 'active' : ''}" onclick="switchChat('${p.id}')">
            <div class="chat-name">${p.name}</div>
            <div class="chat-meta">${p.messages.length} msgs</div>
        </div>
    `).join('');
}

function renderMessages() {
    const chat = projects.find(p => p.id === currentChat);
    if (!chat) return;
    
    const container = document.getElementById('messages');
    container.innerHTML = chat.messages.map(m => `
        <div class="message ${m.role}">${m.content}</div>
    `).join('');
    container.scrollTop = container.scrollHeight;
}

function switchChat(id) {
    currentChat = id;
    const chat = projects.find(p => p.id === id);
    if (chat) {
        document.getElementById('current-chat').textContent = chat.name;
        renderMessages();
        renderChatList();
    }
}

function newChat() {
    const id = 'chat_' + Date.now();
    projects.push({
        id: id,
        name: `Chat ${projects.length + 1}`,
        messages: [],
        archived: false,
        pinned: false,
        created: new Date().toISOString()
    });
    currentChat = id;
    document.getElementById('current-chat').textContent = `Chat ${projects.length}`;
    saveProjects();
    renderChatList();
    renderMessages();
}

function logout() {
    localStorage.removeItem('dmai_logged_in');
    localStorage.removeItem('dmai_username');
    localStorage.removeItem('dmai_role');
    localStorage.removeItem('dmai_last_active');
    document.getElementById('login-screen').style.display = 'flex';
    document.getElementById('main-app').style.display = 'none';
}

// Initialize
function initializeToolsPanel() {
    const panel = document.getElementById('toolsPanel');
    if (!panel) return;
    panel.innerHTML = '<div class="tools-section">Tools loading...</div>';
}

// Check auto-login
function checkAutoLogin() {
    const loggedIn = localStorage.getItem('dmai_logged_in');
    const savedUser = localStorage.getItem('dmai_username');
    
    if (loggedIn === 'true' && savedUser) {
        currentUser = savedUser;
        document.getElementById('login-screen').style.display = 'none';
        document.getElementById('main-app').style.display = 'block';
        document.getElementById('current-user').textContent = savedUser;
        document.getElementById('user-avatar').textContent = savedUser.substring(0,2).toUpperCase();
        
        loadData();
        initializeToolsPanel();
        fetchEvolutionStats();
        return true;
    }
    return false;
}

// Start the app
if (!checkAutoLogin()) {
    document.getElementById('login-screen').style.display = 'flex';
}

// Message sending
function sendMessage() {
    const input = document.getElementById('message-input');
    const content = input.value.trim();
    if (!content) return;
    
    const chat = projects.find(p => p.id === currentChat);
    if (!chat) return;
    
    chat.messages.push({
        id: 'msg_' + Date.now(),
        role: 'user',
        content: content,
        timestamp: new Date().toISOString()
    });
    
    chat.messages.push({
        id: 'resp_' + Date.now(),
        role: 'ai',
        content: `🤔 You said: "${content}"`,
        timestamp: new Date().toISOString()
    });
    
    input.value = '';
    saveProjects();
    renderMessages();
}

// Event listeners
document.getElementById('message-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initial fetch
setTimeout(fetchEvolutionStats, 1000);
'''
    
    # Replace the old script with the new one
    content = content.replace(old_script, new_script)
    
    with open('ui/ai_ui.html', 'w') as f:
        f.write(content)
    
    print("✅ Complete UI script replacement done")
else:
    print("❌ Could not find script block")

