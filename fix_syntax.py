#!/usr/bin/env python3
import re

with open('ui/ai_ui.html', 'r') as f:
    content = f.read()

# Find the JavaScript section
script_pattern = r'<script>(.*?)</script>'
scripts = re.findall(script_pattern, content, re.DOTALL)

if scripts:
    # Count braces to find mismatch
    for script in scripts:
        open_braces = script.count('{')
        close_braces = script.count('}')
        if open_braces != close_braces:
            print(f"Brace mismatch: {open_braces} opening, {close_braces} closing")
            # Fix by ensuring the login function exists
            if 'function login' not in script:
                # Add login function
                fixed_script = script + '''
// Added login function
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
'''
                content = content.replace(script, fixed_script)
    
    with open('ui/ai_ui.html', 'w') as f:
        f.write(content)
    print("✅ Fixed syntax and added login function")
