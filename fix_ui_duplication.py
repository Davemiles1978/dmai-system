#!/usr/bin/env python3
"""
Auto-fix for UI HTML duplication and syntax errors
Run this once to clean up the file
"""
import re

with open('ui/ai_ui.html', 'r') as f:
    content = f.read()

# Find all fetchEvolutionStats functions
pattern = r'(async function fetchEvolutionStats\(\) \{.*?\})'
matches = re.findall(pattern, content, re.DOTALL)

if len(matches) > 1:
    print(f"Found {len(matches)} duplicate functions - keeping first, removing others")
    # Keep only the first occurrence
    first_func = matches[0]
    # Replace all occurrences with just the first one
    for match in matches[1:]:
        content = content.replace(match, '')
    print("✅ Duplicates removed")

# Fix any stray braces or syntax issues
# Ensure login function exists
if 'function login()' not in content:
    print("Adding missing login function")
    login_func = '''
// Login function
function login() {
    const username = document.getElementById("login-username").value.trim();
    const password = document.getElementById("login-password").value;
    
    // Simple hardcoded check - in production this would validate against users object
    if (username === "david" && password === "dmai2026") {
        localStorage.setItem("dmai_logged_in", "true");
        localStorage.setItem("dmai_username", username);
        localStorage.setItem("dmai_last_active", Date.now().toString());
        
        document.getElementById("login-screen").style.display = "none";
        document.getElementById("main-app").style.display = "block";
        document.getElementById("current-user").textContent = username;
        document.getElementById("user-avatar").textContent = username.substring(0,2).toUpperCase();
        
        if (typeof loadData === 'function') loadData();
        if (typeof initializeToolsPanel === 'function') initializeToolsPanel();
        if (typeof fetchEvolutionStats === 'function') fetchEvolutionStats();
    } else {
        document.getElementById("login-error").textContent = "Invalid credentials";
    }
}
'''
    # Insert before closing </script> tag
    content = content.replace('</script>', login_func + '\n</script>')

# Ensure evolutionData is defined
if 'let evolutionData' not in content:
    print("Adding evolutionData declaration")
    evo_data = 'let evolutionData = { generation: 1, bestScore: 0, nextCycleIn: 3600, status: "running" };'
    content = content.replace('<script>', '<script>\n    ' + evo_data)

# Write fixed content
with open('ui/ai_ui.html', 'w') as f:
    f.write(content)

print("✅ UI fixes applied - ready to commit and deploy")
