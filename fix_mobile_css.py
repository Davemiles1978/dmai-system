#!/usr/bin/env python3
"""
Automated Mobile CSS Fix for DMAI
This script safely updates the CSS media query section in ai_ui.html
"""

import re
from pathlib import Path

def fix_mobile_css():
    print("🚀 Starting automated mobile CSS fix...")
    
    # Read the current file
    file_path = Path("ai_ui.html")
    if not file_path.exists():
        print("❌ ai_ui.html not found!")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Backup the file
    backup_path = Path("ai_ui.html.mobile_backup")
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created at {backup_path}")
    
    # Find the current mobile media query section
    pattern = r'@media\s*\(max-width:\s*768px\)\s*\{[^}]*\}[^}]*\}'
    
    # The new improved CSS
    new_css = """/* Mobile Responsive Fix - Works on ALL devices */
@media (max-width: 768px) {
    #app {
        grid-template-columns: 1fr !important;
        grid-template-rows: 60px 1fr 80px !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        width: 100vw !important;
        max-width: 100% !important;
    }
    
    .chat-list-panel, .tools-panel {
        display: none;
        position: fixed;
        top: 60px;
        left: 0;
        width: 100% !important;
        height: calc(100% - 140px);
        z-index: 1000;
        background: var(--bg-secondary);
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
    }
    
    .chat-list-panel.active, .tools-panel.active {
        display: block;
    }
    
    .main-chat-panel {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    
    .messages-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 15px !important;
        overflow-x: hidden !important;
        word-wrap: break-word !important;
    }
    
    .message {
        max-width: 95% !important;
        width: auto !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    .message.ai, .message.user {
        width: auto !important;
        max-width: 95% !important;
    }
    
    .input-area {
        width: 100% !important;
        padding: 10px !important;
    }
    
    .input-container {
        width: 100% !important;
    }
    
    textarea {
        width: calc(100% - 70px) !important;
        max-width: calc(100% - 70px) !important;
    }
    
    .mobile-menu-bar {
        display: flex !important;
        position: fixed;
        bottom: 80px;
        left: 0;
        width: 100%;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border);
        padding: 10px;
        z-index: 100;
    }
    
    .mobile-menu-btn {
        flex: 1;
        padding: 12px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
        margin: 0 5px;
        font-size: 16px;
        cursor: pointer;
        -webkit-tap-highlight-color: transparent;
    }
    
    /* Ensure no horizontal scroll */
    body, html {
        overflow-x: hidden !important;
        max-width: 100% !important;
    }
    
    /* Fix for images and content */
    img, pre, code {
        max-width: 100% !important;
        height: auto !important;
    }
}"""
    
    # Check if the pattern exists
    if re.search(pattern, content, re.DOTALL):
        # Replace the existing media query
        new_content = re.sub(pattern, new_css, content, flags=re.DOTALL)
        print("✅ Found and replaced existing mobile CSS")
    else:
        # If not found, append at the end of style section
        print("⚠️ Existing mobile CSS not found, appending new one")
        style_end = content.rfind('</style>')
        if style_end != -1:
            new_content = content[:style_end] + '\n' + new_css + '\n' + content[style_end:]
        else:
            print("❌ Could not find </style> tag")
            return False
    
    # Write the updated content
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully updated ai_ui.html with mobile fixes")
    print("\n📱 Changes made:")
    print("   • Fixed horizontal scrolling")
    print("   • Improved message wrapping")
    print("   • Enhanced mobile menu buttons")
    print("   • Added touch-friendly scrolling")
    print("   • Ensured all content fits screen width")
    
    return True

def verify_fix():
    """Quick verification of the fix"""
    with open("ai_ui.html", 'r') as f:
        content = f.read()
    
    checks = [
        ("overflow-x: hidden" in content, "✅ No horizontal scroll"),
        ("max-width: 95%" in content, "✅ Messages fit screen"),
        ("mobile-menu-bar" in content, "✅ Mobile menu present"),
        ("-webkit-overflow-scrolling: touch" in content, "✅ Touch scrolling enabled")
    ]
    
    print("\n🔍 Verification:")
    all_good = True
    for passed, message in checks:
        print(f"   {message}")
        if not passed:
            all_good = False
    
    if all_good:
        print("\n🎉 All mobile fixes verified!")
    else:
        print("\n⚠️ Some fixes may need manual review")
    
    return all_good

if __name__ == "__main__":
    print("="*60)
    print("📱 DMAI Mobile CSS Fixer")
    print("="*60)
    
    if fix_mobile_css():
        verify_fix()
        print("\n🚀 Next steps:")
        print("   git add ai_ui.html")
        print("   git commit -m 'Fix mobile responsiveness'")
        print("   git push origin main")
        print("   # Then deploy on Render")
    else:
        print("\n❌ Fix failed - please check manually")
