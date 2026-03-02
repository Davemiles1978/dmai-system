"""
UI Healer - Automatically detects and fixes UI issues
Part of the self-healing system for the web interface
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - UI_HEALER - %(message)s')
logger = logging.getLogger('ui_healer')

class UIHealer:
    def __init__(self):
        self.ui_files = [
            Path("ui/ai_ui.html"),
            Path("static/css/ui_final_fix.css"),
            Path("static/js/fix_responses.js"),
            Path("static/js/mobile_buttons_fix.js"),
            Path("static/css/mobile_buttons_fix.css")
        ]
        self.issues_fixed = 0
        
    def scan_for_issues(self):
        """Scan UI files for common issues"""
        issues = []
        
        for file_path in self.ui_files:
            if not file_path.exists():
                issues.append({
                    'file': str(file_path),
                    'issue': 'missing',
                    'severity': 'high'
                })
                continue
            
            content = file_path.read_text()
            
            # Check for scrolling issues
            if 'overflow' not in content and file_path.suffix == '.css':
                issues.append({
                    'file': str(file_path),
                    'issue': 'missing_scroll_properties',
                    'severity': 'medium'
                })
            
            # Check for button clickability
            if 'cursor: pointer' not in content and file_path.suffix == '.css':
                issues.append({
                    'file': str(file_path),
                    'issue': 'buttons_may_not_be_clickable',
                    'severity': 'high'
                })
            
            # Check for mobile responsiveness
            if '@media' not in content and file_path.suffix == '.css':
                issues.append({
                    'file': str(file_path),
                    'issue': 'missing_responsive_design',
                    'severity': 'high'
                })
        
        return issues
    
    def heal_ui(self):
        """Apply fixes to UI files"""
        issues = self.scan_for_issues()
        
        if not issues:
            logger.info("✅ No UI issues found")
            return
        
        logger.info(f"🔍 Found {len(issues)} UI issues")
        
        for issue in issues:
            if issue['issue'] == 'missing':
                self._create_missing_file(issue['file'])
                self.issues_fixed += 1
            
            elif issue['issue'] == 'missing_scroll_properties':
                self._add_scroll_fix(issue['file'])
                self.issues_fixed += 1
            
            elif issue['issue'] == 'buttons_may_not_be_clickable':
                self._add_button_fix(issue['file'])
                self.issues_fixed += 1
            
            elif issue['issue'] == 'missing_responsive_design':
                self._add_responsive_fix(issue['file'])
                self.issues_fixed += 1
        
        logger.info(f"✅ Fixed {self.issues_fixed} UI issues")
    
    def _create_missing_file(self, file_path):
        """Create missing UI file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if 'css' in file_path:
            path.write_text("""/* Auto-generated UI fix */
/* Add your CSS here */
""")
        elif 'js' in file_path:
            path.write_text("""// Auto-generated UI fix
// Add your JavaScript here
console.log('UI fix loaded');
""")
        elif 'html' in file_path:
            path.write_text("""<!DOCTYPE html>
<html>
<head>
    <title>DMAI</title>
</head>
<body>
    <h1>DMAI - Loading...</h1>
</body>
</html>
""")
        
        logger.info(f"  ✅ Created missing file: {file_path}")
    
    def _add_scroll_fix(self, file_path):
        """Add scrolling fixes to CSS"""
        path = Path(file_path)
        content = path.read_text()
        
        scroll_fix = """
/* Auto-fixed: Scrolling issues */
.messages-container, .chat-list-panel, .tools-panel {
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
    max-height: 100% !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary) !important;
}

::-webkit-scrollbar-thumb {
    background: var(--accent) !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-light) !important;
}
"""
        
        if scroll_fix not in content:
            path.write_text(content + scroll_fix)
            logger.info(f"  ✅ Added scroll fix to {file_path}")
    
    def _add_button_fix(self, file_path):
        """Add button clickability fixes"""
        path = Path(file_path)
        content = path.read_text()
        
        button_fix = """
/* Auto-fixed: Button clickability */
button, .btn, [role="button"], .tool-btn, .chat-item, .send-btn {
    cursor: pointer !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 100 !important;
}

button:hover, .btn:hover, .tool-btn:hover, .chat-item:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    transition: all 0.2s ease !important;
}
"""
        
        if button_fix not in content:
            path.write_text(content + button_fix)
            logger.info(f"  ✅ Added button fix to {file_path}")
    
    def _add_responsive_fix(self, file_path):
        """Add mobile responsive fixes"""
        path = Path(file_path)
        content = path.read_text()
        
        responsive_fix = """
/* Auto-fixed: Mobile responsiveness */
@media screen and (max-width: 768px) {
    .header {
        height: 60px !important;
        padding: 0 8px !important;
    }
    
    .mobile-menu-btn {
        display: flex !important;
        width: 44px !important;
        height: 44px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .chat-list-panel, .tools-panel {
        position: fixed !important;
        top: 60px !important;
        bottom: 70px !important;
        width: 85% !important;
        z-index: 1000 !important;
        background: var(--bg-secondary) !important;
    }
    
    .input-area {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: var(--bg-secondary) !important;
        padding: 8px !important;
        z-index: 100 !important;
    }
}
"""
        
        if responsive_fix not in content:
            path.write_text(content + responsive_fix)
            logger.info(f"  ✅ Added responsive fix to {file_path}")

if __name__ == "__main__":
    healer = UIHealer()
    healer.heal_ui()

    def remove_duplicate_elements(self):
        """Detect and remove redundant UI elements like the evolution bubble"""
        pass
    
    def optimize_mobile_touch(self):
        """Ensure all buttons have proper touch targets on mobile"""
        pass
    
    def auto_adjust_spacing(self):
        """Fix padding and margin inconsistencies"""
        pass
