// Ensure mobile buttons are properly sized
(function() {
    function fixMobileButtons() {
        const menuBtn = document.getElementById('menuToggle');
        const toolsBtn = document.getElementById('toolsToggle');
        
        if (menuBtn) {
            menuBtn.style.width = '40px';
            menuBtn.style.height = '40px';
            menuBtn.style.padding = '0';
            menuBtn.style.fontSize = '20px';
            menuBtn.style.lineHeight = '40px';
        }
        
        if (toolsBtn) {
            toolsBtn.style.width = '40px';
            toolsBtn.style.height = '40px';
            toolsBtn.style.padding = '0';
            toolsBtn.style.fontSize = '20px';
            toolsBtn.style.lineHeight = '40px';
        }
        
        // On desktop, hide them
        if (window.innerWidth > 768) {
            if (menuBtn) menuBtn.style.display = 'none';
            if (toolsBtn) toolsBtn.style.display = 'none';
        } else {
            if (menuBtn) menuBtn.style.display = 'block';
            if (toolsBtn) toolsBtn.style.display = 'block';
        }
    }
    
    // Run on load and resize
    window.addEventListener('load', fixMobileButtons);
    window.addEventListener('resize', fixMobileButtons);
    
    // Run after any potential DOM changes
    setTimeout(fixMobileButtons, 500);
    setTimeout(fixMobileButtons, 1000);
})();
