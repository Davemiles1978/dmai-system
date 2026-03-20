// Fix button positioning
(function() {
    function fixButtonPosition() {
        const buttons = Array.from(document.querySelectorAll('button'));
        const chatsBtn = buttons.find(b => b.textContent.includes('Chats'));
        const toolsBtn = buttons.find(b => b.textContent.includes('Tools'));
        
        if (chatsBtn && toolsBtn) {
            // Create container if needed
            let container = chatsBtn.parentElement;
            if (!container.classList.contains('action-buttons')) {
                container.classList.add('action-buttons');
            }
            
            // Add classes to buttons
            chatsBtn.classList.add('chats-btn');
            toolsBtn.classList.add('tools-btn');
            
            // Apply responsive positioning
            if (window.innerWidth > 768) {
                container.style.position = 'fixed';
                container.style.top = '20px';
                container.style.right = '20px';
                container.style.zIndex = '1000';
                container.style.display = 'flex';
                container.style.gap = '10px';
                container.style.background = 'transparent';
            } else {
                container.style.position = 'sticky';
                container.style.bottom = '0';
                container.style.width = '100%';
                container.style.display = 'flex';
                container.style.justifyContent = 'space-around';
                container.style.background = 'white';
                container.style.boxShadow = '0 -2px 10px rgba(0,0,0,0.1)';
            }
        }
    }
    
    // Run on load and resize
    window.addEventListener('load', fixButtonPosition);
    window.addEventListener('resize', fixButtonPosition);
    
    // Run after any dynamic content changes
    const observer = new MutationObserver(fixButtonPosition);
    observer.observe(document.body, { childList: true, subtree: true });
})();
