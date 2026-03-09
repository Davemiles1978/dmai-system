// Move admin controls to dropdown and fix scrolling
(function() {
    function moveAdminControls() {
        // Only run on desktop
        if (window.innerWidth <= 768) return;
        
        // Find admin controls in tools panel
        const adminSection = document.querySelector('.tools-section:has(.danger)');
        if (!adminSection) return;
        
        // Find dropdown
        const dropdown = document.querySelector('.dropdown-content');
        if (!dropdown) return;
        
        // Check if already moved
        if (document.getElementById('admin-menu-items')) return;
        
        // Create admin section in dropdown
        const adminDiv = document.createElement('div');
        adminDiv.id = 'admin-menu-items';
        adminDiv.className = 'dropdown-item admin-section';
        adminDiv.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.8em; padding: 5px 0;">👑 ADMIN</div>';
        
        // Add pause button
        const pauseItem = document.createElement('div');
        pauseItem.className = 'dropdown-item danger-item';
        pauseItem.textContent = '⏸️ Pause Evolution';
        pauseItem.onclick = function(e) {
            e.stopPropagation();
            if (typeof emergencyPause === 'function') emergencyPause();
        };
        
        // Add kill button
        const killItem = document.createElement('div');
        killItem.className = 'dropdown-item danger-item';
        killItem.textContent = '⛔ Kill System';
        killItem.onclick = function(e) {
            e.stopPropagation();
            if (typeof emergencyKill === 'function') emergencyKill();
        };
        
        adminDiv.appendChild(pauseItem);
        adminDiv.appendChild(killItem);
        
        // Add to dropdown before logout
        const logoutItem = Array.from(dropdown.children).find(el => el.textContent.includes('Logout'));
        if (logoutItem) {
            dropdown.insertBefore(adminDiv, logoutItem);
        } else {
            dropdown.appendChild(adminDiv);
        }
        
        // Hide original admin controls
        if (adminSection) {
            adminSection.style.display = 'none';
        }
    }
    
    function ensureInputVisible() {
        if (window.innerWidth <= 768) {
            const inputArea = document.querySelector('.input-area');
            if (inputArea) {
                inputArea.style.display = 'block';
                document.querySelector('.main-chat-panel').style.marginBottom = '70px';
            }
        }
    }
    
    // Run on load
    window.addEventListener('load', function() {
        moveAdminControls();
        ensureInputVisible();
    });
    
    // Run on resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            moveAdminControls();
        } else {
            ensureInputVisible();
        }
    });
    
    // Run after a delay
    setTimeout(moveAdminControls, 1000);
    setTimeout(moveAdminControls, 2000);
})();
