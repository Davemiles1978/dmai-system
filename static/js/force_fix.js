// FORCE FIX - Directly override and fix everything
(function() {
    console.log('🔧 FORCE FIX: Loading...');
    
    // Wait for page to be fully loaded
    function waitForElements() {
        const input = document.getElementById('message-input');
        const sendBtn = document.querySelector('.send-btn');
        
        if (!input || !sendBtn) {
            console.log('⏳ Waiting for elements...');
            setTimeout(waitForElements, 500);
            return;
        }
        
        console.log('✅ Elements found, applying fixes...');
        
        // Completely replace the sendMessage function on window
        window.sendMessage = function() {
            console.log('📤 sendMessage called');
            const input = document.getElementById('message-input');
            if (!input) return;
            
            const content = input.value.trim();
            if (!content) return;
            
            // Find or create chat
            if (!window.projects) {
                window.projects = [{
                    id: 'chat_1',
                    name: 'Chat 1',
                    messages: [],
                    created: new Date().toISOString()
                }];
                window.currentChat = 'chat_1';
            }
            
            if (!window.currentChat) {
                window.currentChat = window.projects[0].id;
            }
            
            const chat = window.projects.find(p => p.id === window.currentChat);
            if (!chat) return;
            
            // Add user message
            const userMsg = {
                id: 'msg_' + Date.now(),
                role: 'user',
                content: content,
                timestamp: new Date().toISOString()
            };
            chat.messages.push(userMsg);
            
            // Generate response
            let response = '';
            const lower = content.toLowerCase();
            
            if (lower.includes('hello') || lower.includes('hi')) {
                response = '👋 Hello! How can I help you today?';
            } else if (lower.includes('image') || lower.includes('picture')) {
                response = '🎨 Try the image generation tools on the right panel!';
            } else {
                response = `🤔 You said: "${content}"\n\nHow can I assist further?`;
            }
            
            // Add AI response
            const aiMsg = {
                id: 'resp_' + Date.now(),
                role: 'ai',
                content: response,
                timestamp: new Date().toISOString()
            };
            chat.messages.push(aiMsg);
            
            // Clear input
            input.value = '';
            
            // Update display
            if (window.renderMessages) {
                window.renderMessages();
            } else {
                // Manual render if function missing
                const container = document.getElementById('messages');
                if (container) {
                    container.innerHTML = chat.messages.map(m => 
                        `<div class="message ${m.role}">${m.content}</div>`
                    ).join('');
                }
            }
            
            console.log('✅ Message sent and response added');
        };
        
        // Attach click handler to send button
        sendBtn.onclick = function(e) {
            e.preventDefault();
            window.sendMessage();
        };
        
        // Attach enter key to input
        input.onkeydown = function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                window.sendMessage();
            }
        };
        
        // Create messages container if missing
        if (!document.getElementById('messages')) {
            const container = document.createElement('div');
            container.id = 'messages';
            container.className = 'messages-container';
            document.querySelector('.main-chat-panel')?.appendChild(container);
        }
        
        console.log('✅ FORCE FIX complete');
    }
    
    // Start waiting
    waitForElements();
})();
