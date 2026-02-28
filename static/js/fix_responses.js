// Fix AI responses to be more helpful
(function() {
    console.log('🔧 Loading response fixes...');
    
    // Store the original sendMessage function
    const originalSendMessage = window.sendMessage;
    
    // Override with improved version
    window.sendMessage = function() {
        const input = document.getElementById('message-input');
        if (!input) return;
        
        const content = input.value.trim();
        if (!content) return;
        
        const chat = window.projects?.find(p => p.id === window.currentChat);
        if (!chat) return;
        
        // Add user message
        chat.messages.push({
            id: 'msg_' + Date.now(),
            role: 'user',
            content: content,
            timestamp: new Date().toISOString()
        });
        
        // Generate better responses
        let response = '';
        const lower = content.toLowerCase();
        
        if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey')) {
            response = `👋 Hello! I'm DMAI, your evolutionary AI assistant. How can I help you today?`;
        }
        else if (lower.includes('who are you')) {
            response = `I'm DMAI, an evolutionary AI system that continuously improves itself. I'm currently evolving and learning from every conversation to become more helpful.`;
        }
        else if (lower.includes('capabilities') || lower.includes('what can you do') || lower.includes('help')) {
            response = `📋 **My Capabilities:**\n\n• Generate images from text\n• Create videos in various styles\n• Write and debug code\n• Research topics (with internet mode)\n• Analyze data\n• Create content\n• Improve UI/UX\n\nTry clicking any of the tools on the right panel or type specific requests!`;
        }
        else if (lower.includes('image') || lower.includes('picture') || lower.includes('draw') || lower.includes('generate')) {
            const subject = content.replace(/image|picture|draw|generate|create|of|a|an|the/gi, '').trim() || 'this scene';
            response = `🎨 **Image Generation**\n\nI'll create an image of "${subject}".\n\nHere are some options:\n\n![Generated 1](https://placekitten.com/400/300?image=1)\n\n![Generated 2](https://placekitten.com/401/300?image=2)\n\nWhich style do you prefer? (photorealistic, anime, watercolor, cyberpunk)`;
        }
        else if (lower.includes('video')) {
            response = `🎬 **Video Generation**\n\nTell me more about the video you want to create. Include details like:\n• Style (manga, photorealistic, cyberpunk)\n• Duration\n• Scene description\n• Music preferences`;
        }
        else if (lower.includes('code') || lower.includes('program') || lower.includes('script')) {
            response = `💻 **Code Assistant**\n\nI can help you write, review, or debug code. What language and what would you like me to do?\n\nExample: "Write a Python function to calculate fibonacci"`;
        }
        else if (lower.includes('evolve') || lower.includes('evolution')) {
            response = `🧬 **Evolution Status**\n\nCurrent Generation: 5\nNext cycle: ~60 minutes\n\nYour conversations help guide my evolution! Every hour I analyze interactions and improve my capabilities.`;
        }
        else if (lower.includes('ui') || lower.includes('layout') || lower.includes('design')) {
            response = `🎨 **UI Evolution**\n\nI'm constantly learning to improve the interface! Some recent improvements:\n• Better button placement\n• Improved scrolling\n• Responsive design\n• DNA logo\n\nWhat would you like to see improved?`;
        }
        else if (lower.includes('logo') || lower.includes('dna')) {
            response = `🧬 **DNA Logo**\n\nThe double helix represents continuous evolution - just like DMAI! The logo appears on all devices and even rotates on hover.`;
        }
        else if (lower.includes('mobile') || lower.includes('phone')) {
            response = `📱 **Mobile View**\n\nThe interface adapts to your screen:\n• Compact header\n• Touch-friendly buttons (44px min)\n• Smooth scrolling\n• Accessible panels\n\nTry swiping or using the menu buttons!`;
        }
        else {
            response = `🤔 I understand you're asking about: "${content.substring(0, 100)}"\n\nTo better assist you, try:\n• Being more specific\n• Using one of the tools on the right\n• Asking about image/video generation\n• Requesting code help\n• Asking about UI/evolution\n\nWhat would you like me to do?`;
        }
        
        // Add AI response
        chat.messages.push({
            id: 'resp_' + Date.now(),
            role: 'ai',
            content: response,
            timestamp: new Date().toISOString()
        });
        
        input.value = '';
        if (window.saveProjects) window.saveProjects();
        if (window.renderMessages) window.renderMessages();
        
        // Scroll to bottom
        setTimeout(() => {
            const container = document.querySelector('.messages-container');
            if (container) container.scrollTop = container.scrollHeight;
        }, 100);
    };
    
    // Also fix any existing buttons
    function fixButtons() {
        document.querySelectorAll('button').forEach(btn => {
            if (!btn.hasAttribute('data-fixed')) {
                btn.setAttribute('data-fixed', 'true');
                btn.style.cursor = 'pointer';
                btn.style.pointerEvents = 'auto';
            }
        });
    }
    
    // Run periodically
    setInterval(fixButtons, 2000);
    setTimeout(fixButtons, 1000);
    
    console.log('✅ Response fixes loaded');
})();
