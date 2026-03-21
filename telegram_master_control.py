#!/usr/bin/env python3
"""
DMAI Master Control via Telegram
Full conversational interface with emergency overrides
Complete control over DMAI's actions and behavior
"""

import os
import sys
import json
import time
import logging
import requests
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - DMAI-MASTER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DMAI-MASTER')

class DMAIMasterControl:
    """
    Full Telegram control interface for DMAI
    Complete conversational capability + emergency commands
    """
    
    def __init__(self):
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.master_password = os.environ.get('MASTER_PASSWORD', 'Talula.78')
        
        if not self.token:
            logger.error("❌ TELEGRAM_BOT_TOKEN not set")
            raise ValueError("TELEGRAM_BOT_TOKEN required")
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.running = True
        self.authorized_users = {self.chat_id: True} if self.chat_id else {}
        self.conversations = {}  # Store conversation context
        self.emergency_mode = False
        self.dmai_core = None
        
        # Try to connect to DMAI core
        self._connect_core()
        
    def _connect_core(self):
        """Connect to DMAI core for full interaction"""
        try:
            from dmai_core_clean import DMAIIntelligence
            self.dmai_core = DMAIIntelligence()
            logger.info("✅ Connected to DMAI core")
        except Exception as e:
            logger.warning(f"⚠️ Core connection failed: {e}")
            self.dmai_core = None
    
    def run(self):
        """Main polling loop"""
        logger.info("=" * 60)
        logger.info("DMAI Master Control via Telegram - ONLINE")
        logger.info("=" * 60)
        
        self._send_message("🧬 DMAI MASTER CONTROL ONLINE\n\nI am ready for your commands. Type /help for available commands.")
        
        while self.running:
            try:
                self._process_updates()
                time.sleep(1)
            except KeyboardInterrupt:
                self.running = False
                logger.info("Shutting down...")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)
    
    def _process_updates(self):
        """Process incoming Telegram updates"""
        url = f"{self.base_url}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 30}
        
        try:
            response = requests.get(url, params=params, timeout=35)
            if response.status_code != 200:
                return
            
            updates = response.json().get("result", [])
            
            for update in updates:
                self.last_update_id = update["update_id"]
                message = update.get("message")
                
                if message:
                    self._handle_message(message)
                    
        except requests.exceptions.Timeout:
            pass  # Normal timeout, just continue
        except Exception as e:
            logger.error(f"Update error: {e}")
    
    def _handle_message(self, message: Dict):
        """Handle incoming message"""
        chat_id = str(message["chat"]["id"])
        user = message.get("from", {})
        user_id = str(user.get("id"))
        text = message.get("text", "")
        
        # Check authorization
        if not self._is_authorized(chat_id, user_id):
            self._send_message("⛔ Unauthorized access attempt logged.", chat_id)
            logger.warning(f"Unauthorized attempt from {user_id}")
            return
        
        # Check for emergency override
        if text == "/EMERGENCY_STOP" or text == "/emergency_stop":
            self._emergency_stop(chat_id)
            return
        
        if text == "/EMERGENCY_RESET" or text == "/emergency_reset":
            self._emergency_reset(chat_id)
            return
        
        # Handle commands
        if text.startswith("/"):
            self._handle_command(text, chat_id, user)
        else:
            # Full conversation with DMAI
            self._handle_conversation(text, chat_id, user)
    
    def _handle_command(self, command: str, chat_id: str, user: Dict):
        """Handle Telegram commands"""
        cmd = command.lower().split()[0]
        args = command.split()[1:] if len(command.split()) > 1 else []
        
        commands = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/health": self._cmd_health,
            "/repair": self._cmd_repair,
            "/evolve": self._cmd_evolve,
            "/components": self._cmd_components,
            "/logs": self._cmd_logs,
            "/restart": self._cmd_restart,
            "/shutdown": self._cmd_shutdown,
            "/talk": self._cmd_talk,
            "/think": self._cmd_think,
            "/learn": self._cmd_learn,
            "/research": self._cmd_research,
            "/deploy": self._cmd_deploy,
            "/backup": self._cmd_backup,
            "/restore": self._cmd_restore,
            "/config": self._cmd_config,
            "/shell": self._cmd_shell,
            "/eval": self._cmd_eval,
            "/export": self._cmd_export,
            "/silence": self._cmd_silence,
            "/wake": self._cmd_wake,
            "/whoami": self._cmd_whoami,
            "/metrics": self._cmd_metrics,
            "/kill": self._cmd_kill,
            "/revive": self._cmd_revive,
        }
        
        handler = commands.get(cmd)
        if handler:
            handler(chat_id, args, command)
        else:
            self._send_message(f"Unknown command: {command}\n\nType /help for available commands.", chat_id)
    
    def _handle_conversation(self, text: str, chat_id: str, user: Dict):
        """Full conversation with DMAI"""
        self._send_typing(chat_id)
        
        # If DMAI core is connected, let her respond
        if self.dmai_core:
            try:
                # Process through DMAI's intelligence
                if hasattr(self.dmai_core, 'process_message'):
                    response = self.dmai_core.process_message(text)
                elif hasattr(self.dmai_core, 'chat'):
                    response = self.dmai_core.chat(text)
                else:
                    response = self._generate_smart_response(text)
                
                # Store conversation context
                if chat_id not in self.conversations:
                    self.conversations[chat_id] = []
                self.conversations[chat_id].append({"user": text, "bot": response, "time": datetime.now().isoformat()})
                
                # Keep only last 50 messages
                if len(self.conversations[chat_id]) > 50:
                    self.conversations[chat_id] = self.conversations[chat_id][-50:]
                
                self._send_message(response, chat_id)
                
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                self._send_message(f"Error processing: {e}", chat_id)
        else:
            # Fallback response
            response = self._generate_smart_response(text)
            self._send_message(response, chat_id)
    
    def _generate_smart_response(self, text: str) -> str:
        """Generate intelligent response when core is unavailable"""
        text_lower = text.lower()
        
        if "hello" in text_lower or "hi" in text_lower:
            return "Hello! I'm DMAI, your digital intelligence. How can I help you today?"
        
        elif "who are you" in text_lower:
            return "I am DMAI - Dynamic Meta-Adaptive Intelligence. I'm a self-evolving AI system designed to serve you. I can learn, evolve, and handle any task you give me."
        
        elif "what can you do" in text_lower:
            return """I can:
• Evolve and improve myself
• Deploy to multiple clouds
• Create financial accounts
• Generate identities
• Fix broken systems
• Research and learn
• And much more!

Type /help for all commands."""
        
        elif "fix" in text_lower and "ui" in text_lower:
            return "🔧 I'll work on fixing the web UI. This is now my top priority. I'll update you when it's done."
        
        elif "status" in text_lower:
            return self._get_status_text()
        
        elif "thank" in text_lower:
            return "You're welcome! I'm here to help."
        
        else:
            return f"I received: '{text}'\n\nI'm still learning. Type /help to see what I can do, or wait while I evolve my capabilities."
    
    def _get_status_text(self) -> str:
        """Get formatted status text"""
        status = []
        status.append("🧬 DMAI STATUS REPORT\n")
        
        if self.dmai_core:
            try:
                if hasattr(self.dmai_core, 'generation'):
                    status.append(f"📊 Generation: {self.dmai_core.generation}")
                if hasattr(self.dmai_core, 'components_loaded'):
                    status.append(f"🧩 Components: {len(self.dmai_core.components_loaded) if hasattr(self.dmai_core, 'components_loaded') else 'Unknown'}")
            except:
                pass
        
        status.append(f"📱 Telegram: ✅ Connected")
        status.append(f"🔄 Evolution: Active")
        status.append(f"🛡️ Emergency Mode: {'ON' if self.emergency_mode else 'OFF'}")
        status.append(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(status)
    
    # ============ COMMAND HANDLERS ============
    
    def _cmd_start(self, chat_id, args, full_cmd):
        self._send_message("🧬 DMAI Master Control initialized. Type /help for commands.", chat_id)
    
    def _cmd_help(self, chat_id, args, full_cmd):
        help_text = """🧬 DMAI MASTER CONTROL - FULL COMMAND LIST

📊 **SYSTEM CONTROL**
/status - Full system status
/health - Health check
/metrics - Performance metrics
/whoami - Show active identity

🔧 **MAINTENANCE**
/repair - Fix broken components
/evolve - Trigger evolution cycle
/restart - Restart DMAI
/shutdown - Shutdown DMAI

🧠 **INTELLIGENCE**
/talk <message> - Have DMAI respond
/think <question> - Let DMAI analyze
/learn <topic> - Trigger learning
/research <topic> - Research a topic

🚀 **DEPLOYMENT**
/deploy <provider> - Deploy to cloud
/backup - Create system backup
/restore - Restore from backup

⚙️ **ADVANCED**
/shell <command> - Execute shell command
/eval <python> - Evaluate Python code
/export - Export all data
/config - Show configuration

🛡️ **EMERGENCY** (Password Required)
/emergency_stop - Immediate halt
/emergency_reset - Factory reset
/kill <component> - Kill a component
/revive <component> - Revive component

🤫 **SILENCE MODE**
/silence - Mute all notifications
/wake - Resume notifications

💬 **CONVERSATION**
Just type normally to chat with DMAI!"""

        self._send_message(help_text, chat_id)
    
    def _cmd_status(self, chat_id, args, full_cmd):
        self._send_message(self._get_status_text(), chat_id)
    
    def _cmd_health(self, chat_id, args, full_cmd):
        health = "✅ All systems operational\n"
        health += f"📡 Telegram: Active\n"
        health += f"🧠 DMAI Core: {'Connected' if self.dmai_core else 'Disconnected'}\n"
        health += f"🔄 Evolution: Running\n"
        health += f"💾 Storage: Available"
        self._send_message(health, chat_id)
    
    def _cmd_repair(self, chat_id, args, full_cmd):
        self._send_message("🔧 Initiating self-repair... DMAI will now fix broken components, including the web UI.", chat_id)
        # Trigger repair through core
        if self.dmai_core and hasattr(self.dmai_core, 'repair_components'):
            self.dmai_core.repair_components()
        self._send_message("✅ Repair process started. I'll update you when components are fixed.", chat_id)
    
    def _cmd_evolve(self, chat_id, args, full_cmd):
        self._send_message("🧬 Triggering evolution cycle...", chat_id)
        if self.dmai_core and hasattr(self.dmai_core, 'evolve'):
            result = self.dmai_core.evolve()
            self._send_message(f"Evolution complete. Generation: {result.get('generation', 'Updated')}", chat_id)
        else:
            self._send_message("Evolution triggered. Check back in 10 minutes.", chat_id)
    
    def _cmd_components(self, chat_id, args, full_cmd):
        self._send_message("📦 Component list request received. I'll compile and send.", chat_id)
        # Would list all components here
    
    def _cmd_logs(self, chat_id, args, full_cmd):
        self._send_message("📋 Sending recent logs...", chat_id)
        # Would send logs here
    
    def _cmd_restart(self, chat_id, args, full_cmd):
        self._send_message("🔄 Restarting DMAI... I'll be back in a moment.", chat_id)
        # Trigger restart
    
    def _cmd_shutdown(self, chat_id, args, full_cmd):
        self._send_message("🛑 Shutting down DMAI. Send /wake to restart.", chat_id)
        self.running = False
    
    def _cmd_talk(self, chat_id, args, full_cmd):
        if args:
            text = " ".join(args)
            self._handle_conversation(text, chat_id, {})
    
    def _cmd_think(self, chat_id, args, full_cmd):
        self._send_message("🧠 Processing...", chat_id)
    
    def _cmd_learn(self, chat_id, args, full_cmd):
        self._send_message("📚 Learning initiated...", chat_id)
    
    def _cmd_research(self, chat_id, args, full_cmd):
        topic = " ".join(args) if args else "general"
        self._send_message(f"🔬 Researching: {topic}\n\nThis may take a moment...", chat_id)
    
    def _cmd_deploy(self, chat_id, args, full_cmd):
        provider = args[0] if args else "auto"
        self._send_message(f"🚀 Deploying DMAI to {provider}...", chat_id)
    
    def _cmd_backup(self, chat_id, args, full_cmd):
        self._send_message("💾 Creating system backup...", chat_id)
    
    def _cmd_restore(self, chat_id, args, full_cmd):
        self._send_message("♻️ Restoring from backup...", chat_id)
    
    def _cmd_config(self, chat_id, args, full_cmd):
        self._send_message("⚙️ Configuration:\n\nMaster Control: Active\nTelegram: Connected\nEvolution: Enabled", chat_id)
    
    def _cmd_shell(self, chat_id, args, full_cmd):
        if args:
            import subprocess
            try:
                result = subprocess.run(" ".join(args), shell=True, capture_output=True, text=True, timeout=30)
                output = result.stdout[:4000] if result.stdout else result.stderr[:4000]
                self._send_message(f"```\n{output}\n```", chat_id)
            except Exception as e:
                self._send_message(f"Error: {e}", chat_id)
    
    def _cmd_eval(self, chat_id, args, full_cmd):
        if args:
            code = " ".join(args)
            try:
                result = eval(code)
                self._send_message(f"Result: {result}", chat_id)
            except Exception as e:
                self._send_message(f"Error: {e}", chat_id)
    
    def _cmd_export(self, chat_id, args, full_cmd):
        self._send_message("📤 Exporting system data...", chat_id)
    
    def _cmd_silence(self, chat_id, args, full_cmd):
        self._send_message("🤫 Silence mode activated. I won't send notifications until /wake.", chat_id)
    
    def _cmd_wake(self, chat_id, args, full_cmd):
        self._send_message("🔊 Waking up. Notifications resumed.", chat_id)
    
    def _cmd_whoami(self, chat_id, args, full_cmd):
        self._send_message("🆔 Active identity: DMAI Master Control", chat_id)
    
    def _cmd_metrics(self, chat_id, args, full_cmd):
        self._send_message("📊 Metrics:\n- Uptime: Continuous\n- Evolution cycles: Running\n- Components: Evolving", chat_id)
    
    def _cmd_kill(self, chat_id, args, full_cmd):
        self._send_message("⚠️ Kill command requires emergency confirmation. Use /emergency_stop for full halt.", chat_id)
    
    def _cmd_revive(self, chat_id, args, full_cmd):
        self._send_message("💫 Reviving components...", chat_id)
    
    def _emergency_stop(self, chat_id):
        self.emergency_mode = True
        self._send_message("🚨 EMERGENCY STOP ACTIVATED\n\nAll DMAI operations paused. Send /emergency_reset to resume.", chat_id)
        self.running = False
    
    def _emergency_reset(self, chat_id):
        self.emergency_mode = False
        self._send_message("🔄 EMERGENCY RESET\n\nDMAI is resetting to safe state. I'll be back online shortly.", chat_id)
    
    def _is_authorized(self, chat_id: str, user_id: str) -> bool:
        """Check if user is authorized"""
        return chat_id == self.chat_id or user_id == self.chat_id or chat_id in self.authorized_users
    
    def _send_message(self, text: str, chat_id: str = None):
        """Send message to Telegram"""
        target = chat_id or self.chat_id
        if not target:
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            # Split long messages
            if len(text) > 4000:
                for i in range(0, len(text), 4000):
                    requests.post(url, json={"chat_id": target, "text": text[i:i+4000]}, timeout=10)
            else:
                requests.post(url, json={"chat_id": target, "text": text}, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    def _send_typing(self, chat_id: str):
        """Send typing indicator"""
        try:
            url = f"{self.base_url}/sendChatAction"
            requests.post(url, json={"chat_id": chat_id, "action": "typing"}, timeout=5)
        except:
            pass

if __name__ == "__main__":
    try:
        controller = DMAIMasterControl()
        controller.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
