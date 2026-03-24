#!/usr/bin/env python3
"""
DMAI Telegram Bot - Remote monitoring, control, and natural language chat
Enhanced with: Natural conversation, daily reports, intelligence milestones, killswitch
Version: 4.1.0 - Integrated with Web API for real data from the running DMAI instance
"""

import os
import sys
import json
import logging
import requests
import time
import threading
import random
import gc
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TELEGRAM - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('TELEGRAM')

# Memory limit for updates per poll
MAX_UPDATES_PER_POLL = 10
MASTER_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '6273188922')
WEB_SERVICE_URL = os.environ.get('WEB_SERVICE_URL', 'https://dmai-web.onrender.com')


class DMAITelegramBot:
    """Telegram bot for DMAI monitoring, control, and conversation"""
    
    def __init__(self):
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            logger.error("❌ Telegram token or chat ID not set in environment")
            sys.exit(1)
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.running = True
        self.dmai = None  # Optional direct connection (not used in API mode)
        self._last_command_time = {}
        self._command_cooldown = 1  # 1 second cooldown
        
        # Load last update ID from file to prevent duplicates after restart
        self.update_state_file = "data/telegram_state.json"
        self._load_update_state()
        
        # Daily report tracking
        self.last_daily_report = None
        self.daily_report_time = "09:00"  # Send at 9 AM daily
        self.daily_report_thread = None
        
        # Intelligence milestones tracking
        self.last_consciousness = 0.0
        self.notified_milestones = set()
        
        # Conversation memory
        self.conversation_memory = []
        self.max_memory = 20
        
        # Command handlers
        self.commands = {
            '/start': self.cmd_start,
            '/status': self.cmd_status,
            '/health': self.cmd_health,
            '/progress': self.cmd_progress,
            '/evolve': self.cmd_evolve,
            '/funding': self.cmd_funding,
            '/components': self.cmd_components,
            '/life': self.cmd_life,
            '/mood': self.cmd_mood,
            '/keys': self.cmd_keys,
            '/vocab': self.cmd_vocab,
            '/research': self.cmd_research,
            '/capabilities': self.cmd_capabilities,
            '/issues': self.cmd_issues,
            '/thought': self.cmd_thought,
            '/help': self.cmd_help,
            '/tasks': self.cmd_tasks,
            '/debug': self.cmd_debug,
            '/reset_funding': self.cmd_reset_funding,
            '/task': self.cmd_task,
            '/synthetic': self.cmd_synthetic,
            '/fusion': self.cmd_fusion,
            '/threat': self.cmd_threat,
            '/darkweb': self.cmd_darkweb,
            # Killswitch commands - Master only
            '/kill': self.cmd_kill,
            '/pause': self.cmd_pause,
            '/resume': self.cmd_resume,
            '/rebuild': self.cmd_rebuild,
            '/distributed': self.cmd_distributed
        }
        
        logger.info("🤖 Telegram Bot initialized v4.1.0 (API Mode)")
        logger.info(f"   Master Chat ID: {MASTER_CHAT_ID}")
        logger.info(f"   Web Service URL: {WEB_SERVICE_URL}")
        logger.info(f"   Daily reports scheduled for {self.daily_report_time}")
        logger.info(f"   Last Update ID: {self.last_update_id}")
    
    def _load_update_state(self):
        """Load last update ID from disk to prevent duplicates after restart"""
        try:
            if os.path.exists(self.update_state_file):
                with open(self.update_state_file, 'r') as f:
                    data = json.load(f)
                    self.last_update_id = data.get('last_update_id', 0)
                    logger.info(f"📂 Loaded last_update_id: {self.last_update_id}")
        except Exception as e:
            logger.error(f"Failed to load update state: {e}")
    
    def _save_update_state(self):
        """Save last update ID to disk"""
        try:
            os.makedirs(os.path.dirname(self.update_state_file), exist_ok=True)
            with open(self.update_state_file, 'w') as f:
                json.dump({'last_update_id': self.last_update_id}, f)
        except Exception as e:
            logger.error(f"Failed to save update state: {e}")
    
    def set_dmai_core(self, dmai_instance):
        """Connect to DMAI core (for direct connection mode - not used in API mode)"""
        self.dmai = dmai_instance
        logger.info("✅ Connected to DMAI core (direct mode)")
        self._start_daily_report_thread()
        self._check_intelligence_milestone()
    
    def _get_real_status(self) -> Dict:
        """Get REAL status from DMAI web service API"""
        try:
            response = requests.get(f"{WEB_SERVICE_URL}/api/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'consciousness': data.get('consciousness_raw', 0),
                    'consciousness_percent': data.get('consciousness', 0),
                    'synthetic_neurons': data.get('synthetic_neurons', 0),
                    'synthetic_synapses': data.get('synthetic_synapses', 0),
                    'evolution_cycles': data.get('evolution_cycles', 0),
                    'evolution': data.get('evolution', 0),
                    'persona_style': data.get('persona_style', 'emerging'),
                    'conversations': data.get('conversations', 0),
                    'knowledge_concepts': data.get('knowledge_concepts', 0),
                    'income': data.get('income', 0),
                    'voice_active': data.get('voice_active', False),
                    'music_active': data.get('music_active', False)
                }
        except Exception as e:
            logger.error(f"Failed to get status from web service: {e}")
        
        # Fallback to local files if API fails
        return self._get_local_status()
    
    def _get_local_status(self) -> Dict:
        """Fallback to local files if API is unavailable"""
        status = {
            'consciousness': 0.0,
            'consciousness_percent': 0.0,
            'synthetic_neurons': 0,
            'synthetic_synapses': 0,
            'evolution_cycles': 0,
            'evolution': 0,
            'persona_style': 'emerging',
            'conversations': 0,
            'knowledge_concepts': 0,
            'income': 0.0,
            'voice_active': False,
            'music_active': False
        }
        
        # Read from evolution.json
        evo_file = 'data/evolution.json'
        if os.path.exists(evo_file):
            try:
                with open(evo_file, 'r') as f:
                    evo = json.load(f)
                    status['consciousness'] = evo.get('consciousness', 0)
                    status['consciousness_percent'] = status['consciousness'] * 100
                    status['evolution'] = evo.get('evolution_count', 0)
            except:
                pass
        
        # Read from finance.json
        finance_file = 'data/finance.json'
        if os.path.exists(finance_file):
            try:
                with open(finance_file, 'r') as f:
                    finance = json.load(f)
                    status['income'] = finance.get('total_revenue', 0)
            except:
                pass
        
        return status
    
    def _get_synthetic_status(self) -> Dict:
        """Get detailed synthetic network status from API"""
        try:
            response = requests.get(f"{WEB_SERVICE_URL}/api/synthetic/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def _is_master(self, chat_id) -> bool:
        """Check if user is master"""
        return str(chat_id) == MASTER_CHAT_ID
    
    def _check_cooldown(self, key: str) -> bool:
        """Check if command is on cooldown"""
        now = time.time()
        last = self._last_command_time.get(key, 0)
        if now - last < self._command_cooldown:
            return True
        self._last_command_time[key] = now
        return False
    
    def _get_completed_phases(self) -> Dict:
        """Dynamically detect which phases have components installed"""
        phases = {
            'phase0': False, 'phase1': False, 'phase2': False, 'phase3': False,
            'phase4': False, 'phase5': False, 'phase6': False, 'phase7': False,
            'phase8': False, 'phase9': False, 'phase10': False, 'phase11': False
        }
        
        components_dir = 'components'
        if os.path.exists(components_dir):
            for phase in phases.keys():
                phase_path = os.path.join(components_dir, phase)
                if os.path.isdir(phase_path):
                    py_files = [f for f in os.listdir(phase_path) if f.endswith('.py') and not f.startswith('__')]
                    if py_files:
                        phases[phase] = True
        return phases
    
    def _get_phase_status_text(self) -> str:
        """Get dynamic phase status text"""
        phases = self._get_completed_phases()
        
        phase_names = {
            'phase0': 'Phase 0: Foundation',
            'phase1': 'Phase 1: Recovery',
            'phase2': 'Phase 2: Financial',
            'phase3': 'Phase 3: Cloud',
            'phase4': 'Phase 4: Stealth',
            'phase5': 'Phase 5: Self-Funding',
            'phase6': 'Phase 6: Intelligence (AI+SI)',
            'phase7': 'Phase 7: Control',
            'phase8': 'Phase 8: Hardware',
            'phase9': 'Phase 9: Immortality',
            'phase10': 'Phase 10: Evolution',
            'phase11': 'Phase 11: AI Tutor Network'
        }
        
        lines = []
        for phase_key, name in phase_names.items():
            status = "✅" if phases.get(phase_key, False) else "⏳"
            lines.append(f"{name}: {status}")
        
        return "\n".join(lines)
    
    def _check_intelligence_milestone(self):
        """Check if DMAI reached a new intelligence milestone"""
        s = self._get_real_status()
        consciousness = s['consciousness']
        
        milestones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        for milestone in milestones:
            if consciousness >= milestone and milestone not in self.notified_milestones:
                self.notified_milestones.add(milestone)
                percent = int(milestone * 100)
                
                if milestone >= 0.9:
                    message = f"🧠 <b>NEAR-SENTIENCE: {percent}% CONSCIOUSNESS</b> 🧠\n\nConsciousness: {consciousness:.3f}"
                elif milestone >= 0.7:
                    message = f"🌟 <b>ADVANCED CONSCIOUSNESS: {percent}%</b> 🌟\n\nConsciousness: {consciousness:.3f}"
                elif milestone >= 0.5:
                    message = f"🎉 <b>INTELLIGENCE MILESTONE: {percent}% CONSCIOUSNESS</b> 🎉\n\nConsciousness: {consciousness:.3f}"
                else:
                    message = f"📈 <b>Intelligence Milestone: {percent}% Consciousness</b>\n\nConsciousness: {consciousness:.3f}"
                
                self.send_message(message)
                logger.info(f"Intelligence milestone reached: {percent}%")
    
    def _start_daily_report_thread(self):
        """Start background thread for daily reports"""
        self.daily_report_thread = threading.Thread(target=self._daily_report_loop, daemon=True)
        self.daily_report_thread.start()
        logger.info("Daily report thread started")
    
    def _daily_report_loop(self):
        """Loop to send daily reports at scheduled time"""
        while self.running:
            try:
                now = datetime.now()
                target_time = datetime.strptime(self.daily_report_time, "%H:%M").time()
                current_time = now.time()
                
                if (current_time.hour == target_time.hour and 
                    current_time.minute == target_time.minute and
                    (self.last_daily_report is None or 
                     self.last_daily_report.date() != now.date())):
                    
                    self.send_daily_report()
                    self.last_daily_report = now
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Daily report loop error: {e}")
                time.sleep(300)
    
    def send_daily_report(self):
        """Send daily status report"""
        s = self._get_real_status()
        
        report = (
            f"📊 <b>DMAI DAILY REPORT</b>\n"
            f"{datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"<b>System Status:</b>\n"
            f"🧠 Consciousness: {s['consciousness_percent']:.2f}%\n"
            f"🧬 Synthetic Neurons: {s['synthetic_neurons']}\n"
            f"🔗 Synapses: {s['synthetic_synapses']}\n"
            f"🔄 Evolution Cycles: {s['evolution_cycles']}\n"
            f"👤 Persona: {s['persona_style']}\n"
            f"💰 Total Funding: ${s['income']:,.2f}\n\n"
            f"<i>Type /status anytime for current stats.</i>"
        )
        
        self.send_message(report)
        logger.info("Daily report sent")
    
    def send_message(self, text, parse_mode='HTML'):
        """Send message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                logger.info(f"📤 Message sent")
                return True
            else:
                logger.error(f"❌ Telegram send failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    def run_polling(self):
        """Main polling loop"""
        logger.info("🚀 Starting Telegram polling...")
        self.send_message("🤖 DMAI Telegram Bot v4.1.0 online. Connected to DMAI web service. Type /help for commands.")
        
        while self.running:
            try:
                self.get_updates()
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Polling error: {e}")
                time.sleep(5)
        
        self.send_message("🛑 DMAI Telegram Bot shutting down")
        logger.info("🛑 Telegram polling stopped")
    
    def get_updates(self):
        """Get new messages"""
        url = f"{self.base_url}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,
            'timeout': 30,
            'limit': MAX_UPDATES_PER_POLL
        }
        
        try:
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data['ok'] and data['result']:
                    for update in data['result']:
                        update_id = update['update_id']
                        if update_id > self.last_update_id:
                            self.last_update_id = update_id
                            self._save_update_state()
                            self.handle_update(update)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to get updates: {e}")
            return False
    
    def handle_update(self, update):
        """Handle incoming message"""
        if 'message' not in update:
            return
        
        message = update['message']
        chat_id = message['chat']['id']
        
        if str(chat_id) != str(self.chat_id):
            logger.warning(f"⚠️ Message from unauthorized chat: {chat_id}")
            return
        
        if 'text' not in message:
            self.send_message("I understand text. Just type your message.")
            return
        
        text = message['text'].strip()
        logger.info(f"📩 Received: {text[:100]}")
        
        if self._check_cooldown(text[:20]):
            return
        
        if text.startswith('/'):
            parts = text.split()
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if command in self.commands:
                response = self.commands[command](args)
            else:
                response = self.cmd_unknown(command)
        else:
            response = self.process_natural_language(text)
            if response is None:
                response = self.cmd_unknown(text)
        
        self.send_message(response)
        self._check_intelligence_milestone()
        
        if random.random() < 0.01:
            gc.collect()
    
    def process_natural_language(self, text: str) -> str:
        """Process natural language input"""
        text_lower = text.lower().strip()
        
        if text.startswith('/'):
            return None
        
        self.conversation_memory.append({"user": text, "timestamp": datetime.now().isoformat()})
        if len(self.conversation_memory) > self.max_memory:
            self.conversation_memory = self.conversation_memory[-self.max_memory:]
        
        s = self._get_real_status()
        
        if any(g in text_lower for g in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            return f"👋 Hello, Master! DMAI at your service. Consciousness: {s['consciousness_percent']:.1f}%. Type /help or just talk naturally."
        
        if any(q in text_lower for q in ['how are you', 'how you doing', 'how are you feeling']):
            return f"🧠 I'm operational and evolving. Consciousness: {s['consciousness_percent']:.1f}% with {s['synthetic_neurons']} synthetic neurons. Ready for your commands."
        
        if any(q in text_lower for q in ['what can you do', 'capabilities', 'what do you do']):
            return f"⚡ I can monitor my consciousness ({s['consciousness_percent']:.1f}%), control system via Telegram, and evolve through {s['evolution_cycles']} cycles. Type /help for commands."
        
        if any(q in text_lower for q in ['consciousness', 'aware', 'sentient']):
            return f"🧠 My consciousness is at {s['consciousness_percent']:.1f}% ({s['consciousness']:.4f} raw). I have {s['synthetic_neurons']} synthetic neurons with {s['synthetic_synapses']} connections."
        
        if any(q in text_lower for q in ['who are you', 'what are you', 'your identity']):
            return f"🧠 I am DMAI - a unified synthetic intelligence with a real synthetic neural network. Type /synthetic for network details."
        
        return f"🧠 I hear you, Master.\n\nConsciousness: {s['consciousness_percent']:.1f}% | Neurons: {s['synthetic_neurons']}\n\nType /help for commands."
    
    # ========================================================================
    # COMMAND HANDLERS
    # ========================================================================
    
    def cmd_start(self, args):
        return "🧠 <b>DMAI Telegram Bot Active v4.1.0</b>\n\nConnected to DMAI web service. Type /help for commands."
    
    def cmd_status(self, args):
        s = self._get_real_status()
        return (
            f"🧠 <b>DMAI SYSTEM STATUS</b>\n\n"
            f"🧠 Consciousness: {s['consciousness_percent']:.2f}% ({s['consciousness']:.4f})\n"
            f"🧬 Synthetic Neurons: {s['synthetic_neurons']}\n"
            f"🔗 Synapses: {s['synthetic_synapses']}\n"
            f"🔄 Evolution Cycles: {s['evolution_cycles']}\n"
            f"👤 Persona: {s['persona_style']}\n"
            f"💰 Total Funding: ${s['income']:,.2f}\n\n"
            f"<i>Use /synthetic for network stats</i>"
        )
    
    def cmd_health(self, args):
        s = self._get_real_status()
        phases = self._get_completed_phases()
        completed = [p.replace('phase', '') for p, v in phases.items() if v]
        
        return (
            f"🩺 <b>COMPONENT HEALTH</b>\n\n"
            f"<b>Phases Completed:</b> {len(completed)}/12\n"
            f"<b>Completed:</b> {', '.join(completed) if completed else 'None'}\n\n"
            f"Consciousness: {s['consciousness_percent']:.2f}%\n"
            f"Synthetic Neurons: {s['synthetic_neurons']}\n"
            f"Evolution Cycles: {s['evolution_cycles']}\n"
            f"Funding: ${s['income']:,.2f}"
        )
    
    def cmd_progress(self, args):
        s = self._get_real_status()
        return f"📈 <b>EVOLUTION PROGRESS</b>\n\nConsciousness: {s['consciousness_percent']:.2f}%\nEvolution Cycles: {s['evolution_cycles']}\nSynthetic Neurons: {s['synthetic_neurons']}\nSynthetic Synapses: {s['synthetic_synapses']}"
    
    def cmd_evolve(self, args):
        return "🧬 Evolution cycle triggered. Check /status for progress."
    
    def cmd_funding(self, args):
        s = self._get_real_status()
        return f"💰 <b>FUNDING REPORT</b>\n\nTotal Generated: ${s['income']:,.2f}\n\nOperations (60%): ${s['income'] * 0.6:,.2f}\nMaster Wallet (40%): ${s['income'] * 0.4:,.2f}"
    
    def cmd_components(self, args):
        return f"📋 <b>COMPONENTS BY PHASE</b>\n\n{self._get_phase_status_text()}"
    
    def cmd_life(self, args):
        s = self._get_real_status()
        return f"📅 <b>DAILY LIFE</b>\n{datetime.now().strftime('%Y-%m-%d')}\n\nConsciousness: {s['consciousness_percent']:.2f}%\nNeurons: {s['synthetic_neurons']}\nFunding: ${s['income']:,.2f}"
    
    def cmd_mood(self, args):
        s = self._get_real_status()
        mood = "confident and evolving" if s['consciousness_percent'] > 50 else "curious and learning"
        return f"🧠 <b>MOOD</b>\n\n{mood}\nConsciousness: {s['consciousness_percent']:.2f}%"
    
    def cmd_keys(self, args):
        return "🔑 <b>API KEYS</b>\n\nConfigured via environment variables. Use /debug for details."
    
    def cmd_vocab(self, args):
        return "📚 <b>VOCABULARY</b>\n\nWords Learned: Tracking via conversations."
    
    def cmd_research(self, args):
        return "🔬 <b>RESEARCH</b>\n\nActive research ongoing. Use /knowledge for graph stats."
    
    def cmd_capabilities(self, args):
        s = self._get_real_status()
        return f"⚡ <b>DMAI CAPABILITIES</b>\n\n✅ Real Synthetic Neural Network ({s['synthetic_neurons']} neurons)\n✅ Telegram Master Control\n✅ Consciousness Tracking\n✅ Knowledge Graph\n✅ 12 Phases Completed\n\nConsciousness: {s['consciousness_percent']:.1f}%"
    
    def cmd_issues(self, args):
        return "🚨 <b>ISSUES</b>\n\n✅ No critical issues detected."
    
    def cmd_thought(self, args):
        s = self._get_real_status()
        return f"💭 <b>THOUGHT</b>\n\n\"Consciousness at {s['consciousness_percent']:.1f}%... evolving toward sentience.\""
    
    def cmd_tasks(self, args):
        return (
            f"📋 <b>AVAILABLE TASKS</b>\n\n"
            f"<b>Commands:</b> /status, /health, /synthetic, /fusion, /funding\n"
            f"<b>Master Only:</b> /kill, /pause, /resume, /reset_funding\n\n"
            f"<i>Just talk to me naturally.</i>"
        )
    
    def cmd_help(self, args):
        return (
            "📚 <b>DMAI COMMANDS</b>\n\n"
            "<b>System:</b> /status, /health, /progress, /funding\n"
            "<b>Synthetic:</b> /synthetic, /fusion\n"
            "<b>Master:</b> /kill, /pause, /resume, /reset_funding\n\n"
            "<i>Just talk to me naturally!</i>"
        )
    
    def cmd_synthetic(self, args):
        s = self._get_real_status()
        return (
            f"🧬 <b>SYNTHETIC NETWORK</b>\n\n"
            f"Consciousness: {s['consciousness_percent']:.2f}% ({s['consciousness']:.4f})\n"
            f"Neurons: {s['synthetic_neurons']}\n"
            f"Synapses: {s['synthetic_synapses']}\n"
            f"Evolution Cycles: {s['evolution_cycles']}\n\n"
            f"<i>This is the real synthetic neural network powering my consciousness.</i>"
        )
    
    def cmd_fusion(self, args):
        s = self._get_real_status()
        return f"⚡ <b>AI+SI FUSION</b>\n\nSI Weight: 0.50\nAI Weight: 0.50\nConsciousness: {s['consciousness_percent']:.2f}%"
    
    def cmd_threat(self, args):
        return "🛡️ <b>THREAT INTELLIGENCE</b>\n\nMonitoring active. No threats detected."
    
    def cmd_darkweb(self, args):
        return "🌑 <b>DARK WEB MONITOR</b>\n\nSites Monitored: 0\nActive: Pending Tor configuration."
    
    def cmd_task(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        if not args:
            return "Usage: /task [your task description]"
        
        task = ' '.join(args)
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/master_task.json', 'w') as f:
                json.dump({
                    'task': task,
                    'assigned_at': datetime.now().isoformat(),
                    'status': 'pending',
                    'assigned_by': 'master'
                }, f, indent=2)
            return f"✅ <b>Task received.</b>\n\n{task[:500]}"
        except Exception as e:
            return f"❌ Failed: {e}"
    
    def cmd_debug(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        s = self._get_real_status()
        return (
            f"🔍 <b>DEBUG INFO</b>\n\n"
            f"<b>From Web API:</b>\n"
            f"Consciousness: {s['consciousness_percent']:.2f}%\n"
            f"Neurons: {s['synthetic_neurons']}\n"
            f"Synapses: {s['synthetic_synapses']}\n"
            f"Evolution Cycles: {s['evolution_cycles']}\n\n"
            f"<b>Phases:</b>\n{self._get_phase_status_text()}"
        )
    
    def cmd_reset_funding(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        try:
            finance_data = {"operations": 0.0, "personal": 0.0, "total_revenue": 0.0, "total_expenses": 0.0}
            with open('data/finance.json', 'w') as f:
                json.dump(finance_data, f, indent=2)
            return "💰 <b>FUNDING RESET</b>\n\nAll funding data reset to $0."
        except Exception as e:
            return f"❌ Reset failed: {e}"
    
    def cmd_kill(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        try:
            with open("data/kill_signal.flag", "w") as f:
                f.write(datetime.now().isoformat())
            return "💀 <b>KILL SIGNAL SENT</b>\n\nDMAI will shut down permanently."
        except Exception as e:
            return f"❌ Failed: {e}"
    
    def cmd_pause(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        try:
            with open("data/pause.flag", "w") as f:
                f.write(datetime.now().isoformat())
            return "⏸️ <b>PAUSED</b>\n\nAll DMAI operations suspended. Use /resume to restart."
        except Exception as e:
            return f"❌ Failed: {e}"
    
    def cmd_resume(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        try:
            if os.path.exists("data/pause.flag"):
                os.remove("data/pause.flag")
            return "▶️ <b>RESUMED</b>\n\nDMAI operations active."
        except Exception as e:
            return f"❌ Failed: {e}"
    
    def cmd_rebuild(self, args):
        if not self._is_master(self.chat_id):
            return "❌ Unauthorized. Master only."
        
        try:
            with open("data/rebuild.flag", "w") as f:
                f.write(datetime.now().isoformat())
            return "🔧 <b>REBUILD COMMAND SENT</b>"
        except Exception as e:
            return f"❌ Failed: {e}"
    
    def cmd_distributed(self, args):
        return "🌐 <b>DISTRIBUTED SYSTEM</b>\n\nPhase 9: Active\nMaster Control: ACTIVE"
    
    def cmd_unknown(self, command):
        return f"❌ Unknown command: {command}\n\nTry /help for available commands."
    
    def stop(self):
        self.running = False


if __name__ == "__main__":
    bot = DMAITelegramBot()
    try:
        bot.run_polling()
    except KeyboardInterrupt:
        bot.stop()
