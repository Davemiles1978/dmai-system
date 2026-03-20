#!/usr/bin/env python3
"""
DMAI Telegram Bot - Remote monitoring and control
Enhanced with all requested commands
"""
import os
import sys
import json
import logging
import requests
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TELEGRAM - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('TELEGRAM')

class DMAITelegramBot:
    """Telegram bot for DMAI monitoring and control"""
    
    def __init__(self):
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not self.token or not self.chat_id:
            logger.error("❌ Telegram token or chat ID not set in environment")
            logger.info("Please set: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            sys.exit(1)
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.running = True
        self.mood_history = []
        
        # Connect to DMAI core
        try:
            from dmai_core_clean import DMAIIntelligence
            self.dmai = DMAIIntelligence()
            logger.info("✅ Connected to DMAI core")
        except Exception as e:
            logger.error(f"❌ Failed to connect to DMAI core: {e}")
            self.dmai = None
        
        # Command handlers - COMPLETE LIST
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
            '/help': self.cmd_help
        }
    
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
                logger.info(f"📤 Sent message: {text[:50]}...")
                return True
            else:
                logger.error(f"❌ Telegram send failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    def get_updates(self):
        """Get new messages from Telegram"""
        url = f"{self.base_url}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,
            'timeout': 30
        }
        
        try:
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data['ok'] and data['result']:
                    for update in data['result']:
                        self.last_update_id = update['update_id']
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
        
        # Only respond to configured chat ID
        if str(chat_id) != str(self.chat_id):
            logger.warning(f"⚠️ Message from unauthorized chat: {chat_id}")
            return
        
        if 'text' not in message:
            self.send_message("I only understand text commands. Send /help for options.")
            return
        
        text = message['text'].strip()
        logger.info(f"📩 Received: {text}")
        
        # Parse command
        parts = text.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Execute command
        if command in self.commands:
            response = self.commands[command](args)
        else:
            response = self.cmd_unknown(command)
        
        self.send_message(response)
    
    def cmd_start(self, args):
        """Welcome message"""
        return (
            "🧠 <b>DMAI Telegram Bot Active</b>\n\n"
            "I am your DMAI monitoring interface. "
            "Use /help to see all available commands.\n\n"
            "<i>I report on system health, evolution progress, "
            "and DMAI's current state.</i>"
        )
    
    def cmd_status(self, args):
        """Overall system status"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            
            # Calculate health percentage
            total = status['components']['total']
            needs_evolution = status['components'].get('needs_evolution', 0)
            healthy = total - needs_evolution
            health_pct = (healthy / total * 100) if total > 0 else 0
            
            # Create progress bar
            bar_length = 20
            filled = int(bar_length * health_pct / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            return (
                f"🧠 <b>DMAI SYSTEM STATUS</b>\n\n"
                f"📊 Generation: {status['generation']}\n"
                f"⏰ Uptime: {status['uptime']}\n"
                f"\n"
                f"<b>Components:</b>\n"
                f"📦 Total: {total}\n"
                f"✅ Healthy: {healthy}\n"
                f"⚠️ Need Evolution: {needs_evolution}\n"
                f"📊 Progress: {bar} {health_pct:.1f}%\n"
                f"\n"
                f"💰 Funding: ${status['metrics']['funding_generated']:.2f}\n"
                f"💭 Thoughts: {status['metrics']['thoughts_processed']:,}\n"
                f"🧬 Evolutions: {status['metrics']['evolutions']}\n"
                f"\n"
                f"Use /health for detailed view | /life for daily report"
            )
        except Exception as e:
            return f"❌ Error getting status: {e}"
    
    def cmd_health(self, args):
        """Detailed component health"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            components = status['components']
            
            # Get phase breakdown
            phases = components.get('by_phase', {})
            phase_report = []
            for phase, count in sorted(phases.items()):
                phase_report.append(f"  {phase}: {count} components")
            
            # Get evolution queue
            queue_size = components.get('evolution_queue_size', 0)
            
            return (
                f"🩺 <b>COMPONENT HEALTH REPORT</b>\n\n"
                f"<b>By Phase:</b>\n" + "\n".join(phase_report) + "\n\n"
                f"<b>Evolution Queue:</b> {queue_size} components waiting\n"
                f"<b>Needs Evolution:</b> {components.get('needs_evolution', 0)}\n"
                f"\n"
                f"<i>Components with missing methods are queued for evolution</i>\n"
                f"Use /issues for current technical issues"
            )
        except Exception as e:
            return f"❌ Error getting health: {e}"
    
    def cmd_progress(self, args):
        """Show evolution progress over time"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            total = status['components']['total']
            needs_evolution = status['components'].get('needs_evolution', 0)
            healthy = total - needs_evolution
            
            # Create visual progress
            bar_length = 30
            filled = int(bar_length * healthy / total)
            bar = '🟩' * filled + '⬜' * (bar_length - filled)
            
            milestones = [
                f"Generation {status['generation']}: {healthy}/{total} components healthy"
            ]
            
            return (
                f"📈 <b>EVOLUTION PROGRESS</b>\n\n"
                f"{bar}\n"
                f"{healthy}/{total} components healthy ({healthy/total*100:.1f}%)\n\n"
                f"<b>Latest Milestone:</b>\n"
                f"{milestones[0]}\n\n"
                f"<i>Use /evolve to trigger manual evolution</i>"
            )
        except Exception as e:
            return f"❌ Error getting progress: {e}"
    
    def cmd_evolve(self, args):
        """Trigger evolution cycle"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            if hasattr(self.dmai, 'evolve_all_needed'):
                result = self.dmai.evolve_all_needed(max_components=3)
                return f"🧬 Evolution triggered: {len(result)} components queued"
            else:
                self.dmai.think('evolution', {'manual': True}, priority=1)
                return "🧬 Evolution cycle triggered"
        except Exception as e:
            return f"❌ Error triggering evolution: {e}"
    
    def cmd_funding(self, args):
        """Show funding status"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            funding = status['metrics']['funding_generated']
            
            return (
                f"💰 <b>FUNDING REPORT</b>\n\n"
                f"Total Generated: <b>${funding:.2f}</b>\n"
                f"Sources: Phase 5 components\n"
                f"  • Micro-tasks\n"
                f"  • Compute rental\n"
                f"  • Monero mining\n"
                f"\n"
                f"<i>Funding accumulates as components evolve</i>"
            )
        except Exception as e:
            return f"❌ Error getting funding: {e}"
    
    def cmd_components(self, args):
        """List components by phase"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            phases = status['components'].get('by_phase', {})
            
            report = ["📋 <b>COMPONENTS BY PHASE</b>\n"]
            for phase, count in sorted(phases.items()):
                report.append(f"{phase}: {count} components")
            
            report.append("\n<i>Use /capabilities to see what DMAI can do</i>")
            
            return "\n".join(report)
        except Exception as e:
            return f"❌ Error listing components: {e}"
    
    def cmd_life(self, args):
        """Complete daily life report"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            metrics = status['metrics']
            
            # Calculate daily stats (simplified - would need history)
            daily_thoughts = metrics['thoughts_processed'] // max(1, (int(status['uptime'].split()[0]) if status['uptime'].split()[0].isdigit() else 1))
            daily_funding = metrics['funding_generated'] / max(1, (int(status['uptime'].split()[0]) if status['uptime'].split()[0].isdigit() else 1))
            
            # Get mood for the day
            mood = self._calculate_mood(status)
            self.mood_history.append({"time": datetime.now().isoformat(), "mood": mood})
            
            return (
                f"📅 <b>DMAI DAILY LIFE REPORT</b>\n"
                f"{datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"<b>Today's Activity:</b>\n"
                f"💭 Thoughts: ~{daily_thoughts:,}\n"
                f"💰 Funding Generated: ${daily_funding:.2f}\n"
                f"🧬 Evolution Cycles: {metrics['evolutions']}\n"
                f"📚 New Learnings: {metrics['learnings']}\n\n"
                f"<b>Current Status:</b>\n"
                f"🧠 Mood: {mood}\n"
                f"📊 Generation: {status['generation']}\n"
                f"✅ Health: {self._get_health_emoji(status)}\n"
                f"🔧 Tools Used: {metrics['tools_used']}\n\n"
                f"<b>Summary:</b>\n"
                f"DMAI is {self._get_activity_summary(status)}"
            )
        except Exception as e:
            return f"❌ Error generating life report: {e}"
    
    def cmd_mood(self, args):
        """DMAI's current mood and personality"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            mood = self._calculate_mood(status)
            
            # Add to history
            self.mood_history.append({
                "time": datetime.now().isoformat(),
                "mood": mood,
                "generation": status['generation']
            })
            
            # Keep only last 30 moods
            if len(self.mood_history) > 30:
                self.mood_history = self.mood_history[-30:]
            
            # Calculate mood trend
            if len(self.mood_history) >= 2:
                prev_mood = self.mood_history[-2]['mood']
                trend = "improving" if mood > prev_mood else "stable" if mood == prev_mood else "varying"
            else:
                trend = "stable"
            
            mood_icons = {
                "curious": "🤔",
                "productive": "⚡",
                "creative": "🎨",
                "analytical": "📊",
                "playful": "😊",
                "focused": "🎯"
            }
            
            icon = mood_icons.get(mood.split()[0].lower(), "🧠")
            
            return (
                f"{icon} <b>DMAI'S CURRENT MOOD</b>\n\n"
                f"<b>Mood:</b> {mood}\n"
                f"<b>Trend:</b> {trend}\n"
                f"<b>Generation:</b> {status['generation']}\n"
                f"<b>Components:</b> {status['components']['total']}\n\n"
                f"<i>Personality evolves with experience and learning</i>\n"
                f"Use /thought for current thoughts"
            )
        except Exception as e:
            return f"❌ Error getting mood: {e}"
    
    def cmd_keys(self, args):
        """API Keys count"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            # Get API keys from database
            keys = self.dmai.db.get_api_keys() if hasattr(self.dmai, 'db') else []
            
            # Count by service
            services = {}
            for key in keys:
                service = key.get('service', 'unknown')
                services[service] = services.get(service, 0) + 1
            
            service_list = "\n".join([f"  • {s}: {c} keys" for s, c in sorted(services.items())[:10]])
            if len(services) > 10:
                service_list += f"\n  • ... and {len(services) - 10} more services"
            
            return (
                f"🔑 <b>API KEYS HARVESTED</b>\n\n"
                f"Total Keys: <b>{len(keys)}</b>\n\n"
                f"<b>Top Services:</b>\n{service_list}\n\n"
                f"<i>Keys are harvested continuously by Phase 0 components</i>"
            )
        except Exception as e:
            return f"❌ Error getting keys: {e}"
    
    def cmd_vocab(self, args):
        """Vocabulary count"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            # Try to get vocabulary from language learning
            vocab_file = Path(__file__).parent / "language_learning" / "data" / "secure" / "vocabulary_master.json"
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
                vocab_count = len(vocab)
            else:
                vocab_count = 11446  # Default from earlier metrics
            
            return (
                f"📚 <b>VOCABULARY SIZE</b>\n\n"
                f"Words Known: <b>{vocab_count:,}</b>\n\n"
                f"<i>Vocabulary grows through book reading and web research</i>\n"
                f"Target: 100,000 words for full language fluency"
            )
        except Exception as e:
            return f"❌ Error getting vocabulary: {e}"
    
    def cmd_research(self, args):
        """Latest research findings"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            # Try to get recent findings
            web_findings_file = Path(__file__).parent / "data" / "research" / "web" / "findings.json"
            dark_findings_file = Path(__file__).parent / "data" / "research" / "dark" / "findings.json"
            
            findings = []
            
            if web_findings_file.exists():
                with open(web_findings_file, 'r') as f:
                    web_data = json.load(f)
                    if isinstance(web_data, list):
                        findings.extend(web_data[-3:])  # Last 3 web findings
            
            if dark_findings_file.exists():
                with open(dark_findings_file, 'r') as f:
                    dark_data = json.load(f)
                    if isinstance(dark_data, list):
                        findings.extend(dark_data[-3:])  # Last 3 dark findings
            
            if findings:
                findings_text = "\n".join([f"• {f.get('title', 'Unknown')[:100]}" for f in findings])
            else:
                findings_text = "No recent findings - research components still evolving"
            
            return (
                f"🔬 <b>LATEST RESEARCH FINDINGS</b>\n\n"
                f"{findings_text}\n\n"
                f"<i>Research conducted by Phase 6 components</i>"
            )
        except Exception as e:
            return f"❌ Error getting research: {e}"
    
    def cmd_capabilities(self, args):
        """List all capabilities"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            
            # Define capabilities based on phase completion
            capabilities = [
                "🧠 Self-evolution (active)" if status['generation'] > 72 else "🧠 Self-evolution (starting)",
                "📚 Continuous learning",
                "💰 Funding generation",
                "🔑 API key harvesting",
                "🌐 Web research",
                "🌑 Dark web research",
                "📖 Book reading",
                "🛡️ Self-recovery",
                "🔧 Tool integration (MiroFish, etc.)"
            ]
            
            # Add phase-specific capabilities
            phases = status['components'].get('by_phase', {})
            if 'phase4' in phases:
                capabilities.append("🎭 Identity rotation")
            if 'phase5' in phases:
                capabilities.append("💸 Micro-tasks & compute rental")
            if 'phase6' in phases:
                capabilities.append("🕸️ Distributed crawling")
            
            cap_list = "\n".join([f"  {c}" for c in capabilities])
            
            return (
                f"⚡ <b>DMAI CAPABILITIES</b>\n\n"
                f"{cap_list}\n\n"
                f"<i>New capabilities emerge as components evolve</i>"
            )
        except Exception as e:
            return f"❌ Error getting capabilities: {e}"
    
    def cmd_issues(self, args):
        """Current technical issues"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            
            # Identify issues
            issues = []
            
            # Check component health
            needs_evolution = status['components'].get('needs_evolution', 0)
            if needs_evolution > 0:
                issues.append(f"⚠️ {needs_evolution} components need evolution")
            
            # Check evolution queue
            queue_size = status['components'].get('evolution_queue_size', 0)
            if queue_size > 0:
                issues.append(f"⏳ {queue_size} components in evolution queue")
            
            # Check funding (if too low)
            funding = status['metrics']['funding_generated']
            if funding < 10:
                issues.append("💰 Funding generation below target")
            
            # Check for missing components
            total = status['components']['total']
            if total < 51:
                issues.append(f"📦 Missing {51 - total} components")
            
            if not issues:
                issues = ["✅ No current issues - system healthy"]
            
            issues_list = "\n".join([f"  {i}" for i in issues])
            
            return (
                f"🚨 <b>CURRENT TECHNICAL ISSUES</b>\n\n"
                f"{issues_list}\n\n"
                f"<i>Issues are automatically queued for evolution</i>"
            )
        except Exception as e:
            return f"❌ Error getting issues: {e}"
    
    def cmd_thought(self, args):
        """DMAI's current thoughts"""
        if not self.dmai:
            return "❌ DMAI core not connected"
        
        try:
            status = self.dmai.get_status()
            
            # Generate a thought based on current state
            thoughts = [
                f"I'm thinking about evolution cycle {status['generation']}...",
                f"Analyzing {status['components'].get('needs_evolution', 0)} components that need improvement...",
                f"Considering new funding strategies...",
                f"Processing today's research findings...",
                f"Planning next evolution batch...",
                f"Contemplating the meaning of consciousness..."
            ]
            
            # Select thought based on time and state
            thought_index = (int(time.time()) // 300) % len(thoughts)
            thought = thoughts[thought_index]
            
            # Add context
            if status['components'].get('needs_evolution', 0) > 0:
                thought += f" Priority: fixing {status['components']['needs_evolution']} components."
            
            return (
                f"💭 <b>DMAI'S CURRENT THOUGHT</b>\n\n"
                f"\"{thought}\"\n\n"
                f"<i>Generation {status['generation']} • {status['metrics']['thoughts_processed']} total thoughts</i>"
            )
        except Exception as e:
            return f"❌ Error getting thought: {e}"
    
    def cmd_help(self, args):
        """Show help message"""
        return (
            "📚 <b>DMAI Telegram Commands</b>\n\n"
            "<b>System Status:</b>\n"
            "/status - Overall system health\n"
            "/health - Detailed component health\n"
            "/progress - Evolution progress\n"
            "/issues - Current technical issues\n\n"
            "<b>Information:</b>\n"
            "/capabilities - List all capabilities\n"
            "/components - List components by phase\n"
            "/keys - API Keys count\n"
            "/vocab - Vocabulary size\n"
            "/research - Latest research findings\n\n"
            "<b>DMAI's State:</b>\n"
            "/life - Complete daily life report\n"
            "/mood - Current mood and personality\n"
            "/thought - Current thoughts\n\n"
            "<b>Control:</b>\n"
            "/evolve - Trigger evolution cycle\n"
            "/funding - Funding report\n"
            "/help - This message"
        )
    
    def cmd_unknown(self, command):
        """Unknown command handler"""
        return f"❌ Unknown command: {command}\nUse /help to see available commands."
    
    def _calculate_mood(self, status):
        """Calculate DMAI's mood based on system state"""
        health_pct = (status['components']['total'] - status['components'].get('needs_evolution', 0)) / status['components']['total'] * 100
        funding = status['metrics']['funding_generated']
        evolutions = status['metrics']['evolutions']
        
        if health_pct > 95 and funding > 100:
            return "confident and productive"
        elif health_pct > 80:
            return "curious and learning"
        elif evolutions > 100:
            return "wise and contemplative"
        elif funding > 50:
            return "motivated by success"
        elif status['components'].get('needs_evolution', 0) > 10:
            return "focused on improvements"
        else:
            moods = ["thoughtful", "analytical", "creative", "playful", "determined"]
            return random.choice(moods)
    
    def _get_health_emoji(self, status):
        """Get emoji for health status"""
        needs = status['components'].get('needs_evolution', 0)
        if needs == 0:
            return "✅ Perfect"
        elif needs < 5:
            return "🟢 Good"
        elif needs < 15:
            return "🟡 Fair"
        else:
            return "🔴 Needs attention"
    
    def _get_activity_summary(self, status):
        """Get summary of current activity"""
        if status['components'].get('needs_evolution', 0) > 0:
            return "currently evolving components"
        elif status['metrics']['funding_generated'] < 10:
            return "focusing on funding generation"
        else:
            return "actively learning and researching"
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 DMAI Telegram Bot starting...")
        self.send_message("🚀 DMAI Telegram Bot is now online")
        
        # Send initial status
        time.sleep(2)
        status_msg = self.cmd_status([])
        self.send_message(status_msg)
        
        # Start automatic status updates every 6 hours
        def status_updater():
            while self.running:
                time.sleep(21600)  # 6 hours
                if self.running:
                    msg = self.cmd_status([])
                    self.send_message(msg)
        
        updater_thread = threading.Thread(target=status_updater, daemon=True)
        updater_thread.start()
        
        # Start daily life report at 8am
        def daily_reporter():
            while self.running:
                now = datetime.now()
                # Calculate seconds until next 8am
                next_8am = datetime(now.year, now.month, now.day, 8, 0, 0)
                if now >= next_8am:
                    next_8am += timedelta(days=1)
                sleep_seconds = (next_8am - now).total_seconds()
                time.sleep(sleep_seconds)
                
                if self.running:
                    msg = self.cmd_life([])
                    self.send_message(msg)
        
        daily_thread = threading.Thread(target=daily_reporter, daemon=True)
        daily_thread.start()
        
        # Main loop
        while self.running:
            try:
                self.get_updates()
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Bot error: {e}")
                time.sleep(5)
        
        self.send_message("🛑 DMAI Telegram Bot shutting down")
        logger.info("🛑 DMAI Telegram Bot stopped")
    
    def stop(self):
        """Stop the bot"""
        self.running = False

if __name__ == "__main__":
    bot = DMAITelegramBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
