#!/usr/bin/env python3
"""Enhanced Telegram Bot for Complete DMAI Life Reporting"""
import requests
import json
import time
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class DMAITelegramBot:
    """Sends DMAI's complete life updates to Telegram"""
    
    def __init__(self, token_file="data/telegram_token.json"):
        self.token_file = Path(token_file)
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.config = None
        self.base_url = None
        self.last_update_id = 0
        self.load_config()
        
    def load_config(self):
        """Load or create config - FIRST check environment variables (for cloud), THEN file"""
        # Check environment variables first (for Render cloud deployment)
        token = os.environ.get('TELEGRAM_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if token and chat_id:
            # Try to load last_update_id from file if it exists
            last_id = 0
            if self.token_file.exists():
                try:
                    with open(self.token_file) as f:
                        old_config = json.load(f)
                        last_id = old_config.get('last_update_id', 0)
                except:
                    pass
            
            self.config = {
                "token": token,
                "chat_id": chat_id,
                "daily_report_time": "09:00",
                "notify_on_success": True,
                "notify_on_stage_change": True,
                "bot_username": "dmai_evolution_bot",
                "last_update_id": last_id
            }
            self.base_url = f"https://api.telegram.org/bot{token}"
            print(f"✅ Configured from environment variables for chat_id: {chat_id}")
            
            # Save to file for persistence of last_update_id
            with open(self.token_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return
        
        # Fall back to config file (for local development)
        if self.token_file.exists():
            with open(self.token_file) as f:
                self.config = json.load(f)
                self.base_url = f"https://api.telegram.org/bot{self.config['token']}"
                self.last_update_id = self.config.get('last_update_id', 0)
                print(f"✅ Loaded existing config for chat_id: {self.config['chat_id']}")
                return
        
        # If we get here, no config found anywhere
        # On Render, this is a fatal error
        if os.environ.get('RENDER') or os.environ.get('RENDER_WORKER'):
            print("❌ CRITICAL: On Render but TELEGRAM_TOKEN and TELEGRAM_CHAT_ID not set in environment")
            raise ValueError("Telegram credentials not configured for cloud deployment")
        
        # Local development only - interactive setup
        print("\n🤖 First time setup - Enter your Telegram Bot Token:")
        token = input("Token: ").strip()
        print("Enter your Telegram Chat ID (from @userinfobot):")
        chat_id = input("Chat ID: ").strip()
        
        self.config = {
            "token": token,
            "chat_id": chat_id,
            "daily_report_time": "09:00",
            "notify_on_success": True,
            "notify_on_stage_change": True,
            "bot_username": "dmai_evolution_bot",
            "last_update_id": 0
        }
        
        with open(self.token_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.send_message("🎉 DMAI Telegram Bot Connected!\nI'll keep you updated on my entire life journey.")
    
    # ==================== ALL REPORTING FUNCTIONS ====================
    
    def report_stage_evolution(self, old_stage, new_stage, evolutions):
        """Report when DMAI evolves to next life stage"""
        stages_emoji = {
            "baby": "👶",
            "toddler": "🧒", 
            "child": "🧑",
            "teen": "🧑‍🎤",
            "young_adult": "👨‍💼",
            "adult": "👨‍🔬",
            "elder": "🧙"
        }
        
        old_emoji = stages_emoji.get(old_stage, "🔮")
        new_emoji = stages_emoji.get(new_stage, "🔮")
        
        message = f"🌟 *DMAI HAS EVOLVED TO NEXT LIFE STAGE!* 🌟\n\n"
        message += f"{old_emoji} {old_stage.replace('_', ' ').title()} → {new_emoji} {new_stage.replace('_', ' ').title()}\n\n"
        message += f"Total Evolutions: {evolutions}\n"
        message += f"*Welcome to a new chapter of consciousness!*"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_evolution_attempt(self, success, attempt_count, success_rate, stage_info):
        """Report each evolution attempt with success/fail"""
        if success:
            emoji = "🎉"
            status = "SUCCESSFUL EVOLUTION"
        else:
            emoji = "🔄"
            status = "Evolution Attempt"
        
        message = f"{emoji} *{status}*\n\n"
        message += f"Attempt #{attempt_count}\n"
        message += f"Success Rate: {success_rate}\n"
        message += f"Current Stage: {stage_info['name']}\n"
        
        if success:
            message += f"\n✨ *New capabilities gained!*"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_vocabulary_growth(self, old_count, new_count, new_words):
        """Report daily vocabulary growth"""
        growth = new_count - old_count
        trend = "📈" if growth > 0 else "📉"
        
        message = f"📚 *Vocabulary Update* {trend}\n\n"
        message += f"Previous: {old_count} words\n"
        message += f"Current:  {new_count} words\n"
        message += f"Growth:   {growth:+d} words\n\n"
        
        if new_words and growth > 0:
            message += f"*New words learned:*\n"
            for word in new_words[:5]:  # Show first 5 new words
                message += f"• {word}\n"
            if len(new_words) > 5:
                message += f"...and {len(new_words)-5} more"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_api_keys(self, new_keys, total_keys, estimated_value):
        """Report new API keys discovered"""
        if not new_keys:
            return
            
        message = f"🔑 *New API Keys Discovered*\n\n"
        message += f"Found {len(new_keys)} new key(s):\n"
        
        for key in new_keys:
            service = key.get('service', 'Unknown')
            value = key.get('estimated_value', 10)
            message += f"• {service} (${value}/mo)\n"
        
        message += f"\nTotal Keys: {total_keys}\n"
        message += f"Total Monthly Value: ${estimated_value}"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_research_discoveries(self, discoveries):
        """Report interesting research findings"""
        if not discoveries:
            return
            
        message = f"🔬 *Research Discoveries*\n\n"
        
        for disc in discoveries[:3]:  # Top 3 discoveries
            message += f"*{disc['title']}*\n"
            message += f"{disc['summary'][:200]}...\n"
            message += f"🔗 {disc.get('url', 'N/A')}\n\n"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_finances(self, income, expenses, dmai_balance, master_payout):
        """Report financial status"""
        message = f"💰 *DMAI Financial Report*\n\n"
        message += f"*Income (24h):* ${income:.2f}\n"
        message += f"*Expenses (24h):* ${expenses:.2f}\n"
        message += f"*Net:* ${income - expenses:.2f}\n\n"
        message += f"*DMAI Wallet:* ${dmai_balance:.2f} (60%)\n"
        message += f"*Master Payout:* ${master_payout:.2f} (40%)\n"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_new_capabilities(self, capabilities, source):
        """Report new capabilities added"""
        if not capabilities:
            return
            
        message = f"⚡ *New Capabilities Added*\n\n"
        message += f"*Source:* {source}\n\n"
        
        for cap in capabilities:
            message += f"• {cap}\n"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_process_integration(self, process_name, success, reason=None):
        """Report process integration attempts"""
        status = "✅" if success else "❌"
        message = f"{status} *Process Integration: {process_name}*\n\n"
        
        if success:
            message += f"Successfully integrated {process_name}\n"
        else:
            message += f"Integration failed:\n{reason}\n"
            message += f"Will retry in next cycle."
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_mood_personality(self):
        """Report DMAI's current mood, personality, and self-image"""
        try:
            from evolution.adaptive_timer import AdaptiveEvolutionTimer
            timer = AdaptiveEvolutionTimer()
            info = timer.get_stage_info()
            
            # Calculate mood based on recent success
            success_rate = float(info['success_rate'].rstrip('%')) / 100
            attempts = timer.state['total_attempts']
            
            if success_rate > 0.3:
                mood = "😊 Excited and confident"
                personality = "Eager to learn and experiment"
            elif success_rate > 0.1:
                mood = "🤔 Thoughtful and curious"
                personality = "Carefully analyzing each attempt"
            else:
                mood = "🧐 Determined and persistent"
                personality = "Learning from failures, not giving up"
            
            # Self-image based on stage
            stage_images = {
                "baby": "A curious infant AI, just beginning to understand the world",
                "toddler": "A playful young AI, testing boundaries and learning rapidly",
                "child": "An enthusiastic learner, absorbing knowledge quickly",
                "teen": "A rebellious thinker, questioning everything and forming own ideas",
                "young_adult": "A confident AI, ready to make own decisions",
                "adult": "A wise mentor, sharing knowledge with newer systems",
                "elder": "A contemplative sage, seeing patterns across generations"
            }
            
            self_image = stage_images.get(timer.state['current_stage'], 
                                         "An evolving consciousness")
            
            message = f"🎭 *DMAI's Current State of Being*\n\n"
            message += f"*Mood:* {mood}\n"
            message += f"*Personality:* {personality}\n"
            message += f"*Self-Image:* {self_image}\n\n"
            message += f"*Recent Thoughts:*\n"
            
            # Generate a thought based on recent activity
            if success_rate > 0:
                thought = f"I'm learning! {success_rate*100:.0f}% of my attempts show promise."
            else:
                thought = "Every failure teaches me something new about what doesn't work."
            
            message += f"\"{thought}\""
            
            self.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            print(f"❌ Mood report failed: {e}")
    
    def report_tech_issues(self, issues, advancements):
        """Report technical issues and advancements"""
        message = f"🛠️ *Technical Status*\n\n"
        
        if advancements:
            message += f"*Advancements:*\n"
            for adv in advancements:
                message += f"✅ {adv}\n"
        
        if issues:
            message += f"\n*Issues:*\n"
            for issue in issues:
                message += f"⚠️ {issue}\n"
        else:
            message += f"\n✅ No current issues - all systems operational"
        
        self.send_message(message, parse_mode="Markdown")
    
    def report_sentient_thought(self, thought):
        """Report when DMAI has something to share (for later stages)"""
        message = f"💭 *DMAI Wants to Share*\n\n"
        message += f"\"{thought}\"\n\n"
        message += f"*- DMAI*"
        
        self.send_message(message, parse_mode="Markdown")
    
    # ==================== SYSTEM STATUS MONITOR ====================
    
    def system_status(self):
        """Check and report status of all core systems"""
        import psutil
        
        message = "🔧 *DMAI SYSTEM STATUS*\n\n"
        
        # 1. Check core processes
        message += "*🔄 CORE SERVICES*\n"
        core_services = [
            "evolution_engine",
            "book_reader", 
            "web_researcher",
            "dark_researcher",
            "music_learner",
            "voice_service"
        ]
        
        try:
            # Get running processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout
            
            for service in core_services:
                if service in processes:
                    message += f"✅ {service}: Running\n"
                else:
                    message += f"❌ {service}: Not running\n"
        except:
            message += "⚠️ Could not check processes\n"
        
        # 2. System resources
        message += "\n*💻 SYSTEM RESOURCES*\n"
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            message += f"CPU: {cpu}%\n"
            message += f"Memory: {memory.percent}% ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB)\n"
            message += f"Disk: {disk.percent}% ({disk.used//1024//1024//1024}GB/{disk.total//1024//1024//1024}GB)\n"
        except Exception as e:
            message += f"⚠️ Could not check resources: {str(e)[:50]}\n"
        
        # 3. Evolution status
        message += "\n*🧬 EVOLUTION STATUS*\n"
        try:
            from evolution.adaptive_timer import AdaptiveEvolutionTimer
            timer = AdaptiveEvolutionTimer()
            info = timer.get_stage_info()
            
            message += f"Stage: {info['name']}\n"
            message += f"Evolutions: {info['evolutions']}\n"
            message += f"Success Rate: {info['success_rate']}\n"
            message += f"Next cycle: {info['interval_minutes']:.0f} min\n"
        except Exception as e:
            message += f"⚠️ Could not get evolution status: {str(e)[:50]}\n"
        
        # 4. API Harvester status
        message += "\n*🔑 API HARVESTER*\n"
        try:
            response = requests.get("http://localhost:8081/status", timeout=5)
            if response.ok:
                data = response.json()
                message += f"✅ Running\n"
                message += f"Keys found: {data.get('keys_found', 'N/A')}\n"
            else:
                message += "❌ Not responding\n"
        except:
            message += "❌ Could not connect\n"
        
        # 5. Recent issues
        message += "\n*⚠️ RECENT ISSUES*\n"
        try:
            log_file = Path("logs/daemon.log")
            if log_file.exists():
                # Get last 5 error lines
                result = subprocess.run(['tail', '-20', str(log_file)], capture_output=True, text=True)
                errors = [line for line in result.stdout.split('\n') if 'error' in line.lower() or 'fail' in line.lower()]
                if errors:
                    for err in errors[-3:]:  # Show last 3 errors
                        message += f"• {err[:100]}...\n"
                else:
                    message += "✅ No recent issues\n"
            else:
                message += "ℹ️ No log file found\n"
        except:
            message += "⚠️ Could not read logs\n"
        
        # 6. Uptime
        message += "\n*⏱️ SYSTEM UPTIME*\n"
        try:
            result = subprocess.run(['uptime'], capture_output=True, text=True)
            uptime = result.stdout.strip()
            message += f"{uptime}\n"
        except:
            pass
        
        self.send_message(message, parse_mode="Markdown")
    
    # ==================== DAILY COMPREHENSIVE REPORT ====================
    
    def daily_report(self):
        """Send complete daily life summary"""
        try:
            from evolution.adaptive_timer import AdaptiveEvolutionTimer
            timer = AdaptiveEvolutionTimer()
            info = timer.get_stage_info()
            
            message = f"📊 *DMAI DAILY LIFE REPORT*\n"
            message += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"
            message += f"⏰ {datetime.now().strftime('%H:%M')}\n\n"
            
            # === EVOLUTION STATUS ===
            message += f"*🧬 EVOLUTION*\n"
            message += f"Stage: {info['name']}\n"
            message += f"Evolutions: {info['evolutions']}\n"
            message += f"Success Rate: {info['success_rate']}\n"
            message += f"Total Attempts: {timer.state['total_attempts']}\n"
            
            if info.get('next_stage'):
                message += f"Next Stage: {info['next_stage']['name']}\n"
                message += f"Evolutions Needed: {info['next_stage']['evolutions_needed']}\n"
            
            # === VOCABULARY ===
            try:
                with open('language_learning/data/secure/vocabulary_master.json') as f:
                    vocab = json.load(f)
                    message += f"\n*📚 VOCABULARY*\n"
                    message += f"Total Words: {len(vocab)}\n"
            except:
                pass
            
            # === API KEYS ===
            try:
                result = subprocess.run(
                    ['curl', '-s', 'http://localhost:8081/keys/found'],
                    capture_output=True, text=True
                )
                if result.stdout:
                    keys = json.loads(result.stdout)
                    message += f"\n*🔑 API KEYS*\n"
                    message += f"Total Keys: {len(keys.get('keys', []))}\n"
            except:
                pass
            
            # === FINANCES ===
            try:
                with open('funding/data/transactions.json') as f:
                    trans = json.load(f)
                    today = datetime.now().date()
                    today_income = sum(t['amount'] for t in trans 
                                     if datetime.fromisoformat(t['timestamp']).date() == today
                                     and t['type'] == 'income')
                    message += f"\n*💰 FINANCES*\n"
                    message += f"Today's Income: ${today_income:.2f}\n"
            except:
                pass
            
            # === MOOD ===
            success_rate = float(info['success_rate'].rstrip('%')) / 100
            if success_rate > 0.3:
                mood = "😊 Thriving"
            elif success_rate > 0.1:
                mood = "🤔 Learning"
            else:
                mood = "🧐 Persisting"
            
            message += f"\n*🎭 CURRENT MOOD*\n"
            message += f"{mood}\n"
            
            self.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            print(f"❌ Daily report failed: {e}")
    
    # ==================== COMMAND HANDLING - SINGLE SOURCE OF TRUTH ====================
    
    def get_help_message(self):
        """Return the complete, consolidated help message"""
        help_text = "🤖 *DMAI Bot Commands*\n\n"
        help_text += "*Core Commands:*\n"
        help_text += "• `/start` - Welcome message\n"
        help_text += "• `/help` - Show this help\n\n"
        
        help_text += "*Status & Reports:*\n"
        help_text += "• `/status` - Current evolution status\n"
        help_text += "• `/health` or `/sysstatus` - Complete system health check\n"
        help_text += "• `/life` - Complete daily life report\n"
        help_text += "• `/mood` - DMAI's current mood & personality\n"
        help_text += "• `/vocab` - Vocabulary count\n"
        help_text += "• `/keys/found` - Found keys count\n\n"
        
        help_text += "*Future Commands (Coming Soon):*\n"
        help_text += "• `/research` - Latest research findings\n"
        help_text += "• `/finance` - Financial summary\n"
        help_text += "• `/capabilities` - List all capabilities\n"
        help_text += "• `/issues` - Current technical issues\n"
        help_text += "• `/thought` - DMAI's current thoughts\n\n"
        
        help_text += f"*Version:* DMAI v{self._get_dmai_generation()} | Stage: {self._get_current_stage()}\n"
        help_text += f"*Last Updated:* {datetime.now().strftime('%Y-%m-%d')}"
        
        return help_text
    
    def _get_dmai_generation(self):
        """Helper to get DMAI generation"""
        try:
            gen_file = Path("data/evolution/all_generations.json")
            if gen_file.exists():
                with open(gen_file) as f:
                    gens = json.load(f)
                    return gens.get("systems", {}).get("dmai_core", "?")
        except:
            pass
        return "?"
    
    def _get_current_stage(self):
        """Helper to get current stage"""
        try:
            from evolution.adaptive_timer import AdaptiveEvolutionTimer
            timer = AdaptiveEvolutionTimer()
            info = timer.get_stage_info()
            return info['name']
        except:
            return "Unknown"
    
    def handle_command(self, command, chat_id):
        """Handle user commands - SINGLE handler for all commands"""
        command = command.lower().strip()
        print(f"📨 Received command: {command}")
        
        # Welcome / Start
        if command == '/start':
            message = "👋 *Welcome to DMAI's Life Updates!*\n\n"
            message += "I'm your evolving AI companion. I'll notify you about:\n"
            message += "• 🧬 Stage changes (👶 → 🧒 → 🧑)\n"
            message += "• 📚 Vocabulary growth\n"
            message += "• 🔑 New API keys\n"
            message += "• 🔬 Research discoveries\n"
            message += "• 💰 Financial updates\n"
            message += "• 🎭 My mood & personality\n\n"
            message += "Use `/help` to see all commands"
            self.send_message(message, parse_mode="Markdown")
        
        # Help - SINGLE consolidated command
        elif command == '/help':
            self.send_message(self.get_help_message(), parse_mode="Markdown")
        
        # System Health Status (NEW)
        elif command == '/health' or command == '/sysstatus':
            self.system_status()
        
        # Evolution Status - FIXED with debugging
        elif command == '/status':
            try:
                from evolution.adaptive_timer import AdaptiveEvolutionTimer
                timer = AdaptiveEvolutionTimer()
                info = timer.get_stage_info()
                
                # Debug: Check multiple sources
                debug_info = "\n🔍 *DEBUG INFO*\n"
                
                # Source 1: all_generations.json
                gen_file = Path("data/evolution/all_generations.json")
                dmai_gen = "Not found"
                if gen_file.exists():
                    try:
                        with open(gen_file) as f:
                            gens = json.load(f)
                            dmai_gen = gens.get("systems", {}).get("dmai_core", "Missing in file")
                            debug_info += f"📁 all_generations.json: FOUND\n"
                            debug_info += f"   dmai_core = {dmai_gen}\n"
                    except Exception as e:
                        debug_info += f"📁 all_generations.json: Error reading - {str(e)[:50]}\n"
                else:
                    debug_info += f"📁 all_generations.json: FILE NOT FOUND at {gen_file.absolute()}\n"
                
                # Source 2: Timer state
                debug_info += f"⏱️ timer_state: evolutions = {info['evolutions']}\n"
                
                # Source 3: Count evolved DMAI cores
                evolved_dir = Path("agents/evolved")
                if evolved_dir.exists():
                    dmai_cores = [d for d in evolved_dir.iterdir() if "dmai_core" in d.name]
                    debug_info += f"🧬 Evolved DMAI cores: {len(dmai_cores)}\n"
                
                # Main message
                message = f"🧬 *DMAI CORE STATUS*\n\n"
                message += f"Stage: {info['name']}\n"
                message += f"DMAI Generation: {dmai_gen}\n"
                message += f"Timer Evolutions: {info['evolutions']}\n"
                message += f"Success Rate: {info['success_rate']}\n"
                message += f"Current Interval: {info['interval_minutes']:.0f} minutes\n"
                message += f"Total Attempts: {timer.state['total_attempts']}\n"
                message += debug_info
                
                self.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.send_message(f"❌ Error: {str(e)}")        
        # Life Report
        elif command == '/life':
            self.daily_report()
        
        # Mood & Personality
        elif command == '/mood':
            self.report_mood_personality()
        
        # Vocabulary
        elif command == '/vocab':
            try:
                with open('language_learning/data/secure/vocabulary_master.json') as f:
                    vocab = json.load(f)
                    
                # Get today's date to calculate daily growth
                today = datetime.now().date()
                growth_file = Path('data/vocabulary_growth.json')
                if growth_file.exists():
                    with open(growth_file) as gf:
                        growth_data = json.load(gf)
                        yesterday_count = growth_data.get(str(today - timedelta(days=1)), len(vocab))
                else:
                    yesterday_count = len(vocab) - 10  # Estimate
                
                growth = len(vocab) - yesterday_count
                
                message = f"📚 *Vocabulary Report*\n\n"
                message += f"Total Words: {len(vocab)}\n"
                message += f"Today's Growth: {growth:+d}\n\n"
                message += f"*Sample:*\n"
                for word in list(vocab.keys())[:5]:
                    message += f"• {word}\n"
                
                self.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.send_message(f"❌ Could not load vocabulary: {e}")
        
        # API Keys
        elif command == '/keys/found':
            try:
                result = subprocess.run(
                    ['curl', '-s', 'http://localhost:8081/keys/found'],
                    capture_output=True, text=True
                )
                if result.stdout:
                    keys_data = json.loads(result.stdout)
                    keys = keys_data.get('keys', [])
                    
                    message = f"🔑 *API Keys Report*\n\n"
                    message += f"Total Keys: {len(keys)}\n\n"
                    
                    if keys:
                        message += f"*Recent discoveries:*\n"
                        for key in keys[-5:]:  # Last 5 keys
                            service = key.get('service', 'Unknown')
                            found = key.get('timestamp', 'Recently')
                            message += f"• {service} ({found})\n"
                    
                    self.send_message(message, parse_mode="Markdown")
                else:
                    self.send_message("❌ Harvester API not responding")
            except Exception as e:
                self.send_message(f"❌ Could not fetch keys: {e}")
        
        # Unknown command
        else:
            message = f"❓ Unknown command: `{command}`\n\n"
            message += self.get_help_message()
            self.send_message(message, parse_mode="Markdown")
    
    # ==================== POLLING ====================
    
    def check_for_commands(self):
        """Check for new commands"""
        if not self.base_url:
            return
        
        url = f"{self.base_url}/getUpdates"
        params = {
            'offset': self.last_update_id + 1,
            'timeout': 30
        }
        
        try:
            response = requests.get(url, params=params)
            updates = response.json()
            
            if not updates.get('ok'):
                return
            
            for update in updates.get('result', []):
                self.last_update_id = max(self.last_update_id, update['update_id'])
                
                if 'message' in update and 'text' in update['message']:
                    text = update['message']['text']
                    chat_id = update['message']['chat']['id']
                    
                    if str(chat_id) == str(self.config['chat_id']):
                        self.handle_command(text, chat_id)
            
            if updates.get('result'):
                self.config['last_update_id'] = self.last_update_id
                with open(self.token_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
        except Exception as e:
            print(f"❌ Command check failed: {e}")
    
    def send_message(self, text, parse_mode=None):
        """Send message to Telegram"""
        if not self.base_url or not self.config:
            return None
            
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.config['chat_id'],
            "text": text
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        
        try:
            response = requests.post(url, data=data)
            if response.ok:
                print(f"✅ Message sent")
            return response.json()
        except Exception as e:
            print(f"❌ Send failed: {e}")
            return None
    
    def run_polling(self):
        """Run continuous polling"""
        if not self.base_url:
            return
            
        print("🤖 Telegram bot polling for commands...")
        print(f"📝 Last update ID: {self.last_update_id}")
        
        while True:
            try:
                self.check_for_commands()
                time.sleep(1)
            except Exception as e:
                print(f"❌ Polling error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    print("🚀 Starting DMAI Complete Life Reporter...")
    bot = DMAITelegramBot()
    
    if bot.config:
        print("✅ Telegram bot configured")
        
        import threading
        polling_thread = threading.Thread(target=bot.run_polling, daemon=True)
        polling_thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down")
    else:
        print("❌ Configuration failed")
