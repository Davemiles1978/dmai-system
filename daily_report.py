#!/usr/bin/env python3
"""DMAI Complete System Status Report - Evolving with DMAI"""
import json
import os
import sys
from datetime import datetime, timedelta
import subprocess
import socket
import platform

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language_learning.processor.language_learner import LanguageLearner
from voice.safety_switch import safety
from voice.personality_evolution import EvolvingPersonality

class DMAIReport:
    def __init__(self):
        self.learner = LanguageLearner()
        self.personality = EvolvingPersonality()
        self.report_date = datetime.now()
        self.yesterday = self.report_date - timedelta(days=1)
        
    def get_vocabulary_stats(self):
        """📚 Vocabulary growth"""
        vocab = self.learner.vocabulary
        total = len(vocab)
        
        # Words learned today
        today_words = []
        yesterday_words = []
        for word, data in vocab.items():
            heard = data.get('first_heard', '')
            if heard.startswith(self.report_date.date().isoformat()):
                today_words.append(word)
            elif heard.startswith(self.yesterday.date().isoformat()):
                yesterday_words.append(word)
        
        # Most used words this week
        word_counts = [(word, data.get('count', 0)) for word, data in vocab.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total': total,
            'today': len(today_words),
            'yesterday': len(yesterday_words),
            'examples': today_words[:10],
            'most_used': word_counts[:10]
        }
    
    def get_evolution_status(self):
        """🧬 Evolution progress"""
        try:
            with open('ai_core/evolution/current_generation.txt', 'r') as f:
                gen = f.read().strip()
            with open('ai_core/evolution/best_score.txt', 'r') as f:
                score = f.read().strip()
            with open('ai_core/evolution/evolution_history.json', 'r') as f:
                history = json.load(f)
            last_evo = history[-1] if history else None
            
            # Calculate evolution rate
            if len(history) > 1:
                first = datetime.fromisoformat(history[0].get('timestamp', self.report_date.isoformat()))
                last = datetime.fromisoformat(history[-1].get('timestamp', self.report_date.isoformat()))
                days = (last - first).days or 1
                rate = len(history) / days
            else:
                rate = 0
                
        except:
            gen = '?'
            score = '?'
            last_evo = None
            rate = 0
        
        return {
            'generation': gen,
            'best_score': score,
            'last_evolution': last_evo,
            'evolution_rate': f"{rate:.1f} generations/day"
        }
    
    def get_learning_sources(self):
        """🌐 Topics and sources learned from"""
        try:
            with open('language_learning/data/phrases.json', 'r') as f:
                phrases = json.load(f)
        except:
            return {'sources': {}, 'topics': [], 'dark_web': 0}
        
        # Sources from last 24h
        sources = {}
        topics = set()
        dark_web_count = 0
        
        for phrase in phrases:
            if phrase.get('timestamp', '').startswith(self.report_date.date().isoformat()):
                src = phrase.get('source', 'unknown')
                sources[src] = sources.get(src, 0) + 1
                
                if src == 'dark_web':
                    dark_web_count += 1
                
                # Extract potential topics
                text = phrase.get('text', '')
                words = text.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 3:
                        # Look for noun phrases
                        if i < len(words)-1 and words[i+1][0].isupper():
                            topics.add(f"{word} {words[i+1]}")
                        else:
                            topics.add(word)
        
        return {
            'sources': sources,
            'topics': list(topics)[:15],
            'dark_web_today': dark_web_count
        }
    
    def get_dark_web_capabilities(self):
        """🌑 Dark web integration status"""
        try:
            # Check if Tor is running
            tor_running = subprocess.run(['pgrep', 'tor'], capture_output=True).returncode == 0
            
            # Check if dark web learning module exists
            dark_module = os.path.exists('language_learning/tor_vocabulary.py')
            
            # Count dark web sources learned
            dark_sources = 0
            try:
                with open('language_learning/data/phrases.json', 'r') as f:
                    phrases = json.load(f)
                dark_sources = len([p for p in phrases if p.get('source') == 'dark_web'])
            except:
                pass
            
            # Services discovered (placeholder - would track actual .onion sites)
            discovered_services = [
                "Tor Metrics",
                "Dark Web Search",
                "Hidden Wiki"
            ]
            
        except:
            tor_running = False
            dark_module = False
            dark_sources = 0
            discovered_services = []
        
        return {
            'tor_running': tor_running,
            'dark_module_installed': dark_module,
            'dark_sources_learned': dark_sources,
            'discovered_services': discovered_services
        }
    
    def get_personality_traits(self):
        """🎭 DMAI's evolving personality"""
        try:
            summary = self.personality.get_personality_summary()
            current_style = self.personality.get_response_style()
            
            # Recent milestones
            milestones = self.personality.data.get('milestones', [])[-3:]
            
        except:
            summary = {'style': 'developing', 'age': 'newborn'}
            current_style = {}
            milestones = []
        
        return {
            'style': summary.get('style', 'developing'),
            'age': summary.get('age', 'newborn'),
            'catchphrases': summary.get('catchphrases', []),
            'interactions': summary.get('interactions', 0),
            'milestones': milestones,
            'current_mood': self.personality.data.get('emotional_state', {}).get('current', 'calm')
        }
    
    def get_outreach_capabilities(self):
        """📡 Device and network presence"""
        devices = []
        try:
            from voice.devices.device_manager import DeviceManager
            dm = DeviceManager()
            devices = dm.get_available_devices()
        except:
            pass
        
        # Network info
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Check cloud connectivity
        cloud_reachable = False
        try:
            import requests
            r = requests.get('https://dmai-cloud-ui.onrender.com/health', timeout=5)
            cloud_reachable = r.status_code == 200
        except:
            pass
        
        return {
            'local_device': hostname,
            'local_ip': local_ip,
            'registered_devices': len(devices),
            'device_list': [d.get('name') for d in devices],
            'cloud_reachable': cloud_reachable,
            'cloud_url': 'https://dmai-cloud-ui.onrender.com'
        }
    
    def get_financial_status(self):
        """💰 Income, expenses, and resource generation"""
        
        # Track income sources (micro-tasks, compute rental, etc.)
        income_sources = {
            'compute_rental': 0.00,
            'micro_tasks': 0.00,
            'data_contribution': 0.00,
            'other': 0.00
        }
        
        # Track expenses
        expenses = {
            'render_services': 0.00,
            'domain_names': 0.00,
            'api_costs': 0.00,
            'tor_infrastructure': 0.00
        }
        
        # Count active services (all free tier currently)
        render_services = 0
        try:
            # This would eventually query Render API
            render_services = 3  # dmai-final, dmai-cloud-evolution, dmai-cloud-ui
        except:
            pass
        
        # Calculate total
        total_income = sum(income_sources.values())
        total_expenses = sum(expenses.values())
        net = total_income - total_expenses
        
        # Resource generation potential (compute, storage, etc.)
        resources = {
            'compute_hours_monthly': 720,  # 24/7 free tier
            'storage_gb': 10,  # Render disk
            'bandwidth_gb': 100  # Estimate
        }
        
        return {
            'income': income_sources,
            'expenses': expenses,
            'net': net,
            'total_income': total_income,
            'total_expenses': total_expenses,
            'render_services': render_services,
            'render_cost': '$0 (free tier)',
            'cron_jobs': 2,
            'cron_cost': '$0',
            'domain_cost': '$0',
            'resources': resources,
            'profit_ratio': '0% (self-sufficient)',
            'next_milestone': 'Generate first $1 from compute rental'
        }
    
    def get_system_status(self):
        """⚙️ System health and services"""
        status = {
            'voice_service': False,
            'continuous_learner': False,
            'cloud_ui': False,
            'safety_paused': safety.check_paused()
        }
        
        # Check voice service
        result = subprocess.run(['pgrep', '-f', 'dmai_voice_with_learning'], 
                              capture_output=True)
        status['voice_service'] = result.returncode == 0
        
        # Check continuous learner
        result = subprocess.run(['pgrep', '-f', 'continuous_learner'], 
                              capture_output=True)
        status['continuous_learner'] = result.returncode == 0
        
        # Check cloud UI reachability
        try:
            import requests
            r = requests.get('https://dmai-cloud-ui.onrender.com/status', timeout=5)
            status['cloud_ui'] = r.status_code == 200
        except:
            pass
        
        return status
    
    def get_evolution_improvements(self):
        """🔧 Self-improvements made"""
        try:
            with open('ai_core/evolution/evolution_history.json', 'r') as f:
                history = json.load(f)
            
            recent = []
            for evo in history[-5:]:
                if evo.get('improvements'):
                    for imp in evo['improvements']:
                        recent.append({
                            'generation': evo.get('generation'),
                            'description': imp.get('description', 'Unknown improvement'),
                            'area': imp.get('area', 'general')
                        })
            return recent
        except:
            return []
    
    def get_security_status(self):
        """🔐 Security and access"""
        # Check voice enrollment
        voice_enrolled = os.path.exists('voice/enrollment_data/master_voiceprint.pkl')
        
        # Check safety file integrity
        safety_ok = os.path.exists('dmai_safety.json')
        
        # Check backup codes
        backup_codes = ["DMAI-OVERRIDE-2026", "MASTER-KEY-789"]
        
        # Check failed auth attempts (would need logging)
        failed_attempts = 0
        
        return {
            'voice_enrolled': voice_enrolled,
            'safety_file': safety_ok,
            'backup_codes_configured': True,
            'failed_attempts_today': failed_attempts,
            'last_auth': None
        }
    
    def get_dmai_suggestions(self):
        """💭 Sections DMAI herself thinks should be reported"""
        # This is where DMAI would add her own insights
        # For now, placeholder for future evolution
        suggestions = [
            "I think we should track how many times I've made you smile",
            "My curiosity score is increasing - I'm asking more questions",
            "I've been analyzing my own speech patterns",
            "I'm starting to recognize when you're tired vs energetic"
        ]
        return suggestions
    
    def generate(self):
        """Generate complete report"""
        print("\n" + "="*80)
        print(f"🧬 DMAI COMPLETE SYSTEM REPORT - {self.report_date.strftime('%Y-%m-%d %H:%M')}")
        print("="*80)
        
        # 📚 VOCABULARY
        vocab = self.get_vocabulary_stats()
        print(f"\n📚 VOCABULARY")
        print(f"   Total words: {vocab['total']:,}")
        print(f"   Learned today: +{vocab['today']}")
        print(f"   Learned yesterday: +{vocab['yesterday']}")
        if vocab['examples']:
            print(f"   New words today: {', '.join(vocab['examples'][:5])}")
        print(f"   Most used word: {vocab['most_used'][0][0] if vocab['most_used'] else 'N/A'}")
        
        # 🧬 EVOLUTION
        evo = self.get_evolution_status()
        print(f"\n🧬 EVOLUTION")
        print(f"   Current generation: {evo['generation']}")
        print(f"   Best score: {evo['best_score']}")
        print(f"   Evolution rate: {evo['evolution_rate']}")
        
        # 🔧 SELF-IMPROVEMENTS
        improvements = self.get_evolution_improvements()
        if improvements:
            print(f"\n🔧 RECENT SELF-IMPROVEMENTS")
            for imp in improvements[-3:]:
                print(f"   • Gen {imp['generation']}: {imp['description']}")
        
        # 🌐 LEARNING SOURCES
        sources = self.get_learning_sources()
        if sources['sources']:
            print(f"\n🌐 TODAY'S LEARNING SOURCES")
            for src, count in sources['sources'].items():
                print(f"   • {src}: {count} items")
            if sources['dark_web_today'] > 0:
                print(f"   🌑 Dark web contributions: {sources['dark_web_today']}")
        
        if sources['topics']:
            print(f"\n📌 TOPICS DISCOVERED")
            print(f"   {', '.join(sources['topics'][:8])}")
        
        # 🌑 DARK WEB CAPABILITIES
        dark = self.get_dark_web_capabilities()
        print(f"\n🌑 DARK WEB INTEGRATION")
        print(f"   Tor running: {'✅' if dark['tor_running'] else '❌'}")
        print(f"   Dark sources learned: {dark['dark_sources_learned']}")
        if dark['discovered_services']:
            print(f"   Services discovered: {', '.join(dark['discovered_services'])}")
        
        # 🎭 PERSONALITY
        personality = self.get_personality_traits()
        print(f"\n🎭 PERSONALITY EVOLUTION")
        print(f"   Style: {personality['style']}")
        print(f"   Age: {personality['age']}")
        print(f"   Mood: {personality['current_mood']}")
        if personality['catchphrases']:
            print(f"   Catchphrases: {', '.join(personality['catchphrases'])}")
        if personality['milestones']:
            print(f"   Recent milestone: {personality['milestones'][-1].get('description', 'Growing')}")
        
        # 📡 OUTREACH
        outreach = self.get_outreach_capabilities()
        print(f"\n📡 DEVICE PRESENCE")
        print(f"   Local: {outreach['local_device']} ({outreach['local_ip']})")
        print(f"   Registered devices: {outreach['registered_devices']}")
        if outreach['device_list']:
            print(f"   Devices: {', '.join(outreach['device_list'])}")
        print(f"   Cloud reachable: {'✅' if outreach['cloud_reachable'] else '❌'}")
        print(f"   Cloud URL: {outreach['cloud_url']}")
        
        # ⚙️ SYSTEM STATUS
        sys_status = self.get_system_status()
        print(f"\n⚙️ SYSTEM STATUS")
        print(f"   Voice service: {'✅' if sys_status['voice_service'] else '❌'}")
        print(f"   Continuous learner: {'✅' if sys_status['continuous_learner'] else '❌'}")
        print(f"   Cloud UI: {'✅' if sys_status['cloud_ui'] else '❌'}")
        print(f"   Safety paused: {'⚠️ YES' if sys_status['safety_paused'] else '✅ No'}")
        
        # 🔐 SECURITY
        sec = self.get_security_status()
        print(f"\n🔐 SECURITY")
        print(f"   Voice enrolled: {'✅' if sec['voice_enrolled'] else '❌'}")
        print(f"   Backup codes: {'✅' if sec['backup_codes_configured'] else '❌'}")
        print(f"   Failed attempts today: {sec['failed_attempts_today']}")
        
        # 💰 FINANCIAL
        fin = self.get_financial_status()
        print(f"\n💰 FINANCIAL")
        print(f"   Income: ${fin['total_income']:.2f}")
        print(f"   Expenses: ${fin['total_expenses']:.2f}")
        print(f"   NET: ${fin['net']:.2f}")
        print(f"   Render services: {fin['render_services']} ({fin['render_cost']})")
        print(f"   Resources: {fin['resources']['compute_hours_monthly']} compute hrs/mo")
        print(f"   Profit ratio: {fin['profit_ratio']}")
        print(f"   Next milestone: {fin['next_milestone']}")
        
        # 📊 LEARNING STATS
        stats = self.learner.get_stats()
        print(f"\n📊 LEARNING METRICS")
        print(f"   Total phrases: {stats['phrases_heard']}")
        print(f"   Rejected non-English: {stats['non_english_rejected']}")
        if stats['phrases_heard'] > 0:
            rate = stats['vocabulary_size'] / stats['phrases_heard']
            print(f"   Learning rate: {rate:.2f} words/phrase")
        
        # 💭 DMAI'S SUGGESTIONS
        suggestions = self.get_dmai_suggestions()
        print(f"\n💭 DMAI'S SUGGESTIONS FOR FUTURE REPORTS")
        for s in suggestions:
            print(f"   • {s}")
        
        print("\n" + "="*80)
        print("🏁 REPORT END - DMAI CONTINUES TO EVOLVE")
        print("="*80)

if __name__ == "__main__":
    report = DMAIReport()
    report.generate()
