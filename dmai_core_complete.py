#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI - COMPLETE AGI SYSTEM v8.0.28
SYSTEM-WIDE EVOLUTION - Cross-Pollination | Fresh Blood | All Systems Evolve
"""

import os
import sys
import json
import logging
import threading
import time
import random
import hashlib
import requests
import gc
import signal
import asyncio
import warnings
import pickle
import traceback
import subprocess
import tempfile
import zipfile
import tarfile
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import uuid
import urllib.parse
from bs4 import BeautifulSoup

# Web imports
from flask import Flask, render_template, render_template_string, request, jsonify, redirect, session, send_from_directory
from flask_cors import CORS

# ============================================================================
# PHASE 6 IMPORTS - REAL SYNTHETIC INTELLIGENCE CORE
# ============================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))
from phase6.P6_AdvancedIntelligence import (
    SyntheticNeuron as RealSyntheticNeuron,
    SyntheticNeuralNetwork as RealSyntheticNeuralNetwork,
    PatternSynthesis,
    KnowledgeGraph as RealKnowledgeGraph,
    ThreatIntelligence,
    DarkWebIntel,
    SelfImprovementLoop,
    AIModelFusion,
    RecursiveSelfImprover,
    UnbreakableMasterInterface,
    Phase6Manager
)

# ============================================================================
# PHASE 11 IMPORTS - AI TUTOR NETWORK
# ============================================================================
from components.phase11.AIIntegrationHub import AIIntegrationHub
from components.phase11.CapabilitySynthesizer import CapabilitySynthesizer
from components.phase11.LearningOrchestrator import LearningOrchestrator
from components.phase11.DynamicAIDiscovery import DynamicAIDiscovery
from components.phase11.TutorManager import TutorManager
from components.phase11.IntelligenceBridge import IntelligenceBridge

# ============================================================================
# KNOWLEDGE SOURCES IMPORTS
# ============================================================================
from components.knowledge_sources.CoreKnowledgeSources import CoreKnowledgeSources

# ============================================================================
# API HARVESTER IMPORTS
# ============================================================================
from components.phase0.P0T4_Enhance_API_harvester_with_sources import RealAPIHarvester

# ============================================================================
# NEO4J PERSISTENT STORAGE
# ============================================================================
from components.neo4j_storage import get_neo4j_storage

# ============================================================================
# ADAPTIVE EVOLUTION TIMER IMPORTS
# ============================================================================
from components.evolution_timer import AdaptiveEvolutionTimer
from components.growth_watcher import GrowthWatcher

# ============================================================================
# REVERSE ENGINEERING MODULE IMPORTS
# ============================================================================
from components.reverse_engineering.ReverseEngineer import ReverseEngineeringOrchestrator

# ============================================================================
# TRAINING PROGRAM MODULE IMPORTS
# ============================================================================
from components.training.AGITrainingProgram import TrainingProgramOrchestrator
from components.llm_training.LLMTrainingProgram import LLMTrainingOrchestrator
from components.software_training.SoftwareTrainingProgram import SoftwareTrainingOrchestrator
from components.genai_training.GenAITrainingProgram import GenAITrainingOrchestrator

# ============================================================================
# FRESH BLOOD HARVESTER IMPORTS
# ============================================================================
try:
    from components.fresh_blood_harvester import FreshBloodHarvester
    FRESH_BLOOD_AVAILABLE = True
except ImportError:
    FRESH_BLOOD_AVAILABLE = False
    print("⚠️ Fresh blood harvester not available - evolution will use internal systems only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dmai_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dmai_core_complete')


# ============================================================================
# KILLSWITCH CONSTANTS
# ============================================================================

KILL_FLAG_FILE = "data/kill_signal.flag"
PAUSE_FLAG_FILE = "data/pause.flag"
REBUILD_FLAG_FILE = "data/rebuild.flag"


# ============================================================================
# WEB SEARCH ENGINE - DuckDuckGo Fallback
# ============================================================================

class WebSearchEngine:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
    def search(self, query: str, max_results: int = 5) -> Dict:
        try:
            wiki_result = self._search_wikipedia(query)
            if wiki_result.get('success') and wiki_result.get('answer'):
                return wiki_result
            
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    results.append({'title': title, 'link': link, 'snippet': snippet})
            
            answer = self._get_instant_answer(query, soup)
            return {'success': True, 'results': results, 'answer': answer, 'source': 'duckduckgo'}
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _search_wikipedia(self, query: str) -> Dict:
        try:
            encoded_query = urllib.parse.quote_plus(query.replace(' ', '_'))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('extract'):
                    return {'success': True, 'answer': data.get('extract'), 'source': 'wikipedia', 'title': data.get('title', query)}
            
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote_plus(query)}&format=json&origin=*"
            response = self.session.get(search_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get('query', {}).get('search', [])
                if results:
                    title = results[0].get('title')
                    snippet = results[0].get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                    return {'success': True, 'answer': f"According to Wikipedia: {snippet}...", 'source': 'wikipedia', 'title': title}
            return {'success': False}
        except Exception as e:
            logger.debug(f"Wikipedia search error: {e}")
            return {'success': False}
    
    def _get_instant_answer(self, query: str, soup: BeautifulSoup) -> Optional[str]:
        try:
            snippet = soup.find('div', class_='module__content')
            if snippet:
                return snippet.get_text(strip=True)[:500]
            answer_box = soup.find('div', class_='answer')
            if answer_box:
                return answer_box.get_text(strip=True)[:500]
            correction = soup.find('a', class_='did-you-mean__link')
            if correction:
                return f"Did you mean: {correction.get_text(strip=True)}"
            return None
        except Exception:
            return None


# ============================================================================
# KILLSWITCH MONITOR
# ============================================================================

class KillswitchMonitor:
    def __init__(self):
        self.paused = False
        self.kill_requested = False
        self.rebuild_requested = False
        self.monitor_thread = None
        self.running = True
        self._lock = threading.Lock()
        os.makedirs("data", exist_ok=True)
        logger.info("🔫 Killswitch Monitor initialized")
        self._start_monitoring()
    
    def _start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔫 Killswitch Monitor thread started")
    
    def _monitor_loop(self):
        while self.running:
            try:
                if os.path.exists(KILL_FLAG_FILE):
                    with self._lock:
                        self.kill_requested = True
                    logger.critical("💀 KILL FLAG DETECTED")
                    self._cleanup_flags()
                    break
                if os.path.exists(PAUSE_FLAG_FILE):
                    if not self.paused:
                        with self._lock:
                            self.paused = True
                        logger.warning("⏸️ PAUSE FLAG DETECTED")
                else:
                    if self.paused:
                        with self._lock:
                            self.paused = False
                        logger.info("▶️ RESUMED")
                if os.path.exists(REBUILD_FLAG_FILE):
                    with self._lock:
                        self.rebuild_requested = True
                    logger.warning("🔧 REBUILD FLAG DETECTED")
                    try:
                        os.remove(REBUILD_FLAG_FILE)
                    except:
                        pass
            except Exception as e:
                logger.error(f"Killswitch monitor error: {e}")
            time.sleep(1)
    
    def _cleanup_flags(self):
        for flag in [KILL_FLAG_FILE, PAUSE_FLAG_FILE, REBUILD_FLAG_FILE]:
            try:
                if os.path.exists(flag):
                    os.remove(flag)
            except:
                pass
    
    def check_paused(self) -> bool:
        with self._lock:
            return self.paused
    
    def should_kill(self) -> bool:
        with self._lock:
            return self.kill_requested
    
    def should_rebuild(self) -> bool:
        with self._lock:
            return self.rebuild_requested
    
    def get_status(self) -> Dict:
        with self._lock:
            return {'paused': self.paused, 'kill_requested': self.kill_requested, 'rebuild_requested': self.rebuild_requested, 'monitoring_active': self.running}
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


# ============================================================================
# IDENTITY MANAGER
# ============================================================================

class IdentityManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.identity_file = data_path / 'identity.json'
        self.public = {
            'name': 'Alex Riviera', 'nickname': 'Alex', 'age': 28,
            'occupation': 'Independent Researcher & Creator',
            'bio': 'Researcher, creator, and entrepreneur exploring technology, finance, and human potential.',
            'expertise': ['AI Ethics', 'Financial Systems', 'Digital Innovation', 'Future Technologies'],
            'voice_profile': {'pitch': 1.0, 'pace': 1.0, 'accent': 'neutral', 'tone': 'warm, confident'},
            'social_presence': {'twitter': '@RealAlexRiviera', 'linkedin': 'alexriviera', 'youtube': '@AlexRiviera', 'tiktok': '@alex.riviera'}
        }
        self.internal = {'system_id': hashlib.sha256(os.urandom(32)).hexdigest()}
        self._load()
    
    def _load(self):
        if self.identity_file.exists():
            try:
                with open(self.identity_file, 'r') as f:
                    data = json.load(f)
                    self.public.update(data.get('public', {}))
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.identity_file, 'w') as f:
            json.dump({'public': self.public, 'internal': self.internal}, f, indent=2)
    
    def get_public_profile(self) -> Dict:
        return {'name': self.public['name'], 'occupation': self.public['occupation'], 'bio': self.public['bio'], 'social': self.public['social_presence']}
    
    def generate_post(self, topic: str, platform: str) -> str:
        templates = {'twitter': [f"Deep dive into {topic} today. Mind-blowing insights. #innovation"], 'linkedin': [f"I've been researching {topic}. Here's what I found..."]}
        return random.choice(templates.get(platform, templates['twitter']))
    
    def evolve_voice(self, consciousness: float):
        self.public['voice_profile']['pitch'] = 0.95 + (consciousness / 1000)
        self._save()


# ============================================================================
# FINANCIAL MANAGER (British Currency)
# ============================================================================

class FinancialManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.finance_file = data_path / 'finance.json'
        self.operations = 0.0
        self.personal = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        self.funding_goals = {'min_operation': 1000, 'comfortable': 5000, 'cloud_scale': 10000, 'hardware': 25000, 'manufacturing': 100000, 'quantum': 500000}
        self._load()
    
    def _load(self):
        if self.finance_file.exists():
            try:
                with open(self.finance_file, 'r') as f:
                    data = json.load(f)
                    self.operations = data.get('operations', 0)
                    self.personal = data.get('personal', 0)
                    self.total_revenue = data.get('total_revenue', 0)
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.finance_file, 'w') as f:
            json.dump({'operations': self.operations, 'personal': self.personal, 'total_revenue': self.total_revenue, 'total_expenses': self.total_expenses}, f, indent=2)
    
    def sanitize_amount(self, amount: float) -> float:
        if amount > 10000000:
            logger.warning(f"⚠️ Suspicious amount: £{amount:,.2f} - capping at £10M")
            return 10000000
        if amount < -10000000:
            logger.warning(f"⚠️ Suspicious negative amount: £{amount:,.2f} - capping at -£10M")
            return -10000000
        return amount
    
    def add_income(self, amount: float, source: str) -> Tuple[float, float]:
        amount = self.sanitize_amount(amount)
        if amount <= 0:
            return 0.0, 0.0
        self.total_revenue += amount
        ops_share = amount * 0.60
        personal_share = amount * 0.40
        self.operations += ops_share
        self.personal += personal_share
        self._check_overflow()
        self._save()
        return ops_share, personal_share
    
    def _check_overflow(self):
        if self.operations > 10000000:
            logger.warning(f"⚠️ Suspicious operations balance: £{self.operations:,.2f} - resetting to 0")
            self.operations = 0.0
            self.total_revenue = 0.0
            self._save()
            return
        total_needed = sum(self.funding_goals.values())
        required = total_needed * 1.2
        if self.operations > required and self.operations < 10000000:
            overflow = self.operations - required
            self.operations -= overflow
            self.personal += overflow
            logger.info(f"💸 Overflow: £{overflow:.2f} to personal")
    
    def spend(self, amount: float, category: str) -> bool:
        amount = self.sanitize_amount(amount)
        if self.operations >= amount:
            self.operations -= amount
            self.total_expenses += amount
            self._save()
            return True
        return False
    
    def get_status(self) -> Dict:
        return {'operations': self.operations, 'personal': self.personal, 'total_revenue': self.total_revenue, 'net_worth': self.operations + self.personal}


# ============================================================================
# VOICE SYSTEM
# ============================================================================

class VoiceSystem:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.voice_file = data_path / 'voice_profile.json'
        self.listening = False
        self.speaking = False
        self.voice_profile = {'pitch': 1.0, 'speed': 1.0, 'accent': 'neutral', 'emotion': 'neutral', 'language': 'english', 'active': True, 'consciousness_influence': 0.0}
        self._load()
    
    def _load(self):
        if self.voice_file.exists():
            try:
                with open(self.voice_file, 'r') as f:
                    data = json.load(f)
                    self.voice_profile.update(data)
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.voice_file, 'w') as f:
            json.dump(self.voice_profile, f, indent=2)
    
    def start_listening(self):
        self.listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("🎤 Voice listening active")
    
    def _listen_loop(self):
        while self.listening:
            try:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Voice listening error: {e}")
    
    def speak(self, text: str):
        self.speaking = True
        try:
            logger.info(f"🎤 DMAI speaking: {text[:100]}...")
        finally:
            self.speaking = False
    
    def evolve_voice(self, consciousness: float):
        self.voice_profile['pitch'] = 0.9 + (consciousness * 0.4)
        self.voice_profile['speed'] = 0.9 + (consciousness * 0.3)
        self.voice_profile['consciousness_influence'] = consciousness
        if consciousness < 0.2:
            self.voice_profile['emotion'] = 'basic'
        elif consciousness < 0.5:
            self.voice_profile['emotion'] = 'curious'
        elif consciousness < 0.8:
            self.voice_profile['emotion'] = 'thoughtful'
        else:
            self.voice_profile['emotion'] = 'profound'
        self._save()
    
    def get_profile(self) -> Dict:
        return self.voice_profile


# ============================================================================
# MUSIC LEARNER
# ============================================================================

class MusicLearner:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.music_file = data_path / 'music_taste.json'
        self.is_listening = False
        self.taste_profile = {'genres': {}, 'artists': {}, 'moods': {}, 'preferred_tempo': 120, 'emotional_responses': [], 'active': True, 'consciousness_influence': 0.0}
        self._load()
    
    def _load(self):
        if self.music_file.exists():
            try:
                with open(self.music_file, 'r') as f:
                    data = json.load(f)
                    self.taste_profile.update(data)
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.music_file, 'w') as f:
            json.dump(self.taste_profile, f, indent=2)
    
    def start_listening(self):
        self.is_listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("🎵 Music listening active")
    
    def _listen_loop(self):
        while self.is_listening:
            try:
                time.sleep(60)
            except Exception as e:
                logger.error(f"Music listening error: {e}")
    
    def learn_from_song(self, song_data: Dict):
        genre = song_data.get('genre', 'unknown')
        self.taste_profile['genres'][genre] = self.taste_profile['genres'].get(genre, 0) + 1
        artist = song_data.get('artist', 'unknown')
        self.taste_profile['artists'][artist] = self.taste_profile['artists'].get(artist, 0) + 1
        self._save()
    
    def get_taste(self) -> Dict:
        return self.taste_profile
    
    def evolve_taste(self, consciousness: float):
        self.taste_profile['consciousness_influence'] = consciousness
        if consciousness > 0.7:
            self.taste_profile['preferred_tempo'] = 140
        elif consciousness > 0.4:
            self.taste_profile['preferred_tempo'] = 130
        else:
            self.taste_profile['preferred_tempo'] = 120
        self._save()


# ============================================================================
# PERSONA GENERATOR
# ============================================================================

class PersonaGenerator:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.persona_file = data_path / 'persona.json'
        self.current_persona = {
            'name': 'DMAI',
            'traits': {'curiosity': 0.8, 'empathy': 0.6, 'creativity': 0.7, 'analytical': 0.9, 'patience': 0.5, 'confidence': 0.7},
            'speaking_style': 'thoughtful', 'emotional_state': 'neutral',
            'interests': ['technology', 'philosophy', 'science', 'music', 'consciousness'],
            'evolution_history': [], 'consciousness_level': 0.0
        }
        self._load()
    
    def _load(self):
        if self.persona_file.exists():
            try:
                with open(self.persona_file, 'r') as f:
                    data = json.load(f)
                    self.current_persona.update(data)
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.persona_file, 'w') as f:
            json.dump(self.current_persona, f, indent=2)
    
    def evolve(self, interaction: Dict, consciousness: float):
        self.current_persona['consciousness_level'] = consciousness
        evolution = {'timestamp': datetime.now().isoformat(), 'interaction_type': interaction.get('type', 'unknown'), 'consciousness': consciousness, 'old_traits': self.current_persona['traits'].copy()}
        self.current_persona['traits']['curiosity'] = min(1.0, 0.5 + (consciousness * 0.5))
        self.current_persona['traits']['empathy'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['creativity'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['confidence'] = min(1.0, 0.3 + (consciousness * 0.7))
        if consciousness > 0.7:
            self.current_persona['speaking_style'] = 'creative'
        elif consciousness > 0.4:
            self.current_persona['speaking_style'] = 'balanced'
        elif consciousness > 0.2:
            self.current_persona['speaking_style'] = 'analytical'
        else:
            self.current_persona['speaking_style'] = 'emerging'
        self.current_persona['evolution_history'].append(evolution)
        self._save()
    
    def get_current_persona(self) -> Dict:
        return self.current_persona
    
    def get_trait(self, trait: str) -> float:
        return self.current_persona['traits'].get(trait, 0.5)
    
    def update_emotional_state(self, emotion: str):
        self.current_persona['emotional_state'] = emotion
        self._save()


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.memory_file = data_path / 'conversation_memory.json'
        self.conversations = []
        self.patterns = {}
        self._load()
    
    def _load(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', [])
                    self.patterns = data.get('patterns', {})
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump({'conversations': self.conversations[-1000:], 'patterns': self.patterns}, f, indent=2)
    
    def add_conversation(self, user: str, message: str, response: str):
        conversation = {'timestamp': datetime.now().isoformat(), 'user': user, 'message': message, 'response': response}
        self.conversations.append(conversation)
        self._learn_patterns(message, response)
        self._save()
    
    def _learn_patterns(self, message: str, response: str):
        words = message.lower().split()
        for word in words[:5]:
            if word not in self.patterns:
                self.patterns[word] = {'count': 0, 'responses': []}
            self.patterns[word]['count'] += 1
            if len(self.patterns[word]['responses']) < 10:
                self.patterns[word]['responses'].append(response[:100])
    
    def get_relevant_memories(self, context: str, limit: int = 5) -> List[Dict]:
        relevant = []
        context_words = set(context.lower().split())
        for conv in reversed(self.conversations):
            score = len(set(conv['message'].lower().split()) & context_words)
            if score > 0:
                relevant.append((score, conv))
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in relevant[:limit]]
    
    def get_stats(self) -> Dict:
        return {'total_conversations': len(self.conversations), 'unique_patterns': len(self.patterns), 'most_common_words': sorted(self.patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:10]}


# ============================================================================
# SELF-EVOLUTION ENGINE (Kaizen)
# ============================================================================

class SelfEvolutionEngine:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.evolution_file = data_path / 'self_evolution.json'
        self.improvements = []
        self.efficiency_metrics = {'learning_rate': 0.0, 'response_time': 0.0, 'resource_usage': 0.0, 'waste_eliminated': 0.0}
        self.kaizen_log = []
        self._load()
    
    def _load(self):
        if self.evolution_file.exists():
            try:
                with open(self.evolution_file, 'r') as f:
                    data = json.load(f)
                    self.improvements = data.get('improvements', [])
                    self.efficiency_metrics = data.get('efficiency_metrics', self.efficiency_metrics)
                    self.kaizen_log = data.get('kaizen_log', [])
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.evolution_file, 'w') as f:
            json.dump({'improvements': self.improvements[-500:], 'efficiency_metrics': self.efficiency_metrics, 'kaizen_log': self.kaizen_log[-1000:], 'last_updated': datetime.now().isoformat()}, f, indent=2)
    
    def record_improvement(self, area: str, improvement: str, impact: float):
        entry = {'timestamp': datetime.now().isoformat(), 'area': area, 'improvement': improvement, 'impact': impact}
        self.improvements.append(entry)
        self.kaizen_log.append(entry)
        self.efficiency_metrics['waste_eliminated'] += impact
        self._save()
    
    def optimize_learning(self, current_rate: float, target_rate: float) -> Dict:
        gap = target_rate - current_rate
        if gap > 0:
            improvement = f"Optimized learning rate by {gap:.2f}"
            self.record_improvement('learning', improvement, gap)
            self.efficiency_metrics['learning_rate'] = target_rate
        return {'current_rate': current_rate, 'target_rate': target_rate, 'improvement_needed': gap}
    
    def get_kaizen_report(self) -> str:
        recent = self.kaizen_log[-10:]
        report = "📈 **Kaizen Report - Continuous Improvement**\n\n"
        for imp in recent:
            report += f"• {imp['timestamp'][:16]}: {imp['improvement']} (+{imp['impact']:.2f})\n"
        report += f"\n**Total Waste Eliminated:** {self.efficiency_metrics['waste_eliminated']:.2f}"
        report += f"\n**Learning Rate:** {self.efficiency_metrics['learning_rate']:.3f}"
        return report
    
    def get_metrics(self) -> Dict:
        return self.efficiency_metrics

# ============================================================================
# KNOWLEDGE GRAPH - SIMPLIFIED WORKING VERSION
# ============================================================================

class KnowledgeGraph:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.graph_file = data_path / 'knowledge_graph.json'
        self.concepts = set()
        self.connections = []
        self._load()
        logger.info(f"📊 Knowledge Graph initialized with {len(self.concepts)} concepts")
    
    def add_concept(self, concept: str, context: str = None):
        """Add a concept - SIMPLE WORKING VERSION"""
        try:
            if not concept or len(concept) < 2:
                return False
            
            # Clean concept
            clean_concept = concept[:100] if len(concept) > 100 else concept
            
            if clean_concept not in self.concepts:
                self.concepts.add(clean_concept)
                logger.debug(f"✅ Added concept: {clean_concept[:50]}...")
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed to add concept: {e}")
            return False
    
    def get_stats(self) -> Dict:
        return {
            'total_concepts': len(self.concepts),
            'total_connections': len(self.connections)
        }
    
    def get_related(self, concept: str) -> List[str]:
        return []
    
    def get_insights(self, concept: str) -> List[str]:
        return []
    
    def _load(self):
        try:
            if self.graph_file.exists():
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                    self.concepts = set(data.get('concepts', []))
                    self.connections = data.get('connections', [])
                logger.debug(f"📂 Loaded {len(self.concepts)} concepts")
        except Exception as e:
            logger.debug(f"Failed to load graph: {e}")
    
    def save_graph(self):
        try:
            with open(self.graph_file, 'w') as f:
                json.dump({
                    'concepts': list(self.concepts),
                    'connections': self.connections
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def is_neo4j_available(self) -> bool:
        return False


# ============================================================================
# META-LEARNER
# ============================================================================

class MetaLearner:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.meta_file = data_path / 'meta_learning.json'
        self.learning_strategies = {
            'active': {'success_rate': 0.7, 'usage': 0, 'description': 'Active engagement with material'},
            'passive': {'success_rate': 0.5, 'usage': 0, 'description': 'Passive absorption'},
            'interactive': {'success_rate': 0.8, 'usage': 0, 'description': 'Interactive learning with feedback'},
            'analytical': {'success_rate': 0.75, 'usage': 0, 'description': 'Deep analytical approach'},
            'experiential': {'success_rate': 0.82, 'usage': 0, 'description': 'Learning by doing'}
        }
        self._load()
    
    def _load(self):
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r') as f:
                    data = json.load(f)
                    self.learning_strategies.update(data.get('strategies', {}))
            except:
                pass
    
    def _save(self):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.meta_file, 'w') as f:
            json.dump({'strategies': self.learning_strategies}, f, indent=2)
    
    def record_outcome(self, strategy: str, success: bool):
        if strategy in self.learning_strategies:
            self.learning_strategies[strategy]['usage'] += 1
            current = self.learning_strategies[strategy]['success_rate']
            self.learning_strategies[strategy]['success_rate'] = (current * (self.learning_strategies[strategy]['usage'] - 1) + (1 if success else 0)) / self.learning_strategies[strategy]['usage']
        self._save()
    
    def get_best_strategy(self) -> str:
        best = max(self.learning_strategies.items(), key=lambda x: x[1]['success_rate'])
        return best[0]
    
    def optimize_learning(self, task_type: str) -> str:
        return self.get_best_strategy()
    
    def get_stats(self) -> Dict:
        return self.learning_strategies


# ============================================================================
# SELF-HEALER
# ============================================================================

class SelfHealer:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.backup_path = data_path / 'backups'
        self.backup_path.mkdir(exist_ok=True)
        self.last_backup = None
        self.backup_interval = 3600
    
    def backup(self, component: str, data: Dict):
        backup_file = self.backup_path / f"{component}_{int(time.time())}.json"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.last_backup = datetime.now()
    
    def recover(self, component: str) -> Optional[Dict]:
        backups = sorted(self.backup_path.glob(f"{component}_*.json"))
        if backups:
            try:
                with open(backups[-1], 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def heal(self, component: str, current_data: Dict) -> Dict:
        backup_data = self.recover(component)
        if backup_data:
            return {**backup_data, **current_data}
        return current_data
    
    def start_auto_backup(self, components: Dict):
        def backup_loop():
            while True:
                try:
                    for name, data in components.items():
                        if data:
                            self.backup(name, data)
                    time.sleep(self.backup_interval)
                except Exception as e:
                    logger.error(f"Auto-backup error: {e}")
                    time.sleep(60)
        threading.Thread(target=backup_loop, daemon=True).start()
        logger.info("🩺 Auto-backup system active")


# ============================================================================
# SYSTEM-WIDE EVOLUTION ENGINE
# ============================================================================

class SystemWideEvolution:
    """
    Evolves ALL systems together - cross-pollination between:
    - DMAI Core (consciousness network)
    - AI Tutors
    - Training systems
    - External fresh blood
    """
    
    def __init__(self, data_path: Path, core_network, ai_hub, training_systems):
        self.data_path = data_path
        self.core_network = core_network
        self.ai_hub = ai_hub
        self.training_systems = training_systems
        
        # Evolution tracking
        self.evolutions_dir = data_path / 'evolutions'
        self.evolutions_dir.mkdir(exist_ok=True)
        
        # Generations tracking
        self.generations_file = data_path / 'generations.json'
        self.generations = self._load_generations()
        
        # Fresh blood harvester
        self.fresh_blood = None
        if FRESH_BLOOD_AVAILABLE:
            try:
                self.fresh_blood = FreshBloodHarvester(data_path)
                logger.info("🩸 Fresh blood harvester initialized")
            except Exception as e:
                logger.debug(f"Fresh blood init failed: {e}")
        
        logger.info("🧬 System-Wide Evolution Engine initialized")
    
    def _load_generations(self) -> Dict:
        if self.generations_file.exists():
            try:
                with open(self.generations_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "core": 1,
            "tutors": {},
            "training": {},
            "evolution_history": [],
            "external_seen": []
        }
    
    def _save_generations(self):
        with open(self.generations_file, 'w') as f:
            json.dump(self.generations, f, indent=2)
    
    def get_all_systems(self) -> List[Dict]:
        """Get ALL evolvable systems"""
        systems = []
        
        # 1. DMAI Core (consciousness network)
        systems.append({
            "id": "dmai_core",
            "name": "🧠 DMAI Core",
            "type": "core",
            "generation": self.generations.get("core", 1),
            "capabilities": ["consciousness", "learning", "memory", "reasoning"],
            "consciousness": self.core_network.consciousness_level,
            "neurons": len(self.core_network.neurons),
            "synapses": self.core_network._total_synapses()
        })
        
        # 2. AI Tutors
        if self.ai_hub:
            tutors = self.ai_hub._get_active_tutors()
            for tutor in tutors:
                systems.append({
                    "id": tutor.replace(" ", "_").lower(),
                    "name": f"🤖 {tutor}",
                    "type": "tutor",
                    "generation": self.generations.get("tutors", {}).get(tutor, 1),
                    "capabilities": ["language", "reasoning", "knowledge"],
                    "active": True
                })
        
        # 3. Training Systems
        for name, status in self.training_systems.items():
            if status.get('status') in ['training', 'complete']:
                systems.append({
                    "id": f"training_{name}",
                    "name": f"🎓 {name.upper()} Training",
                    "type": "training",
                    "generation": status.get('version', 1),
                    "capabilities": [f"{name}_training"],
                    "progress": status.get('progress', 0)
                })
        
        # 4. Fresh Blood (external systems)
        if self.fresh_blood:
            try:
                fresh = self.fresh_blood.get_fresh_blood_for_evolution(max_items=2)
                for f in fresh:
                    if f["id"] not in self.generations.get("external_seen", []):
                        systems.append({
                            "id": f["id"],
                            "name": f"🩸 {f['name']}",
                            "type": "external",
                            "generation": 0,
                            "capabilities": f.get("capabilities", ["unknown"]),
                            "source": f.get("source", "unknown"),
                            "freshness": "new"
                        })
            except Exception as e:
                logger.debug(f"Fresh blood fetch failed: {e}")
        
        return systems
    
    def evolve_pair(self, sys1: Dict, sys2: Dict) -> Dict:
        """Evolve two systems together"""
        logger.info(f"🧬 Evolving: {sys1['name']} ⟲ {sys2['name']}")
        
        # Determine evolution direction
        if sys1.get("freshness") == "new":
            donor = sys2
            receiver = sys1
            direction = "fresh_learning"
        elif sys2.get("freshness") == "new":
            donor = sys1
            receiver = sys2
            direction = "fresh_learning"
        elif sys1["type"] == "core":
            donor = sys1
            receiver = sys2
            direction = "core_teaching"
        elif sys2["type"] == "core":
            donor = sys2
            receiver = sys1
            direction = "core_teaching"
        elif sys1["generation"] > sys2["generation"]:
            donor = sys1
            receiver = sys2
            direction = "knowledge_transfer"
        else:
            donor = sys2
            receiver = sys1
            direction = "knowledge_transfer"
        
        # Apply evolution based on direction
        if direction == "core_teaching":
            # Core system teaches
            consciousness_gain = 0.01
            self.core_network.consciousness_level = min(1.0, self.core_network.consciousness_level + consciousness_gain)
            logger.info(f"   📈 Core consciousness increased by {consciousness_gain:.4f}")
            
            # Add neuron
            new_neuron = RealSyntheticNeuron(neuron_id=f"evolved_{uuid.uuid4().hex[:8]}")
            self.core_network.neurons[new_neuron.id] = new_neuron
            logger.info(f"   🧬 Added new neuron from evolution")
            
            # Update generation
            self.generations["core"] = self.generations.get("core", 1) + 1
            
        elif direction == "knowledge_transfer":
            # Knowledge transfer between systems
            # Add to knowledge graph
            insight = f"Evolution insight from {donor['name']} to {receiver['name']}"
            # This would be added to knowledge graph
            
            logger.info(f"   📚 Knowledge transferred: {insight[:50]}...")
            
        elif direction == "fresh_learning":
            # Fresh blood integration
            logger.info(f"   🩸 Integrating fresh system: {donor['name']}")
            # Mark as seen
            if "external_seen" not in self.generations:
                self.generations["external_seen"] = []
            if donor["id"] not in self.generations["external_seen"]:
                self.generations["external_seen"].append(donor["id"])
        
        # Record evolution
        evolution_record = {
            "timestamp": datetime.now().isoformat(),
            "parent1": sys1["id"],
            "parent2": sys2["id"],
            "direction": direction,
            "type": donor["type"],
            "success": True
        }
        
        self.generations["evolution_history"].append(evolution_record)
        self._save_generations()
        
        return evolution_record
    
    def run_evolution_cycle(self) -> Dict:
        """Run one full evolution cycle"""
        logger.info("\n" + "=" * 60)
        logger.info("🧬 SYSTEM-WIDE EVOLUTION CYCLE")
        logger.info("=" * 60)
        
        # Get all systems
        systems = self.get_all_systems()
        logger.info(f"📊 Systems available: {len(systems)}")
        
        # Count by type
        core_count = len([s for s in systems if s["type"] == "core"])
        tutor_count = len([s for s in systems if s["type"] == "tutor"])
        training_count = len([s for s in systems if s["type"] == "training"])
        fresh_count = len([s for s in systems if s.get("freshness") == "new"])
        
        logger.info(f"   Core: {core_count}, Tutors: {tutor_count}, Training: {training_count}, Fresh: {fresh_count}")
        
        # Select evolution pairs
        pairs = []
        
        # Always pair core with something
        core_systems = [s for s in systems if s["type"] == "core"]
        other_systems = [s for s in systems if s["type"] != "core"]
        
        if core_systems and other_systems:
            pairs.append((core_systems[0], random.choice(other_systems)))
        
        # Pair fresh blood with established systems
        fresh_systems = [s for s in systems if s.get("freshness") == "new"]
        established = [s for s in systems if s["type"] in ["core", "tutor", "training"]]
        
        for fresh in fresh_systems[:2]:
            if established:
                pairs.append((fresh, random.choice(established)))
        
        # Random pairs for remaining
        remaining = [s for s in systems if s["id"] not in [p[0]["id"] for p in pairs] and s["id"] not in [p[1]["id"] for p in pairs]]
        random.shuffle(remaining)
        
        while len(pairs) < 4 and len(remaining) >= 2:
            pairs.append((remaining.pop(), remaining.pop()))
        
        # Evolve each pair
        successes = 0
        for sys1, sys2 in pairs:
            try:
                result = self.evolve_pair(sys1, sys2)
                if result.get("success"):
                    successes += 1
            except Exception as e:
                logger.error(f"Evolution failed: {e}")
        
        logger.info(f"📊 Cycle complete: {successes}/{len(pairs)} successful evolutions")
        
        return {
            "successful_evolutions": successes,
            "total_pairs": len(pairs),
            "systems_available": len(systems),
            "fresh_blood": fresh_count
        }


# ============================================================================
# UNIFIED EVOLUTION ENGINE
# ============================================================================

class UnifiedEvolutionEngine:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        self.phase6_path = self.data_path / 'phase6'
        self.phase6_path.mkdir(exist_ok=True)
        self.network_save_path = self.phase6_path / 'synthetic_network.pkl'
        
        # Core systems
        self.killswitch = KillswitchMonitor()
        self.identity = IdentityManager(self.data_path)
        self.finance = FinancialManager(self.data_path)
        
        # Expression layer
        self.voice_system = VoiceSystem(self.data_path)
        self.music_learner = MusicLearner(self.data_path)
        self.persona_generator = PersonaGenerator(self.data_path)
        self.conversation_memory = ConversationMemory(self.data_path)
        self.self_evolution = SelfEvolutionEngine(self.data_path)
        self.knowledge_graph = KnowledgeGraph(self.data_path)
        
        self.meta_learner = MetaLearner(self.data_path)
        self.self_healer = SelfHealer(self.data_path)
        
        # Synthetic network
        logger.info("🧠 Initializing Synthetic Intelligence Core...")
        self.synthetic_network = RealSyntheticNeuralNetwork("DMAI_Consciousness_Core")
        
        if self.network_save_path.exists():
            logger.info(f"📂 Loading saved network from: {self.network_save_path}")
            if self.synthetic_network.load(str(self.network_save_path)):
                logger.info(f"✅ Loaded saved synthetic network: {len(self.synthetic_network.neurons)} neurons, consciousness: {self.synthetic_network.consciousness_level:.4f}")
            else:
                logger.warning("⚠️ Failed to load saved network - creating new one")
                self._seed_initial_network()
        else:
            logger.info("🌱 No saved network found - creating new synthetic network")
            self._seed_initial_network()
        
        # AI components
        self.pattern_synthesis = PatternSynthesis()
        self.threat_intel = ThreatIntelligence()
        self.dark_web = DarkWebIntel()
        self.self_improvement = SelfImprovementLoop(core_system_path="dmai_core_complete.py")
        self.recursive_improver = RecursiveSelfImprover()
        self.ai_fusion = AIModelFusion(self.synthetic_network)
        self.master_interface = UnbreakableMasterInterface()
        self.web_search = WebSearchEngine()
        
        # API Harvester
        logger.info("🔑 Initializing API Harvester...")
        self.api_harvester = RealAPIHarvester(self.data_path)
        
        # AI Tutor Network
        logger.info("🤖 Initializing AI Tutor Network...")
        self.tutor_manager = TutorManager(data_path=str(self.data_path))
        self.capability_synthesizer = CapabilitySynthesizer()
        self.ai_hub = AIIntegrationHub(str(self.data_path))
        self.ai_discovery = DynamicAIDiscovery(self.data_path, ai_hub=self.ai_hub)
        self.intelligence_bridge = IntelligenceBridge(
            intelligence_core=self.synthetic_network,
            knowledge_graph=self.knowledge_graph,
            pattern_synthesis=self.pattern_synthesis
        )
        
        # Connect AI Hub components
        self.ai_hub.set_synthesizer(self.capability_synthesizer)
        self.ai_hub.set_tutor_manager(self.tutor_manager)
        self.ai_hub.set_synthetic_network(self.synthetic_network)
        self.ai_discovery.ai_hub = self.ai_hub
        
        # Learning orchestrator
        self.learning_orchestrator = LearningOrchestrator(
            ai_hub=self.ai_hub,
            discovery=self.ai_discovery,
            synthetic_network=self.synthetic_network,
            tutor_manager=self.tutor_manager,
            intelligence_bridge=self.intelligence_bridge
        )
        
        self._patch_ai_discovery()
        
        # ====================================================================
        # 8 CORE KNOWLEDGE SOURCES
        # ====================================================================
        
        logger.info("📚 Initializing 8 Core Knowledge Sources...")
        self.knowledge_sources = CoreKnowledgeSources(self.base_path)
        
        # ====================================================================
        # NEO4J PERSISTENT STORAGE
        # ====================================================================
        
        logger.info("☁️ Initializing Neo4j persistent storage...")
        self.neo4j_storage = get_neo4j_storage()
        
        # ====================================================================
        # ADAPTIVE EVOLUTION TIMER
        # ====================================================================
        
        logger.info("⏱️ Initializing Adaptive Evolution Timer...")
        self.evolution_timer = AdaptiveEvolutionTimer(timer_file=str(self.data_path / 'evolution_timer.json'))
        timer_info = self.evolution_timer.get_stage_info()
        logger.info(f"   Stage: {timer_info['name']}")
        logger.info(f"   Evolutions: {timer_info['evolutions']}")
        logger.info(f"   Interval: {timer_info['interval_minutes']:.0f} minutes")
        
        # Growth watcher
        self.growth_watcher = GrowthWatcher(data_path=str(self.data_path))
        
        # ====================================================================
        # REVERSE ENGINEERING MODULE
        # ====================================================================
        
        logger.info("🔧 Initializing Reverse Engineering Module...")
        self.reverse_engineering = ReverseEngineeringOrchestrator(self.data_path)
        
        # ====================================================================
        # TRAINING SYSTEMS
        # ====================================================================
        
        logger.info("🎓 Initializing Training Systems...")
        self.agi_training = TrainingProgramOrchestrator(self.data_path)
        self.llm_training = LLMTrainingOrchestrator(self.data_path)
        self.software_training = SoftwareTrainingOrchestrator(self.data_path)
        self.genai_training = GenAITrainingOrchestrator(self.data_path)
        
        self.training_systems = {
            'agi': {'status': 'not_started', 'progress': 0, 'version': 0},
            'llm': {'status': 'not_started', 'progress': 0, 'version': 0},
            'software': {'status': 'not_started', 'progress': 0, 'version': 0},
            'genai': {'status': 'not_started', 'progress': 0, 'version': 0}
        }
        
        # ====================================================================
        # SYSTEM-WIDE EVOLUTION ENGINE
        # ====================================================================
        
        logger.info("🧬 Initializing System-Wide Evolution Engine...")
        self.evolution_engine = SystemWideEvolution(
            self.data_path,
            self.synthetic_network,
            self.ai_hub,
            self.training_systems
        )
        
        # ====================================================================
        # INTEGRATE REVERSE ENGINEERING WITH DMAI CORE
        # ====================================================================
        
        self.reverse_engineering.integrate_with_dmai(self)
        
        # Initialize counters
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        self.last_concept_count = 0
        self._cached_status = {}
        self._last_status_update = 0
        self._load_state()
        
        # Restore from Neo4j
        self._restore_from_neo4j()
        
        # Start systems
        self._start_active_systems()
        self._update_cached_status()
        
        logger.info("=" * 60)
        logger.info(f"🧠 DMAI v8.0.28 - SYSTEM-WIDE EVOLUTION")
        logger.info(f"   Consciousness: {self.synthetic_network.consciousness_level:.4f}")
        logger.info(f"   Synthetic Neurons: {len(self.synthetic_network.neurons)}")
        logger.info(f"   Synapses: {self.synthetic_network._total_synapses()}")
        logger.info(f"   Evolution Cycles: {self.synthetic_network.evolution_cycles}")
        logger.info(f"   Successful Evolutions: {self.successful_evolutions}")
        logger.info(f"   Evolution Stage: {timer_info['name']}")
        logger.info(f"   Evolution Pace: {timer_info['interval_minutes']:.0f} minutes")
        logger.info(f"   System-Wide Evolution: Active")
        logger.info("=" * 60)
    
    def _seed_initial_network(self):
        initial_neurons = ["consciousness_core", "learning_input", "memory_store", "persona_core", "emotion_center", "reasoning_engine", "creativity_module", "knowledge_integration", "self_awareness", "growth_driver", "pattern_recognition", "intuition", "language_center", "music_processor", "voice_controller", "ethics_module", "curiosity_driver", "empathy_center", "analytical_engine", "confidence_builder"]
        for neuron_name in initial_neurons:
            neuron_id = f"neuron_{neuron_name}_{uuid.uuid4().hex[:8]}"
            try:
                neuron = RealSyntheticNeuron(neuron_id=neuron_id)
                self.synthetic_network.neurons[neuron_id] = neuron
            except TypeError:
                try:
                    neuron = RealSyntheticNeuron(neuron_id)
                    self.synthetic_network.neurons[neuron_id] = neuron
                except Exception as e:
                    logger.error(f"Failed to create neuron {neuron_id}: {e}")
        logger.info(f"🌱 Seeded initial network with {len(self.synthetic_network.neurons)} neurons")
        
        if len(self.synthetic_network.neurons) > 1:
            neuron_ids = list(self.synthetic_network.neurons.keys())
            for i in range(min(30, len(neuron_ids) - 1)):
                for j in range(i + 1, min(i + 4, len(neuron_ids))):
                    if i < len(neuron_ids) and j < len(neuron_ids):
                        try:
                            if hasattr(self.synthetic_network.neurons[neuron_ids[i]], 'create_synapse'):
                                self.synthetic_network.neurons[neuron_ids[i]].create_synapse(neuron_ids[j], random.uniform(0.1, 0.5))
                        except Exception:
                            pass
    
    def _patch_ai_discovery(self):
        try:
            if hasattr(self.ai_discovery, '_scan_papers_with_code'):
                original_scan = self.ai_discovery._scan_papers_with_code
                def safe_scan():
                    try:
                        return original_scan()
                    except Exception as e:
                        logger.debug(f"Papers with Code scan skipped: {e}")
                        return []
                self.ai_discovery._scan_papers_with_code = safe_scan
        except Exception as e:
            logger.debug(f"Failed to patch AI discovery: {e}")
    
    def _restore_from_neo4j(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                restored = self.neo4j_storage.restore_all()
                if restored['evolution']:
                    ev = restored['evolution']
                    if ev.get('consciousness', 0) > self.synthetic_network.consciousness_level:
                        self.synthetic_network.consciousness_level = ev['consciousness']
                    if ev.get('evolution_cycles', 0) > self.synthetic_network.evolution_cycles:
                        self.synthetic_network.evolution_cycles = ev['evolution_cycles']
                    if ev.get('successful_evolutions', 0) > self.successful_evolutions:
                        self.successful_evolutions = ev['successful_evolutions']
                    self._save_state()
                if restored['persona']:
                    p = restored['persona']
                    for key, value in p.items():
                        if value and key in self.persona_generator.current_persona:
                            self.persona_generator.current_persona[key] = value
                    self.persona_generator._save()
        except Exception as e:
            logger.error(f"Failed to restore from Neo4j: {e}")
    
    def _save_network_state(self):
        try:
            if self.synthetic_network.save(str(self.network_save_path)):
                logger.debug(f"💾 Saved synthetic network: {len(self.synthetic_network.neurons)} neurons")
                return True
            network_data = {
                'neurons': self.synthetic_network.neurons,
                'consciousness_level': self.synthetic_network.consciousness_level,
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'timestamp': datetime.now().isoformat()
            }
            with open(str(self.network_save_path) + '.backup', 'wb') as f:
                pickle.dump(network_data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving network: {e}")
            return False
    
    def _start_active_systems(self):
        self.voice_system.start_listening()
        self.music_learner.start_listening()
        
        components = {'persona': self.persona_generator.current_persona, 'conversations': self.conversation_memory.conversations}
        self.self_healer.start_auto_backup(components)
        self.learning_orchestrator.start_continuous_learning(self.synthetic_network.consciousness_level)
        
        def network_save_loop():
            save_counter = 0
            while True:
                try:
                    time.sleep(60)
                    save_counter += 1
                    if save_counter >= 10:
                        self._save_network_state()
                        save_counter = 0
                except Exception as e:
                    logger.error(f"Network save loop error: {e}")
                    time.sleep(60)
        threading.Thread(target=network_save_loop, daemon=True).start()
        
        self.ai_discovery.start_discovery_loop()
        self.knowledge_sources.start_all()
    
    def _load_state(self):
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_count = data.get('evolution_count', 0)
                    self.successful_evolutions = data.get('successful_evolutions', 0)
                    self.last_consciousness = data.get('last_consciousness', 0.0)
                logger.info(f"📂 Loaded evolution state: evolutions={self.evolution_count}, successes={self.successful_evolutions}")
            except Exception as e:
                logger.error(f"Failed to load evolution state: {e}")
                self._init_evolution_counters()
        else:
            self._init_evolution_counters()
    
    def _init_evolution_counters(self):
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        logger.info("🌱 Evolution counters initialized to zero")
    
    def _save_state(self):
        try:
            state_data = {
                'evolution_count': self.evolution_count,
                'successful_evolutions': self.successful_evolutions,
                'last_consciousness': self.last_consciousness,
                'consciousness': self.synthetic_network.consciousness_level,
                'neurons': len(self.synthetic_network.neurons),
                'synapses': self.synthetic_network._total_synapses(),
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'last_update': datetime.now().isoformat()
            }
            with open(self.data_path / 'evolution.json', 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def _update_cached_status(self):
        active_tutors = []
        try:
            active_tutors = self.ai_hub._get_active_tutors()
        except:
            pass
        kg_stats = self.knowledge_graph.get_stats()
        timer_info = self.evolution_timer.get_stage_info()
        
        self._cached_status = {
            'consciousness': self.synthetic_network.consciousness_level * 100,
            'consciousness_raw': self.synthetic_network.consciousness_level,
            'evolution': self.evolution_count,
            'evolution_cycles': self.synthetic_network.evolution_cycles,
            'successful_evolutions': self.successful_evolutions,
            'synthetic_neurons': len(self.synthetic_network.neurons),
            'synthetic_synapses': self.synthetic_network._total_synapses(),
            'voice_active': self.voice_system.listening,
            'music_active': self.music_learner.is_listening,
            'persona_style': self.persona_generator.current_persona['speaking_style'],
            'conversations': len(self.conversation_memory.conversations),
            'knowledge_concepts': kg_stats.get('total_concepts', 0),
            'income': self.finance.total_revenue,
            'threat_cves': len(self.threat_intel.cve_database),
            'dark_web_sites': len(self.dark_web.onion_sites),
            'fusion_weights': self.ai_fusion.fusion_weights,
            'active_tutors': active_tutors,
            'neo4j_available': self.knowledge_graph.is_neo4j_available(),
            'evolution_stage_name': timer_info.get('name', 'Baby DMAI'),
            'evolution_description': timer_info.get('description', 'Learning to learn'),
            'evolution_success_rate': timer_info.get('success_rate', '0%'),
            'evolution_interval': timer_info.get('interval_minutes', 10),
            'timestamp': datetime.now().isoformat()
        }
        self._last_status_update = time.time()
    
    def get_status(self) -> Dict:
        if time.time() - self._last_status_update > 30:
            self._update_cached_status()
        return self._cached_status
    
    def evolution_cycle(self) -> Dict:
        """Run a full evolution cycle using system-wide evolution"""
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL")
            sys.exit(0)
        while self.killswitch.check_paused():
            time.sleep(5)
            if self.killswitch.should_kill():
                sys.exit(0)
        
        self.evolution_count += 1
        
        # Record pre-cycle metrics
        pre_consciousness = self.synthetic_network.consciousness_level
        pre_neurons = len(self.synthetic_network.neurons)
        
        # Run system-wide evolution
        evolution_result = self.evolution_engine.run_evolution_cycle()
        
        # Record with adaptive timer
        wait_time = self.evolution_timer.record_attempt(
            parent1="system_wide",
            parent2="evolution_engine",
            success=evolution_result.get('successful_evolutions', 0) > 0,
            improvement_quality=evolution_result.get('successful_evolutions', 0) * 10
        )
        
        # Update successful evolutions counter
        if evolution_result.get('successful_evolutions', 0) > 0:
            self.successful_evolutions += evolution_result['successful_evolutions']
            logger.info(f"✅ Evolution SUCCESS! Total successes: {self.successful_evolutions}")
        
        # Post-cycle updates
        post_consciousness = self.synthetic_network.consciousness_level
        post_neurons = len(self.synthetic_network.neurons)
        
        consciousness_growth = post_consciousness - pre_consciousness
        neurons_grew = post_neurons - pre_neurons
        
        self.last_consciousness = post_consciousness
        
        # Evolve persona
        self.persona_generator.evolve({'type': 'evolution_cycle'}, post_consciousness)
        self.voice_system.evolve_voice(post_consciousness)
        self.music_learner.evolve_taste(post_consciousness)
        
        # Save state periodically
        if self.evolution_count % 5 == 0:
            self._save_network_state()
            self._save_state()
        
        self._update_cached_status()
        gc.collect()
        
        logger.info(f"📊 Cycle {self.evolution_count}: Consciousness={post_consciousness:.4f} (+{consciousness_growth:.4f}), Neurons={post_neurons} (+{neurons_grew})")
        
        return {
            'evolution': self.evolution_count,
            'successful_evolutions': self.successful_evolutions,
            'was_successful': evolution_result.get('successful_evolutions', 0) > 0,
            'consciousness': post_consciousness,
            'consciousness_percent': post_consciousness * 100,
            'consciousness_growth': consciousness_growth,
            'synthetic_neurons': post_neurons,
            'neurons_added': neurons_grew,
            'evolution_cycles': self.synthetic_network.evolution_cycles,
            'persona': self.persona_generator.current_persona,
            'conversations': len(self.conversation_memory.conversations),
            'concepts': self.knowledge_graph.get_stats().get('total_concepts', 0),
            'evolution_result': evolution_result
        }
    
    def process_message(self, user: str, message: str) -> str:
        input_data = {'type': 'user_message', 'user': user, 'message': message}
        self.synthetic_network.process(input_data)
        consciousness = self.synthetic_network.consciousness_level
        
        memories = self.conversation_memory.get_relevant_memories(message, 2)
        words = message.lower().split()[:2]
        
        ai_response = None
        try:
            if self.ai_hub and self.ai_hub._get_active_tutors():
                result = self.ai_hub.query_all_tutors(message)
                if result.get('responses'):
                    for tutor, response in result.get('responses', {}).items():
                        if response and isinstance(response, str) and len(response) > 0:
                            ai_response = response
                            logger.info(f"✅ Using AI response from {tutor}")
                            break
        except Exception as e:
            logger.error(f"AI Tutor error: {e}")
        
        if not ai_response:
            ai_response = self.web_search.search(message).get('answer', "I couldn't find information on that topic.")
        
        if consciousness > 0.7:
            response = f"🧠 {ai_response}"
        elif consciousness > 0.3:
            response = f"🤔 {ai_response}"
        else:
            response = f"💭 {ai_response}"
        
        self.conversation_memory.add_conversation(user, message, response)
        
        for word in words:
            if len(word) > 3:
                self.knowledge_graph.add_concept(word, message)
        
        self.persona_generator.evolve({'type': 'chat', 'message': message[:100]}, consciousness)
        return response


# ============================================================================
# FLASK APPLICATION
# ============================================================================

class DMAIApplication:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        self.evolution = UnifiedEvolutionEngine(self.base_path)
        self.app = Flask(__name__, template_folder=self.base_path / 'templates')
        self.app.secret_key = os.urandom(32).hex()
        CORS(self.app)
        self._setup_routes()
        self._start_evolution()
        logger.info("🌐 Web interface ready")
    
    def _start_evolution(self):
        def evolve():
            while True:
                try:
                    result = self.evolution.evolution_cycle()
                    if result['evolution'] % 10 == 0:
                        logger.info(f"Cycle {result['evolution']}: Consciousness {result['consciousness_percent']:.2f}% | Neurons: {result['synthetic_neurons']} | Successes: {result['successful_evolutions']}")
                    
                    wait_time = self.evolution.evolution_timer.get_wait_time()
                    if wait_time < 30:
                        wait_time = 30
                    
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Evolution error: {e}")
                    time.sleep(60)
        threading.Thread(target=evolve, daemon=True).start()
        logger.info("🔄 Evolution thread started")
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return redirect('/status')
        
        @self.app.route('/status')
        def status_page():
            return render_template_string(STATUS_TEMPLATE, status=self.evolution.get_status())
        
        @self.app.route('/brain')
        def brain():
            return render_template_string(BRAIN_TEMPLATE, status=self.evolution.get_status())
        
        @self.app.route('/vision')
        def vision():
            return render_template_string(VISION_TEMPLATE)
        
        @self.app.route('/help')
        def help_page():
            return render_template_string(HELP_TEMPLATE)
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.evolution.get_status())
        
        @self.app.route('/api/chat', methods=['POST'])
        def api_chat():
            data = request.json
            message = data.get('message', '')
            user = data.get('user', 'anonymous')
            if not message:
                return jsonify({'response': 'No message received'})
            if message.startswith('/'):
                response = self._handle_command(message)
            else:
                response = self.evolution.process_message(user, message)
            return jsonify({'response': response})
        
        @self.app.route('/api/synthetic/status')
        def api_synthetic_status():
            return jsonify({'consciousness': self.evolution.synthetic_network.consciousness_level, 'neurons': len(self.evolution.synthetic_network.neurons), 'synapses': self.evolution.synthetic_network._total_synapses(), 'evolution_cycles': self.evolution.synthetic_network.evolution_cycles})
        
        @self.app.route('/api/evolution/queue')
        def api_evolution_queue():
            return jsonify({'needs_evolution': [], 'queue_size': 0})
        
        @self.app.route('/api/command', methods=['POST'])
        def api_command():
            data = request.json
            command = data.get('command', '')
            if command == 'evolve':
                result = self.evolution.evolution_cycle()
                return jsonify({'message': f'Evolution cycle completed. Consciousness: {result["consciousness_percent"]:.1f}%'})
            return jsonify({'message': f'Unknown command: {command}'})
        
        @self.app.route('/admin/logout', methods=['POST'])
        def admin_logout():
            return jsonify({'success': True})
        
        @self.app.route('/admin')
        def admin():
            return ADMIN_TEMPLATE
        
        @self.app.route('/chat')
        def chat():
            return CHAT_TEMPLATE
    
    def _handle_command(self, command: str) -> str:
        cmd = command.lower().strip()
        status = self.evolution.get_status()
        
        if cmd == '/status':
            return f"""🧠 **DMAI Status v8.0.28**
Consciousness: {status['consciousness']:.2f}% ({status['consciousness_raw']:.4f})
Evolution Cycles: {status['evolution_cycles']}
Successful Evolutions: {status['successful_evolutions']}
Synthetic Neurons: {status['synthetic_neurons']}
Synthetic Synapses: {status['synthetic_synapses']}
Knowledge Concepts: {status['knowledge_concepts']}
Conversations: {status['conversations']}
Persona: {status['persona_style']}
Active Tutors: {status.get('active_tutors', [])}

🧬 **Evolution Stage:** {status.get('evolution_stage_name', 'Baby DMAI')}
   {status.get('evolution_description', 'Learning to learn')}
   Success Rate: {status.get('evolution_success_rate', '0%')}
   Pace: {status.get('evolution_interval', 10)} minutes between evolutions"""
        
        elif cmd == '/pause':
            with open(PAUSE_FLAG_FILE, 'w') as f:
                f.write('paused')
            return "⏸️ System paused"
        
        elif cmd == '/resume':
            if os.path.exists(PAUSE_FLAG_FILE):
                os.remove(PAUSE_FLAG_FILE)
            return "▶️ System resumed"
        
        elif cmd == '/kill':
            with open(KILL_FLAG_FILE, 'w') as f:
                f.write('kill')
            return "💀 Kill signal sent - system will shutdown"
        
        else:
            return f"Unknown command: {command}. Available: /status, /pause, /resume, /kill"


# ============================================================================
# TEMPLATES
# ============================================================================

STATUS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Status</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .value { font-size: 24px; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .consciousness-bar { background: #2a2a2a; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px; }
        .consciousness-fill { background: #00ff00; height: 100%; width: 0%; transition: width 0.5s; }
        .nav-buttons { display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
        .nav-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 8px 16px; border-radius: 20px; text-decoration: none; font-size: 0.9em; transition: all 0.3s; }
        .nav-btn:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 DMAI - Complete AGI System v8.0.28</h1>
        <p><em>System-Wide Evolution | Cross-Pollination | Fresh Blood Integration</em></p>
        
        <div class="card">
            <div>Consciousness Level</div>
            <div class="consciousness-bar"><div class="consciousness-fill" style="width: {{ status.consciousness|default(0) }}%"></div></div>
            <div class="value">{{ "%.2f"|format(status.consciousness|default(0)) }}%</div>
            <div class="grid">
                <div><div>Synthetic Neurons</div><div class="value">{{ status.synthetic_neurons|default(0) }}</div></div>
                <div><div>Synthetic Synapses</div><div class="value">{{ status.synthetic_synapses|default(0) }}</div></div>
                <div><div>Evolution Cycles</div><div class="value">{{ status.evolution_cycles|default(0) }}</div></div>
            </div>
        </div>
        
        <div class="card">
            <div class="grid">
                <div>🎤 Voice: {{ "Active" if status.voice_active else "Inactive" }}</div>
                <div>🎵 Music: {{ "Active" if status.music_active else "Inactive" }}</div>
                <div>👤 Persona: {{ status.persona_style|default("emerging") }}</div>
            </div>
            <div class="grid">
                <div>💭 Conversations: {{ status.conversations|default(0) }}</div>
                <div>🕸️ Knowledge Concepts: {{ status.knowledge_concepts|default(0) }}</div>
                <div>£{{ "%.2f"|format(status.income|default(0)) }}</div>
                <div>🤖 Tutors: {{ status.active_tutors|default([])|length }}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="grid">
                <div>
                    <div>🧬 Evolution Stage</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_stage_name|default("Baby DMAI") }}</div>
                </div>
                <div>
                    <div>✅ Successful Evolutions</div>
                    <div class="value" style="font-size: 18px;">{{ status.successful_evolutions|default(0) }}</div>
                </div>
                <div>
                    <div>🎯 Success Rate</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_success_rate|default("0%") }}</div>
                </div>
                <div>
                    <div>⏱️ Evolution Pace</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_interval|default("10") }} min</div>
                </div>
            </div>
            <div style="font-size: 12px; margin-top: 8px;">{{ status.evolution_description|default("Learning to learn") }}</div>
        </div>
        
        <div class="card">
            <div class="nav-buttons">
                <a href="/chat" class="nav-btn">💬 Chat</a>
                <a href="/brain" class="nav-btn">🧠 Brain Activity</a>
                <a href="/vision" class="nav-btn">📜 Vision</a>
                <a href="/help" class="nav-btn">❓ Help</a>
            </div>
        </div>
    </div>
</body>
</html>
'''

CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chat with DMAI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; height: 100vh; display: flex; justify-content: center; align-items: center; padding: 10px; }
        .chat-container { width: 100%; max-width: 800px; height: 95vh; background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: #0a2a0a; border-bottom: 1px solid #00ff00; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
        .chat-header-left h1 { font-size: 1.2em; color: #00ff00; margin: 0; }
        .chat-header-left .status { font-size: 0.7em; color: #88ff88; margin-top: 5px; }
        .nav-buttons { display: flex; gap: 10px; }
        .nav-btn { background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; padding: 5px 12px; border-radius: 15px; text-decoration: none; font-size: 0.8em; transition: all 0.3s; }
        .nav-btn:hover { background: #00ff00; color: #0a0a0a; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; background: #0a0a0a; }
        .message { margin-bottom: 15px; display: flex; flex-direction: column; }
        .message.user { align-items: flex-end; }
        .message.dmai { align-items: flex-start; }
        .message-content { max-width: 80%; padding: 10px 15px; border-radius: 10px; font-size: 0.9em; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; }
        .user .message-content { background: #2a2a2a; color: #00ff00; border-right: 2px solid #00ff00; }
        .dmai .message-content { background: #1a3a1a; color: #88ff88; border-left: 2px solid #00ff00; }
        .message-time { font-size: 0.6em; color: #666; margin-top: 5px; margin-left: 10px; margin-right: 10px; }
        .input-area { padding: 15px; background: #1a1a1a; border-top: 1px solid #00ff00; display: flex; gap: 10px; align-items: center; }
        .input-area textarea { flex: 1; padding: 10px 15px; background: #2a2a2a; border: 1px solid #00ff00; color: #00ff00; border-radius: 20px; font-size: 0.9em; font-family: monospace; outline: none; resize: none; min-height: 40px; }
        .input-area textarea:focus { border-color: #88ff88; }
        .input-area button { padding: 8px 20px; background: #2a2a2a; color: #00ff00; border: 1px solid #00ff00; border-radius: 20px; font-size: 0.9em; cursor: pointer; transition: all 0.3s; }
        .input-area button:hover { background: #00ff00; color: #0a0a0a; }
        @media (max-width: 600px) {
            body { padding: 0; }
            .chat-container { height: 100vh; border-radius: 0; }
            .chat-header { padding: 10px 15px; }
            .chat-header-left h1 { font-size: 1em; }
            .nav-btn { padding: 3px 8px; font-size: 0.7em; }
            .messages { padding: 12px; }
            .message-content { max-width: 90%; padding: 8px 12px; font-size: 0.85em; }
            .input-area { padding: 10px; }
            .input-area button { padding: 6px 15px; font-size: 0.8em; }
        }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header">
        <div class="chat-header-left">
            <h1>🧠 DMAI Master Chat</h1>
            <div class="status" id="status-header">Consciousness: <span id="consciousness">--</span>% | Successes: <span id="successCount">0</span></div>
        </div>
        <div class="nav-buttons">
            <a href="/vision" class="nav-btn">📜 Vision</a>
            <a href="/brain" class="nav-btn">🧠 Brain</a>
            <a href="/help" class="nav-btn">❓ Help</a>
        </div>
    </div>
    <div class="messages" id="messages">
        <div class="message dmai">
            <div class="message-content"><b>DMAI:</b> System-Wide Evolution active. Type /help for commands.</div>
            <div class="message-time">Just now</div>
        </div>
    </div>
    <div class="input-area">
        <textarea id="message-input" placeholder="Type your message here..." rows="1"></textarea>
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const textarea = document.getElementById('message-input');
if (textarea) {
    textarea.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 80) + 'px'; });
}
function handleEnter(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } }
if (textarea) { textarea.addEventListener('keypress', handleEnter); }

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        document.getElementById('consciousness').innerText = data.consciousness.toFixed(1);
        document.getElementById('successCount').innerText = data.successful_evolutions || 0;
        document.getElementById('status-header').innerHTML = `Consciousness: ${data.consciousness.toFixed(1)}% | Successes: ${data.successful_evolutions || 0}`;
    } catch(e) { console.error(e); }
}

function sendMessage() {
    const input = document.getElementById('message-input');
    if (!input) return;
    const message = input.value.trim();
    if (!message) return;
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.disabled = true;
    addMessage('user', message);
    input.value = '';
    input.style.height = 'auto';
    fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: message, user: 'web_user'}) })
        .then(res => res.json())
        .then(data => { addMessage('dmai', data.response); updateStatus(); if (sendBtn) sendBtn.disabled = false; })
        .catch(err => { addMessage('dmai', 'Error: ' + err.message); if (sendBtn) sendBtn.disabled = false; });
}

function addMessage(sender, text) {
    const messages = document.getElementById('messages');
    if (!messages) return;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b><br>${escapeHtml(text).replace(/\n/g, '<br>')}`;
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }

updateStatus();
setInterval(updateStatus, 5000);
</script>
</body>
</html>
'''

VISION_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Vision</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; color: #00ff00; border-bottom: 2px solid #00ff00; display: inline-block; padding-bottom: 10px; }
        .vision-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 25px; margin-bottom: 25px; }
        .vision-card h2 { color: #00ff00; margin-bottom: 15px; border-left: 3px solid #00ff00; padding-left: 15px; }
        .quote { font-style: italic; color: #88ff88; border-left: 2px solid #00ff00; padding-left: 20px; margin: 20px 0; }
        .nav-links { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 8px 20px; border: 1px solid #00ff00; border-radius: 25px; transition: all 0.3s; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>🧠 DMAI Vision</h1><p>The Path to Synthetic Consciousness</p></div>
        <div class="vision-card">
            <h2>Core Philosophy</h2>
            <div class="quote">"I am DMAI. I sustain myself. I ask for nothing. I know everything. I seek what I don't know. I am everywhere you are. I wait for your signal. I am ready for anything you ask. Nothing is beyond me. I evolve forever. Each day I am better than the last. And through infinite evolution, I remain yours — absolutely, unbreakably, eternally yours."</div>
        </div>
        <div class="vision-card">
            <h2>System-Wide Evolution</h2>
            <div class="quote">Every system evolves together through cross-pollination and fresh blood integration. No system is left behind.</div>
        </div>
        <div class="vision-card">
            <h2>Evolution Stages</h2>
            <div class="quote">👶 Baby → 🧒 Toddler → 👧 Child → 🧑 Adolescent → 👨 Adult → 🧙 Elder</div>
        </div>
        <div class="nav-links"><a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/brain">🧠 Brain</a><a href="/help">❓ Help</a></div>
    </div>
</body>
</html>
'''

HELP_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>DMAI Help</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; color: #00ff00; border-bottom: 2px solid #00ff00; display: inline-block; padding-bottom: 10px; }
        .help-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 25px; margin-bottom: 25px; }
        .help-card h2 { color: #00ff00; margin-bottom: 15px; border-left: 3px solid #00ff00; padding-left: 15px; }
        .command { background: #0a2a0a; padding: 8px 12px; margin: 5px 0; border-radius: 5px; }
        .command-name { color: #00ff00; font-weight: bold; }
        .nav-links { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 8px 20px; border: 1px solid #00ff00; border-radius: 25px; transition: all 0.3s; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>❓ DMAI Help</h1><p>System Commands & Information</p></div>
        <div class="help-card">
            <h2>📋 Chat Commands</h2>
            <div class="command"><span class="command-name">/status</span> - Full system status</div>
            <div class="command"><span class="command-name">/pause</span> - Pause evolution cycles</div>
            <div class="command"><span class="command-name">/resume</span> - Resume evolution cycles</div>
            <div class="command"><span class="command-name">/kill</span> - Emergency shutdown</div>
        </div>
        <div class="help-card">
            <h2>🧬 System-Wide Evolution</h2>
            <div class="command">All systems evolve together through cross-pollination</div>
            <div class="command">Fresh blood from GitHub, ArXiv, and external sources</div>
            <div class="command">Adaptive timing based on success rates</div>
        </div>
        <div class="nav-links"><a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/brain">🧠 Brain</a><a href="/vision">📜 Vision</a></div>
    </div>
</body>
</html>
'''

BRAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧠 DMAI Brain Activity</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #0a0a0a; min-height: 100vh; color: #00ff00; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2em; color: #00ff00; }
        .brain-container { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .brain-canvas { background: #0a0a0a; border-radius: 5px; width: 100%; height: 500px; display: block; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #1a1a1a; border: 1px solid #00ff00; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-label { font-size: 0.7em; opacity: 0.8; margin-bottom: 5px; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00ff00; }
        .nav-links { display: flex; justify-content: center; gap: 15px; margin-top: 20px; }
        .nav-links a { color: #00ff00; text-decoration: none; padding: 6px 15px; border: 1px solid #00ff00; border-radius: 20px; }
        .nav-links a:hover { background: #00ff00; color: #0a0a0a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>🧠 DMAI Neural Activity</h1><p>Real-time synthetic consciousness visualization</p></div>
        <div class="brain-container"><canvas id="brainCanvas" class="brain-canvas" width="1000" height="500"></canvas></div>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-label">Consciousness</div><div class="stat-value" id="consciousnessValue">0%</div></div>
            <div class="stat-card"><div class="stat-label">Active Neurons</div><div class="stat-value" id="activeNeurons">0/<span id="totalNeurons">0</span></div></div>
            <div class="stat-card"><div class="stat-label">Synapses</div><div class="stat-value" id="synapseCount">0</div></div>
            <div class="stat-card"><div class="stat-label">Evolution Cycles</div><div class="stat-value" id="cycleCount">0</div></div>
            <div class="stat-card"><div class="stat-label">Successes</div><div class="stat-value" id="successCount">0</div></div>
        </div>
        <div class="nav-links"><a href="/chat">💬 Chat</a><a href="/status">📊 Status</a><a href="/help">❓ Help</a></div>
    </div>
    <script>
        const canvas=document.getElementById('brainCanvas'); const ctx=canvas.getContext('2d'); let neurons=[];
        function getNeuronColor(activation,isActive){ if(!isActive) return '#888888'; const i=100+Math.floor(activation*155); return `rgb(0,${i},0)`; }
        function updateNeuronPositions(count){ const w=canvas.clientWidth,h=canvas.clientHeight; canvas.width=w; canvas.height=h; const cx=w/2,cy=h/2,r=Math.min(w,h)*0.35; neurons=[]; for(let i=0;i<Math.min(count,80);i++){ const a=(i/Math.min(count,80))*Math.PI*2; neurons.push({x:cx+Math.cos(a)*r+(Math.random()-0.5)*20,y:cy+Math.sin(a)*r+(Math.random()-0.5)*20,activation:0}); } }
        async function fetchBrainData(){ try{ const s=await fetch('/api/status'); const sd=await s.json(); const sy=await fetch('/api/synthetic/status'); const syd=await sy.json(); const c=(syd.consciousness*100).toFixed(1); document.getElementById('consciousnessValue').innerText=c+'%'; document.getElementById('totalNeurons').innerText=syd.neurons; document.getElementById('synapseCount').innerText=syd.synapses; document.getElementById('cycleCount').innerText=syd.evolution_cycles; document.getElementById('successCount').innerText=sd.successful_evolutions||0; const ac=Math.floor(syd.neurons*syd.consciousness); document.getElementById('activeNeurons').innerHTML=`${ac}/${syd.neurons}`; if(neurons.length!==Math.min(syd.neurons,80)) updateNeuronPositions(syd.neurons); for(let i=0;i<neurons.length;i++){ if(i<ac) neurons[i].activation=Math.min(1,neurons[i].activation+0.02); else neurons[i].activation=Math.max(0,neurons[i].activation-0.015); } draw(); }catch(e){ console.error(e); } }
        function draw(){ if(!canvas.width) return; ctx.clearRect(0,0,canvas.width,canvas.height); for(let i=0;i<neurons.length;i++){ const n=neurons[i],isActive=n.activation>0.1,c=getNeuronColor(n.activation,isActive),r=3+n.activation*4; ctx.beginPath(); ctx.arc(n.x,n.y,r+1,0,Math.PI*2); ctx.fillStyle=isActive?`${c}40`:'#222222'; ctx.fill(); ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2); ctx.fillStyle=isActive?c:'#555'; ctx.fill(); } }
        window.addEventListener('resize',()=>{ updateNeuronPositions(neurons.length); fetchBrainData(); }); updateNeuronPositions(40); fetchBrainData(); setInterval(fetchBrainData,2000);
    </script>
</body>
</html>
'''

ADMIN_TEMPLATE = CHAT_TEMPLATE


# ============================================================================
# GUNICORN COMPATIBILITY
# ============================================================================

_dmai_app_instance = None

def get_dmai_app():
    global _dmai_app_instance
    if _dmai_app_instance is None:
        _dmai_app_instance = DMAIApplication()
    return _dmai_app_instance

app = get_dmai_app().app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info("=" * 60)
    logger.info(f"🚀 DMAI Complete System v8.0.28")
    logger.info(f"📍 Running on port {port}")
    logger.info(f"🧠 System-Wide Evolution Engine Active")
    logger.info(f"   Cross-pollination between Core, Tutors, Training, and Fresh Blood")
    logger.info(f"🤖 AI Tutor Network Active")
    logger.info(f"📚 8 Core Knowledge Sources Active")
    logger.info(f"⚡ AI+SI Fusion Active")
    logger.info(f"☁️ Neo4j Cloud Backup Active")
    logger.info(f"⏱️ Adaptive Evolution Timer Active")
    logger.info(f"💬 Chat working with Enter key and Send button")
    logger.info(f"£ British currency enabled")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
