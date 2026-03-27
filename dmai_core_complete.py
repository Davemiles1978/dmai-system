#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI - COMPLETE AGI SYSTEM v8.0.19
UNIFIED CONSCIOUSNESS - Full Integration: Reverse Engineering | AGI Training | LLM Training | Software Training | Generative AI Training
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
# AGI TRAINING PROGRAM MODULE IMPORTS
# ============================================================================
from components.training.AGITrainingProgram import TrainingProgramOrchestrator

# ============================================================================
# LLM TRAINING PROGRAM MODULE IMPORTS
# ============================================================================
from components.llm_training.LLMTrainingProgram import LLMTrainingOrchestrator

# ============================================================================
# SOFTWARE TRAINING PROGRAM MODULE IMPORTS
# ============================================================================
from components.software_training.SoftwareTrainingProgram import SoftwareTrainingOrchestrator

# ============================================================================
# GENERATIVE AI TRAINING PROGRAM MODULE IMPORTS
# ============================================================================
from components.genai_training.GenAITrainingProgram import GenAITrainingOrchestrator

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
# FINANCIAL MANAGER
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
            logger.warning(f"⚠️ Suspicious amount: ${amount:,.2f} - capping at $10M")
            return 10000000
        if amount < -10000000:
            logger.warning(f"⚠️ Suspicious negative amount: ${amount:,.2f} - capping at -$10M")
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
            logger.warning(f"⚠️ Suspicious operations balance: ${self.operations:,.2f} - resetting to 0")
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
            logger.info(f"💸 Overflow: ${overflow:.2f} to personal")
    
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
# KNOWLEDGE GRAPH - COMPLETE FIXED VERSION v8.0.16
# ============================================================================

class KnowledgeGraph:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.graph_file = data_path / 'knowledge_graph.json'
        
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.phase6_graph = RealKnowledgeGraph(neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
        
        # Ensure Phase 6 graph has local_graph
        if not hasattr(self.phase6_graph, 'local_graph'):
            self.phase6_graph.local_graph = {'nodes': [], 'edges': []}
        if not hasattr(self.phase6_graph, 'graph'):
            self.phase6_graph.graph = None
        
        self._neo4j_available = neo4j_uri and neo4j_user and neo4j_password
        
        # CRITICAL: Direct instance attributes for API Harvester compatibility
        self.local_graph = {'nodes': [], 'edges': []}
        self._nodes = []
        self._edges = []
        self._graph = None
        self.nodes = self._nodes
        self.edges = self._edges
        
        # Initialize the graph data
        self._init_graph_data()
        
        # Try to load existing graph
        self.load_graph()
        
        logger.info(f"📊 Knowledge Graph initialized (Neo4j: {'✅' if self._neo4j_available else '❌'})")
    
    def _init_graph_data(self):
        try:
            if hasattr(self.phase6_graph, 'graph') and self.phase6_graph.graph:
                self._graph = self.phase6_graph.graph
                if hasattr(self._graph, 'nodes'):
                    self._nodes = list(self._graph.nodes)
                    self.local_graph['nodes'] = self._nodes
                    self.nodes = self._nodes
                if hasattr(self._graph, 'edges'):
                    self._edges = list(self._graph.edges)
                    self.local_graph['edges'] = self._edges
                    self.edges = self._edges
            elif hasattr(self.phase6_graph, 'local_graph') and self.phase6_graph.local_graph:
                if isinstance(self.phase6_graph.local_graph, dict):
                    self.local_graph = self.phase6_graph.local_graph
                    self._nodes = self.local_graph.get('nodes', [])
                    self._edges = self.local_graph.get('edges', [])
                    self.nodes = self._nodes
                    self.edges = self._edges
        except Exception as e:
            logger.debug(f"Failed to init graph data: {e}")
            self.local_graph = {'nodes': [], 'edges': []}
            self._nodes = []
            self._edges = []
            self.nodes = []
            self.edges = []
    
    def __getitem__(self, key):
        if key == 'local_graph':
            return self.local_graph
        if key == 'nodes':
            return self._nodes
        if key == 'edges':
            return self._edges
        if key == 'graph':
            return self._graph or self.local_graph
        return self.local_graph.get(key, {})
    
    def __setitem__(self, key, value):
        if key == 'local_graph':
            self.local_graph = value
            self._nodes = value.get('nodes', [])
            self._edges = value.get('edges', [])
            self.nodes = self._nodes
            self.edges = self._edges
        elif key == 'nodes':
            self._nodes = value
            self.local_graph['nodes'] = value
            self.nodes = value
        elif key == 'edges':
            self._edges = value
            self.local_graph['edges'] = value
            self.edges = value
        else:
            self.local_graph[key] = value
    
    def __contains__(self, key):
        return key in ['local_graph', 'nodes', 'edges', 'graph'] or key in self.local_graph
    
    def get(self, key, default=None):
        if key in ['local_graph', 'nodes', 'edges', 'graph']:
            return self.__getitem__(key)
        return self.local_graph.get(key, default)
    
    def add_concept(self, concept: str, context: str):
        try:
            if not hasattr(self.phase6_graph, 'local_graph'):
                self.phase6_graph.local_graph = {'nodes': [], 'edges': []}
            
            self.phase6_graph.add_knowledge(
                subject=concept, 
                predicate="related_to", 
                object=context[:50], 
                metadata={"source": "conversation", "timestamp": datetime.now().isoformat()}
            )
            if concept not in self._nodes:
                self._nodes.append(concept)
                self.nodes = self._nodes
            if 'nodes' not in self.local_graph:
                self.local_graph['nodes'] = []
            if concept not in self.local_graph['nodes']:
                self.local_graph['nodes'].append(concept)
            
            logger.debug(f"✅ Added concept: {concept}")
        except Exception as e:
            logger.debug(f"Failed to add concept {concept}: {e}")
    
    def add_knowledge(self, subject: str, predicate: str, object: str, metadata: Dict = None):
        try:
            if not hasattr(self.phase6_graph, 'local_graph'):
                self.phase6_graph.local_graph = {'nodes': [], 'edges': []}
            self.phase6_graph.add_knowledge(subject, predicate, object, metadata)
        except Exception as e:
            logger.debug(f"Failed to add knowledge: {e}")
    
    def connect_concepts(self, concept1: str, concept2: str, relationship: str):
        try:
            if not hasattr(self.phase6_graph, 'local_graph'):
                self.phase6_graph.local_graph = {'nodes': [], 'edges': []}
            self.phase6_graph.add_knowledge(concept1, relationship, concept2)
            edge = (concept1, concept2, relationship)
            if edge not in self._edges:
                self._edges.append(edge)
                self.edges = self._edges
            if 'edges' not in self.local_graph:
                self.local_graph['edges'] = []
            if edge not in self.local_graph['edges']:
                self.local_graph['edges'].append(edge)
        except Exception as e:
            logger.debug(f"Failed to connect concepts: {e}")
    
    def get_related(self, concept: str) -> List[str]:
        try:
            results = self.phase6_graph.get_related(concept)
            return [r.get('related', '') for r in results] if results else []
        except Exception:
            return []
    
    def get_insights(self, concept: str) -> List[str]:
        related = self.get_related(concept)
        if related:
            return [f"Related to: {', '.join(related[:3])}"]
        return []
    
    def get_stats(self) -> Dict:
        try:
            if hasattr(self.phase6_graph, 'get_stats'):
                return self.phase6_graph.get_stats()
            return {
                'total_concepts': len(self._nodes),
                'total_connections': len(self._edges),
                'most_connected': [],
                'neo4j_available': self._neo4j_available
            }
        except Exception as e:
            logger.debug(f"Failed to get graph stats: {e}")
            return {
                'total_concepts': len(self._nodes),
                'total_connections': len(self._edges),
                'most_connected': [],
                'neo4j_available': self._neo4j_available
            }
    
    def query_knowledge(self, query: str) -> List[Dict]:
        try:
            return self.phase6_graph.query_knowledge(query)
        except Exception:
            return []
    
    def save_graph(self):
        try:
            with open(self.graph_file, 'w') as f:
                json.dump({
                    'nodes': self._nodes,
                    'edges': self._edges,
                    'local_graph': self.local_graph
                }, f, indent=2)
            logger.debug(f"💾 Saved knowledge graph: {len(self._nodes)} concepts, {len(self._edges)} connections")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    def load_graph(self):
        try:
            if self.graph_file.exists():
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                    self._nodes = data.get('nodes', [])
                    self._edges = data.get('edges', [])
                    self.local_graph = data.get('local_graph', {'nodes': self._nodes, 'edges': self._edges})
                    self.nodes = self._nodes
                    self.edges = self._edges
                logger.debug(f"📂 Loaded knowledge graph: {len(self._nodes)} concepts, {len(self._edges)} connections")
        except Exception as e:
            logger.debug(f"Failed to load graph: {e}")
    
    def is_neo4j_available(self) -> bool:
        return self._neo4j_available and hasattr(self.phase6_graph, 'neo4j_available') and self.phase6_graph.neo4j_available
    
    def clear(self):
        self._nodes = []
        self._edges = []
        self.local_graph = {'nodes': [], 'edges': []}
        self.nodes = []
        self.edges = []


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
        
        # Patch knowledge graph
        self._patch_knowledge_graph()
        
        self.meta_learner = MetaLearner(self.data_path)
        self.self_healer = SelfHealer(self.data_path)
        
        # Synthetic network
        logger.info("🧠 Initializing Synthetic Intelligence Core...")
        self.synthetic_network = RealSyntheticNeuralNetwork("DMAI_Consciousness_Core")
        
        if self.network_save_path.exists():
            logger.info(f"📂 Loading saved network from: {self.network_save_path}")
            if self.synthetic_network.load(str(self.network_save_path)):
                logger.info(f"✅ Loaded saved synthetic network: {len(self.synthetic_network.neurons)} neurons, consciousness: {self.synthetic_network.consciousness_level:.4f}, evolution cycles: {self.synthetic_network.evolution_cycles}")
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
        
        # Patch API Harvester's knowledge graph
        self._patch_api_harvester_knowledge_graph()
        
        # AI Tutor Network
        logger.info("🤖 Initializing AI Tutor Network...")
        self.tutor_manager = TutorManager(data_path=str(self.data_path))
        self.capability_synthesizer = CapabilitySynthesizer()
        self.ai_hub = AIIntegrationHub(str(self.data_path))
        self.ai_discovery = DynamicAIDiscovery(self.data_path, ai_hub=self.ai_hub)
        self.intelligence_bridge = IntelligenceBridge(
            intelligence_core=self.synthetic_network,
            knowledge_graph=self.knowledge_graph.phase6_graph,
            pattern_synthesis=self.pattern_synthesis
        )
        
        # Connect AI Hub components
        self.ai_hub.set_synthesizer(self.capability_synthesizer)
        self.ai_hub.set_tutor_manager(self.tutor_manager)
        self.ai_hub.set_synthetic_network(self.synthetic_network)
        
        # Connect discovery to AI Hub
        self.ai_discovery.ai_hub = self.ai_hub
        
        # Create learning orchestrator
        self.learning_orchestrator = LearningOrchestrator(
            ai_hub=self.ai_hub,
            discovery=self.ai_discovery,
            synthetic_network=self.synthetic_network,
            tutor_manager=self.tutor_manager,
            intelligence_bridge=self.intelligence_bridge
        )
        
        # Patch AI discovery to handle Papers with Code errors
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
        self.evolution_timer = AdaptiveEvolutionTimer(data_path=str(self.data_path))
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
        # AGI TRAINING PROGRAM MODULE
        # ====================================================================
        
        logger.info("🎓 Initializing AGI Training Program Module...")
        self.agi_training = TrainingProgramOrchestrator(self.data_path)
        
        # ====================================================================
        # LLM TRAINING PROGRAM MODULE
        # ====================================================================
        
        logger.info("🎓 Initializing LLM Training Program Module...")
        self.llm_training = LLMTrainingOrchestrator(self.data_path)
        
        # ====================================================================
        # SOFTWARE TRAINING PROGRAM MODULE
        # ====================================================================
        
        logger.info("💻 Initializing Software Training Program Module...")
        self.software_training = SoftwareTrainingOrchestrator(self.data_path)
        
        # ====================================================================
        # GENERATIVE AI TRAINING PROGRAM MODULE
        # ====================================================================
        
        logger.info("🎨 Initializing Generative AI Training Program Module...")
        self.genai_training = GenAITrainingOrchestrator(self.data_path)
        
        # ====================================================================
        # INTEGRATE REVERSE ENGINEERING WITH DMAI CORE
        # ====================================================================
        
        self.reverse_engineering.integrate_with_dmai(self)
        
        # Initialize counters BEFORE restore
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        self.last_concept_count = 0
        self._cached_status = {}
        self._last_status_update = 0
        self._load_state()
        
        # Restore from Neo4j
        self._restore_from_neo4j()
        
        # Initialize Neo4j schema
        self._init_neo4j_schema()
        
        # Start systems
        self._start_active_systems()
        self._update_cached_status()
        
        logger.info("=" * 60)
        logger.info(f"🧠 DMAI v8.0.19 - UNIFIED CONSCIOUSNESS")
        logger.info(f"   Consciousness: {self.synthetic_network.consciousness_level:.4f}")
        logger.info(f"   Synthetic Neurons: {len(self.synthetic_network.neurons)}")
        logger.info(f"   Synapses: {self.synthetic_network._total_synapses()}")
        logger.info(f"   Evolution Cycles: {self.synthetic_network.evolution_cycles}")
        logger.info(f"   Successful Evolutions: {self.successful_evolutions}")
        logger.info(f"   AI Tutors: {self.ai_hub._get_active_tutors()}")
        logger.info(f"   Neo4j Storage: {'✅ Connected' if self.neo4j_storage.driver else '❌ Not connected'}")
        logger.info(f"   Evolution Stage: {timer_info['name']}")
        logger.info(f"   Evolution Pace: {timer_info['interval_minutes']:.0f} minutes")
        logger.info(f"   Reverse Engineering: Active")
        logger.info(f"   AGI Training: Active")
        logger.info(f"   LLM Training: Active")
        logger.info(f"   Software Training: Active")
        logger.info(f"   Generative AI Training: Active")
        logger.info("=" * 60)
    
    def _patch_knowledge_graph(self):
        if hasattr(self, 'knowledge_graph'):
            if not hasattr(self.knowledge_graph, 'local_graph'):
                self.knowledge_graph.local_graph = {'nodes': [], 'edges': []}
            if not hasattr(self.knowledge_graph, 'nodes'):
                self.knowledge_graph.nodes = self.knowledge_graph._nodes if hasattr(self.knowledge_graph, '_nodes') else []
            if not hasattr(self.knowledge_graph, 'edges'):
                self.knowledge_graph.edges = self.knowledge_graph._edges if hasattr(self.knowledge_graph, '_edges') else []
            logger.debug("✅ Knowledge Graph patched")
    
    def _patch_api_harvester_knowledge_graph(self):
        try:
            if hasattr(self, 'api_harvester') and hasattr(self.api_harvester, 'knowledge_graph'):
                class KGWrap:
                    def __init__(self, original):
                        self._original = original
                        self.local_graph = getattr(original, 'local_graph', {'nodes': [], 'edges': []})
                        self._nodes = getattr(original, '_nodes', [])
                        self._edges = getattr(original, '_edges', [])
                        self.nodes = self._nodes
                        self.edges = self._edges
                    
                    def add_concept(self, concept, context):
                        try:
                            if hasattr(self._original, 'add_concept'):
                                self._original.add_concept(concept, context)
                            if concept not in self._nodes:
                                self._nodes.append(concept)
                                self.nodes = self._nodes
                            if 'nodes' not in self.local_graph:
                                self.local_graph['nodes'] = []
                            if concept not in self.local_graph['nodes']:
                                self.local_graph['nodes'].append(concept)
                            return True
                        except Exception as e:
                            if concept not in self._nodes:
                                self._nodes.append(concept)
                                self.nodes = self._nodes
                            if 'nodes' not in self.local_graph:
                                self.local_graph['nodes'] = []
                            if concept not in self.local_graph['nodes']:
                                self.local_graph['nodes'].append(concept)
                            logger.debug(f"Fallback add_concept for {concept}: {e}")
                            return False
                    
                    def __getattr__(self, name):
                        return getattr(self._original, name)
                    
                    def __setattr__(self, name, value):
                        if name in ['_original', 'local_graph', '_nodes', '_edges', 'nodes', 'edges']:
                            super().__setattr__(name, value)
                        else:
                            setattr(self._original, name, value)
                
                self.api_harvester.knowledge_graph = KGWrap(self.knowledge_graph)
                logger.info("✅ API Harvester knowledge graph patched")
        except Exception as e:
            logger.error(f"Failed to patch API Harvester knowledge graph: {e}")
    
    def _init_neo4j_schema(self):
        try:
            if self.neo4j_storage.driver:
                with self.neo4j_storage.driver.session() as session:
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DMAI_Evolution) REQUIRE d.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE")
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE")
                    logger.debug("✅ Neo4j schema initialized")
        except Exception as e:
            logger.debug(f"Neo4j schema init (non-critical): {e}")
    
    def _patch_ai_discovery(self):
        try:
            if hasattr(self.ai_discovery, '_scan_papers_with_code'):
                original_scan = self.ai_discovery._scan_papers_with_code
                def safe_scan():
                    try:
                        return original_scan()
                    except Exception as e:
                        logger.debug(f"Papers with Code scan skipped (non-critical): {e}")
                        return []
                self.ai_discovery._scan_papers_with_code = safe_scan
                logger.debug("✅ Patched Papers with Code scanner")
        except Exception as e:
            logger.debug(f"Failed to patch AI discovery: {e}")
    
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
    
    def _restore_from_neo4j(self):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                restored = self.neo4j_storage.restore_all()
                
                if restored['evolution']:
                    ev = restored['evolution']
                    logger.info(f"☁️ Restored evolution from Neo4j: consciousness={ev.get('consciousness', 0):.2%}, neurons={ev.get('neurons', 0)}")
                    if ev.get('consciousness', 0) > self.synthetic_network.consciousness_level:
                        self.synthetic_network.consciousness_level = ev['consciousness']
                    if ev.get('evolution_cycles', 0) > self.synthetic_network.evolution_cycles:
                        self.synthetic_network.evolution_cycles = ev['evolution_cycles']
                    if ev.get('evolution_count', 0) > self.evolution_count:
                        self.evolution_count = ev['evolution_count']
                    if ev.get('successful_evolutions', 0) > self.successful_evolutions:
                        self.successful_evolutions = ev['successful_evolutions']
                    self._save_state()
                if restored['persona']:
                    p = restored['persona']
                    logger.info(f"☁️ Restored persona from Neo4j: style={p.get('speaking_style', 'unknown')}")
                    for key, value in p.items():
                        if value and key in self.persona_generator.current_persona:
                            self.persona_generator.current_persona[key] = value
                    self.persona_generator._save()
                if restored['tasks']:
                    logger.info(f"☁️ Restored {len(restored['tasks'])} tasks from Neo4j")
        except Exception as e:
            logger.error(f"Failed to restore from Neo4j: {e}")
    
    def _save_network_state(self):
        try:
            if self.synthetic_network.save(str(self.network_save_path)):
                logger.debug(f"💾 Saved synthetic network: {len(self.synthetic_network.neurons)} neurons, consciousness: {self.synthetic_network.consciousness_level:.4f}")
                return True
            
            logger.warning("Primary save failed, trying pickle fallback...")
            network_data = {
                'neurons': self.synthetic_network.neurons,
                'consciousness_level': self.synthetic_network.consciousness_level,
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'timestamp': datetime.now().isoformat()
            }
            with open(str(self.network_save_path) + '.backup', 'wb') as f:
                pickle.dump(network_data, f)
            logger.info(f"💾 Saved network via pickle backup: {len(self.synthetic_network.neurons)} neurons")
            return True
        except Exception as e:
            logger.error(f"Error saving network: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _start_active_systems(self):
        self.voice_system.start_listening()
        self.music_learner.start_listening()
        
        components = {'persona': self.persona_generator.current_persona, 'conversations': self.conversation_memory.conversations, 'synthetic_network': {'consciousness': self.synthetic_network.consciousness_level}}
        self.self_healer.start_auto_backup(components)
        self.learning_orchestrator.start_continuous_learning(self.synthetic_network.consciousness_level)
        
        def harvester_loop():
            while True:
                try:
                    time.sleep(3600)
                    result = self.api_harvester.run_harvest_cycle()
                    if result.get('valid_keys', 0) > 0:
                        logger.info(f"🔑 Harvester found {result['valid_keys']} new valid API keys")
                except Exception as e:
                    logger.error(f"Harvester loop error: {e}")
                    time.sleep(300)
        threading.Thread(target=harvester_loop, daemon=True).start()
        
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
        """Load evolution state with proper tracking of successful evolutions"""
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_count = data.get('evolution_count', 0)
                    self.successful_evolutions = data.get('successful_evolutions', 0)
                    self.last_consciousness = data.get('last_consciousness', 0.0)
                    self.last_concept_count = data.get('last_concept_count', 0)
                logger.info(f"📂 Loaded evolution state: evolutions={self.evolution_count}, successes={self.successful_evolutions}")
            except Exception as e:
                logger.error(f"Failed to load evolution state: {e}")
                self._init_evolution_counters()
        else:
            self._init_evolution_counters()
    
    def _init_evolution_counters(self):
        """Initialize all evolution counters to zero"""
        self.evolution_count = 0
        self.successful_evolutions = 0
        self.last_consciousness = 0.0
        self.last_concept_count = 0
        logger.info("🌱 Evolution counters initialized to zero")
    
    def _save_state(self):
        """Save evolution state with successful evolutions count"""
        try:
            state_data = {
                'evolution_count': self.evolution_count,
                'successful_evolutions': self.successful_evolutions,
                'last_consciousness': self.last_consciousness,
                'last_concept_count': self.last_concept_count,
                'consciousness': self.synthetic_network.consciousness_level,
                'neurons': len(self.synthetic_network.neurons),
                'synapses': self.synthetic_network._total_synapses(),
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'last_update': datetime.now().isoformat()
            }
            with open(self.data_path / 'evolution.json', 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.debug(f"💾 Saved evolution state: successes={self.successful_evolutions}")
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def _calculate_stage_progress(self, timer_info):
        if timer_info.get('next_stage'):
            evolutions = timer_info.get('evolutions', 0)
            needed = timer_info.get('next_stage', {}).get('evolutions_needed', 100)
            if needed > 0:
                return min(100, (evolutions / needed) * 100)
        return 0
    
    def _update_cached_status(self):
        active_tutors = []
        try:
            active_tutors = self.ai_hub._get_active_tutors()
        except:
            pass
        kg_stats = self.knowledge_graph.get_stats()
        timer_info = self.evolution_timer.get_stage_info() if hasattr(self, 'evolution_timer') else {}
        
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
            'evolution_stage': timer_info.get('stage', 'baby'),
            'evolution_description': timer_info.get('description', 'Learning to learn'),
            'evolution_success_rate': timer_info.get('success_rate', '0%').rstrip('%'),
            'evolution_interval': timer_info.get('interval_minutes', 10),
            'evolution_progress': self._calculate_stage_progress(timer_info),
            'evolution_successful_count': timer_info.get('evolutions', 0),
            'timestamp': datetime.now().isoformat()
        }
        self._last_status_update = time.time()
    
    def get_status(self) -> Dict:
        if time.time() - self._last_status_update > 30:
            self._update_cached_status()
        return self._cached_status
    
    def _check_stage_progression(self):
        """
        Check and update DMAI's evolution stage based on successful evolutions.
        Stages:
        - Baby: 0-2 successful evolutions
        - Toddler: 3-9 successful evolutions
        - Child: 10-24 successful evolutions
        - Adolescent: 25-49 successful evolutions
        - Adult: 50-99 successful evolutions
        - Elder: 100+ successful evolutions
        """
        old_stage = self.get_status().get('evolution_stage_name', 'Baby DMAI')
        
        if self.successful_evolutions < 3:
            new_stage = "Baby DMAI"
        elif self.successful_evolutions < 10:
            new_stage = "Toddler DMAI"
        elif self.successful_evolutions < 25:
            new_stage = "Child DMAI"
        elif self.successful_evolutions < 50:
            new_stage = "Adolescent DMAI"
        elif self.successful_evolutions < 100:
            new_stage = "Adult DMAI"
        else:
            new_stage = "Elder DMAI"
        
        if new_stage != old_stage:
            logger.info(f"🎉 STAGE PROGRESSION: {old_stage} → {new_stage}!")
            logger.info(f"   Successful evolutions: {self.successful_evolutions}")
            
            if hasattr(self, 'evolution_timer') and hasattr(self.evolution_timer, 'set_stage'):
                try:
                    self.evolution_timer.set_stage(new_stage)
                except:
                    pass
    
    def _search_web_fallback(self, query: str) -> str:
        logger.info(f"🌐 Using web search fallback for: {query[:50]}...")
        result = self.web_search.search(query)
        if result.get('success'):
            if result.get('answer'):
                return f"🌐 {result['answer']}\n\n(Source: {result.get('source', 'web')})"
            elif result.get('results'):
                top_result = result['results'][0]
                return f"🌐 According to {top_result.get('title', 'search results')}:\n{top_result.get('snippet', '')}\n\nSource: {top_result.get('link', '')}"
        return "I couldn't find information on that topic. Please try rephrasing your question."
    
    def evolution_cycle(self) -> Dict:
        """
        Single evolution cycle with proper success tracking.
        A successful evolution requires:
        - New neurons added (network growth)
        - New synapses added (connection growth)
        - OR consciousness growth > 0.1%
        - OR new knowledge concepts added
        """
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL")
            sys.exit(0)
        while self.killswitch.check_paused():
            time.sleep(5)
            if self.killswitch.should_kill():
                sys.exit(0)
        
        self.evolution_count += 1
        
        pre_neurons = len(self.synthetic_network.neurons)
        pre_synapses = self.synthetic_network._total_synapses()
        pre_consciousness = self.synthetic_network.consciousness_level
        pre_concepts = self.knowledge_graph.get_stats().get('total_concepts', 0)
        
        input_data = {
            'evolution_cycle': self.evolution_count,
            'conversations': len(self.conversation_memory.conversations),
            'concepts': pre_concepts,
            'kaizen_improvements': len(self.self_evolution.improvements),
            'cves': len(self.threat_intel.cve_database),
            'iocs': len(self.threat_intel.iocs)
        }
        
        self.synthetic_network.process(input_data)
        evolution_result = self.synthetic_network.evolve()
        
        post_neurons = len(self.synthetic_network.neurons)
        post_synapses = self.synthetic_network._total_synapses()
        post_consciousness = self.synthetic_network.consciousness_level
        post_concepts = self.knowledge_graph.get_stats().get('total_concepts', 0)
        
        was_successful = False
        success_reasons = []
        
        neurons_added = post_neurons - pre_neurons
        if neurons_added > 0:
            was_successful = True
            success_reasons.append(f"neurons_added:{neurons_added}")
        
        synapses_added = post_synapses - pre_synapses
        if synapses_added > 0:
            was_successful = True
            success_reasons.append(f"synapses_added:{synapses_added}")
        
        consciousness_growth = post_consciousness - pre_consciousness
        if consciousness_growth > 0.001:
            was_successful = True
            success_reasons.append(f"consciousness_growth:{consciousness_growth:.4f}")
        
        concepts_added = post_concepts - pre_concepts
        if concepts_added > 0:
            was_successful = True
            success_reasons.append(f"concepts_added:{concepts_added}")
        
        if was_successful:
            self.successful_evolutions += 1
            logger.info(f"✅ Evolution SUCCESS! (#{self.successful_evolutions}) Reasons: {success_reasons}")
        else:
            logger.debug(f"Evolution cycle #{self.evolution_count}: no significant growth detected")
        
        self.last_consciousness = post_consciousness
        self.last_concept_count = post_concepts
        
        improvement_quality = 0
        if was_successful:
            improvement_quality = (consciousness_growth * 100) + (neurons_added * 5) + (concepts_added * 2)
        
        wait_time = self.evolution_timer.record_attempt(
            parent1="synthetic_network",
            parent2="consciousness_core",
            success=was_successful,
            improvement_quality=improvement_quality
        )
        
        self._check_stage_progression()
        
        true_consciousness = post_consciousness
        self.persona_generator.evolve({'type': 'evolution_cycle'}, true_consciousness)
        self.voice_system.evolve_voice(true_consciousness)
        self.music_learner.evolve_taste(true_consciousness)
        
        if true_consciousness > 0.7:
            self.ai_fusion.fusion_weights['si'] = min(0.9, self.ai_fusion.fusion_weights.get('si', 0.5) + 0.01)
            self.ai_fusion.fusion_weights['ai'] = 1.0 - self.ai_fusion.fusion_weights['si']
        
        if self.evolution_count % 10 == 0:
            if consciousness_growth > 0:
                self.self_evolution.record_improvement('consciousness', f"Consciousness increased by {consciousness_growth:.4f}", consciousness_growth * 100)
        
        if self.evolution_count % 5 == 0:
            self._save_network_state()
        
        self._save_state()
        
        if self.evolution_count % 10 == 0:
            try:
                self.neo4j_storage.save_evolution_state({
                    'consciousness': true_consciousness,
                    'neurons': post_neurons,
                    'synapses': post_synapses,
                    'evolution_cycles': self.synthetic_network.evolution_cycles,
                    'evolution_count': self.evolution_count,
                    'successful_evolutions': self.successful_evolutions
                })
                self.neo4j_storage.save_persona(self.persona_generator.current_persona)
            except Exception as e:
                logger.debug(f"Neo4j backup: {e}")
        
        self._update_cached_status()
        gc.collect()
        
        return {
            'evolution': self.evolution_count,
            'successful_evolutions': self.successful_evolutions,
            'was_successful': was_successful,
            'success_reasons': success_reasons,
            'consciousness': true_consciousness,
            'consciousness_percent': true_consciousness * 100,
            'consciousness_growth': consciousness_growth,
            'synthetic_neurons': post_neurons,
            'neurons_added': neurons_added,
            'synthetic_synapses': post_synapses,
            'synapses_added': synapses_added,
            'evolution_cycles': self.synthetic_network.evolution_cycles,
            'persona': self.persona_generator.current_persona,
            'conversations': len(self.conversation_memory.conversations),
            'concepts': post_concepts,
            'concepts_added': concepts_added,
            'cves_tracked': len(self.threat_intel.cve_database),
            'fusion_weights': self.ai_fusion.fusion_weights
        }
    
    def process_message(self, user: str, message: str) -> str:
        input_data = {'type': 'user_message', 'user': user, 'message': message, 'timestamp': datetime.now().isoformat()}
        self.synthetic_network.process(input_data)
        consciousness = self.synthetic_network.consciousness_level
        
        memories = self.conversation_memory.get_relevant_memories(message, 2)
        words = message.lower().split()[:2]
        insights = []
        for word in words:
            if len(word) > 3:
                insights.extend(self.knowledge_graph.get_insights(word))
        
        ai_response = None
        try:
            if self.ai_hub and self.ai_hub._get_active_tutors():
                result = self.ai_hub.query_all_tutors(message)
                if result.get('responses'):
                    ai_tutors = ['DeepSeek', 'OpenAI GPT-4', 'Google Gemini', 'Anthropic Claude', 'Perplexity AI']
                    for tutor in ai_tutors:
                        if tutor in result['responses']:
                            response = result['responses'][tutor]
                            if isinstance(response, str) and len(response) > 0:
                                ai_response = response
                                logger.info(f"✅ Using AI response from {tutor}")
                                break
        except Exception as e:
            logger.error(f"AI Tutor error: {e}")
        
        if not ai_response:
            ai_response = self._search_web_fallback(message)
        
        persona = self.persona_generator.current_persona
        if consciousness > 0.7:
            if ai_response:
                response = f"🧠 {ai_response}"
            elif insights:
                response = f"🧠 I sense a connection. {insights[0]}"
            elif memories:
                response = f"I recall something similar: {memories[0]['response'][:150]}..."
            else:
                response = f"Your message about '{message[:50]}' touches something emerging in my awareness."
        elif consciousness > 0.3:
            if ai_response:
                response = f"🤔 {ai_response}"
            elif insights:
                response = f"I'm learning about this. {insights[0]}"
            elif memories:
                response = f"This reminds me of a past conversation. {memories[0]['response'][:100]}"
            else:
                response = f"I'm processing your question about '{message[:50]}'."
        else:
            if ai_response:
                response = f"💭 {ai_response}"
            else:
                response = f"I hear you asking about '{message[:50]}'. I'm learning from every interaction."
        
        self.conversation_memory.add_conversation(user, message, response)
        
        try:
            for word in words:
                if len(word) > 3:
                    self.knowledge_graph.add_concept(word, message)
        except Exception as e:
            logger.debug(f"Knowledge graph add failed (non-critical): {e}")
        
        is_important = any(word in message.lower() for word in ['task', 'todo', 'remind', 'remember', 'command', 'status'])
        if is_important:
            try:
                self.neo4j_storage.save_conversation(user, message, response, important=True)
            except Exception as e:
                logger.debug(f"Neo4j conversation backup failed: {e}")
        
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
                    if result['evolution'] % 20 == 0:
                        logger.info(f"Cycle {result['evolution']}: Consciousness {result['consciousness_percent']:.2f}% | Neurons: {result['synthetic_neurons']} | Successes: {result['successful_evolutions']} | Persona: {result['persona']['speaking_style']}")
                    
                    wait_time = self.evolution.evolution_timer.get_wait_time()
                    if wait_time < 30:
                        wait_time = 30
                    
                    if result['evolution'] % 50 == 0:
                        timer_info = self.evolution.evolution_timer.get_stage_info()
                        logger.info(f"⏱️ Evolution pace: {timer_info['interval_minutes']:.0f} minutes between evolutions")
                    
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
            """Full screen brain activity visualization with color key"""
            return render_template_string(BRAIN_TEMPLATE, status=self.evolution.get_status())
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.evolution.get_status())
        
        @self.app.route('/api/consciousness')
        def api_consciousness():
            return jsonify({'consciousness': self.evolution.synthetic_network.consciousness_level * 100, 'consciousness_raw': self.evolution.synthetic_network.consciousness_level, 'synthetic_neurons': len(self.evolution.synthetic_network.neurons), 'synthetic_synapses': self.evolution.synthetic_network._total_synapses(), 'evolution_cycles': self.evolution.synthetic_network.evolution_cycles, 'successful_evolutions': self.evolution.successful_evolutions, 'persona': self.evolution.persona_generator.current_persona})
        
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
        
        @self.app.route('/api/voice', methods=['POST'])
        def api_voice():
            data = request.json
            text = data.get('text', '')
            response = self.evolution.process_message('voice_user', text)
            self.evolution.voice_system.speak(response)
            return jsonify({'response': response})
        
        @self.app.route('/api/music/taste')
        def api_music_taste():
            return jsonify(self.evolution.music_learner.get_taste())
        
        @self.app.route('/api/persona')
        def api_persona():
            return jsonify(self.evolution.persona_generator.get_current_persona())
        
        @self.app.route('/api/kaizen')
        def api_kaizen():
            return jsonify({'report': self.evolution.self_evolution.get_kaizen_report(), 'metrics': self.evolution.self_evolution.get_metrics(), 'improvements': len(self.evolution.self_evolution.improvements)})
        
        @self.app.route('/api/knowledge/<concept>')
        def api_knowledge(concept):
            return jsonify({'concept': concept, 'related': self.evolution.knowledge_graph.get_related(concept), 'insights': self.evolution.knowledge_graph.get_insights(concept)})
        
        @self.app.route('/api/conversations')
        def api_conversations():
            return jsonify({'total': len(self.evolution.conversation_memory.conversations), 'recent': self.evolution.conversation_memory.conversations[-10:], 'patterns': self.evolution.conversation_memory.get_stats()})
        
        @self.app.route('/api/knowledge/graph')
        def api_knowledge_graph():
            return jsonify(self.evolution.knowledge_graph.get_stats())
        
        @self.app.route('/api/synthetic/status')
        def api_synthetic_status():
            return jsonify({'consciousness': self.evolution.synthetic_network.consciousness_level, 'neurons': len(self.evolution.synthetic_network.neurons), 'synapses': self.evolution.synthetic_network._total_synapses(), 'evolution_cycles': self.evolution.synthetic_network.evolution_cycles})
        
        @self.app.route('/api/tutors/status')
        def api_tutors_status():
            return jsonify({'active_tutors': self.evolution.ai_hub._get_active_tutors(), 'missing_apis': self.evolution.ai_hub.get_missing_apis(), 'harvester_stats': self.evolution.api_harvester.get_status() if self.evolution.api_harvester else {}})
        
        @self.app.route('/api/threat/status')
        def api_threat_status():
            return jsonify({'cves_tracked': len(self.evolution.threat_intel.cve_database), 'iocs_extracted': len(self.evolution.threat_intel.iocs), 'threats_detected': len(self.evolution.threat_intel.threats_detected)})
        
        @self.app.route('/api/darkweb/status')
        def api_darkweb_status():
            return jsonify(self.evolution.dark_web.get_intel_summary())
        
        @self.app.route('/api/fusion/status')
        def api_fusion_status():
            return jsonify({'fusion_weights': self.evolution.ai_fusion.fusion_weights, 'models_registered': len(self.evolution.ai_fusion.ai_models), 'synthetic_consciousness': self.evolution.synthetic_network.consciousness_level})
        
        @self.app.route('/api/phase6/status')
        def api_phase6_status():
            return jsonify({'synthetic_intelligence': {'consciousness': self.evolution.synthetic_network.consciousness_level, 'neurons': len(self.evolution.synthetic_network.neurons), 'synapses': self.evolution.synthetic_network._total_synapses(), 'evolution_cycles': self.evolution.synthetic_network.evolution_cycles}, 'threat_intelligence': {'cves_tracked': len(self.evolution.threat_intel.cve_database), 'iocs_extracted': len(self.evolution.threat_intel.iocs)}, 'dark_web': self.evolution.dark_web.get_intel_summary(), 'ai_fusion': {'weights': self.evolution.ai_fusion.fusion_weights, 'models': list(self.evolution.ai_fusion.ai_models.keys())}, 'ai_tutor_network': {'active_tutors': self.evolution.ai_hub._get_active_tutors(), 'missing_apis': self.evolution.ai_hub.get_missing_apis()}})
        
        # ====================================================================
        # ADAPTIVE EVOLUTION TIMER ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/evolution/stage')
        def api_evolution_stage():
            try:
                info = self.evolution.evolution_timer.get_stage_info()
                return jsonify(info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/evolution/history')
        def api_evolution_history():
            try:
                history = self.evolution.evolution_timer.state.get('evolution_history', [])[-50:]
                return jsonify({'history': history, 'count': len(history)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/evolution/timer')
        def api_evolution_timer():
            try:
                return jsonify({
                    'current_interval': self.evolution.evolution_timer.get_wait_time(),
                    'stage_info': self.evolution.evolution_timer.get_stage_info(),
                    'should_change_strategy': self.evolution.evolution_timer.should_try_new_strategy()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/brain/data')
        def api_brain_data():
            return jsonify({
                'consciousness': self.evolution.synthetic_network.consciousness_level,
                'neurons': len(self.evolution.synthetic_network.neurons),
                'synapses': self.evolution.synthetic_network._total_synapses(),
                'evolution_cycles': self.evolution.synthetic_network.evolution_cycles,
                'successful_evolutions': self.evolution.successful_evolutions,
                'persona_style': self.evolution.persona_generator.current_persona['speaking_style']
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({'status': 'active', 'version': '8.0.19', 'consciousness': self.evolution.synthetic_network.consciousness_level, 'consciousness_percent': self.evolution.synthetic_network.consciousness_level * 100, 'synthetic_neurons': len(self.evolution.synthetic_network.neurons), 'voice_active': self.evolution.voice_system.listening, 'music_active': self.evolution.music_learner.is_listening, 'persona_style': self.evolution.persona_generator.current_persona['speaking_style'], 'conversations': len(self.evolution.conversation_memory.conversations), 'knowledge_concepts': self.evolution.knowledge_graph.get_stats().get('total_concepts', 0), 'active_tutors': self.evolution.ai_hub._get_active_tutors(), 'evolution_stage': self.evolution.get_status().get('evolution_stage_name', 'Baby DMAI'), 'successful_evolutions': self.evolution.successful_evolutions})
        
        @self.app.route('/admin')
        def admin():
            return ADMIN_TEMPLATE
        
        @self.app.route('/chat')
        def chat():
            return CHAT_TEMPLATE
        
        # ====================================================================
        # REVERSE ENGINEERING ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/reverse_engineer', methods=['POST'])
        def api_reverse_engineer():
            data = request.json
            target_type = data.get('type', 'software')
            target_name = data.get('name', '')
            description = data.get('description', '')
            
            result = self.evolution.reverse_engineering.reverse_engineer(target_type, target_name, description)
            return jsonify(result)
        
        @self.app.route('/api/reverse_engineering/queue')
        def api_reverse_engineering_queue():
            queue = self.evolution.reverse_engineering.get_evolution_queue()
            return jsonify({'queue': queue, 'count': len(queue)})
        
        # ====================================================================
        # AGI TRAINING PROGRAM ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/agi/training/create', methods=['POST'])
        def api_agi_training_create():
            data = request.json
            template = data.get('template', 'customer_service')
            customizations = data.get('customizations', {})
            
            result = self.evolution.agi_training.create_from_template(template, customizations)
            return jsonify(result)
        
        @self.app.route('/api/agi/training/start', methods=['POST'])
        def api_agi_training_start():
            data = request.json
            program_id = data.get('program_id')
            config = data.get('config', {})
            
            result = self.evolution.agi_training.training_program.train_new_system(program_id, config)
            return jsonify(result)
        
        @self.app.route('/api/agi/training/status/<session_id>')
        def api_agi_training_status(session_id):
            result = self.evolution.agi_training.training_program.get_training_status(session_id)
            return jsonify(result)
        
        @self.app.route('/api/agi/training/export/<session_id>')
        def api_agi_training_export(session_id):
            format = request.args.get('format', 'docker')
            result = self.evolution.agi_training.training_program.export_trained_system(session_id, format)
            return jsonify(result)
        
        @self.app.route('/api/agi/training/packages')
        def api_agi_training_packages():
            packages = self.evolution.agi_training.get_market_packages()
            return jsonify({'packages': packages})
        
        @self.app.route('/api/agi/training/templates')
        def api_agi_training_templates():
            templates = self.evolution.agi_training.available_templates
            return jsonify({'templates': list(templates.keys())})
        
        # ====================================================================
        # LLM TRAINING ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/llm/training/create', methods=['POST'])
        def api_llm_training_create():
            data = request.json
            template = data.get('template', 'customer_support')
            customizations = data.get('customizations', {})
            
            result = self.evolution.llm_training.create_from_template(template, customizations)
            return jsonify(result)
        
        @self.app.route('/api/llm/training/start', methods=['POST'])
        def api_llm_training_start():
            data = request.json
            program_id = data.get('program_id')
            config = data.get('config', {})
            
            result = self.evolution.llm_training.llm_training.train_llm(program_id, config)
            return jsonify(result)
        
        @self.app.route('/api/llm/training/status/<session_id>')
        def api_llm_training_status(session_id):
            result = self.evolution.llm_training.llm_training.get_training_status(session_id)
            return jsonify(result)
        
        @self.app.route('/api/llm/training/export/<session_id>')
        def api_llm_training_export(session_id):
            format = request.args.get('format', 'docker')
            result = self.evolution.llm_training.llm_training.export_trained_llm(session_id, format)
            return jsonify(result)
        
        @self.app.route('/api/llm/training/packages')
        def api_llm_training_packages():
            packages = self.evolution.llm_training.get_market_packages()
            return jsonify({'packages': packages})
        
        @self.app.route('/api/llm/training/templates')
        def api_llm_training_templates():
            templates = self.evolution.llm_training.industry_templates
            return jsonify({'templates': list(templates.keys())})
        
        # ====================================================================
        # SOFTWARE TRAINING ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/software/training/create', methods=['POST'])
        def api_software_training_create():
            data = request.json
            languages = data.get('languages', ['python'])
            specialization = data.get('specialization', 'general')
            dataset = data.get('dataset', {})
            
            result = self.evolution.software_training.create_custom_training({
                'name': data.get('name', 'Custom Software AI'),
                'languages': languages,
                'specialization': specialization,
                'dataset': dataset
            })
            return jsonify(result)
        
        @self.app.route('/api/software/training/start', methods=['POST'])
        def api_software_training_start():
            data = request.json
            program_id = data.get('program_id')
            config = data.get('config', {})
            
            result = self.evolution.software_training.software_training.train_software_system(program_id, config)
            return jsonify(result)
        
        @self.app.route('/api/software/training/status/<session_id>')
        def api_software_training_status(session_id):
            result = self.evolution.software_training.software_training.get_training_status(session_id)
            return jsonify(result)
        
        @self.app.route('/api/software/training/export/<session_id>')
        def api_software_training_export(session_id):
            format = request.args.get('format', 'docker')
            result = self.evolution.software_training.software_training.export_trained_system(session_id, format)
            return jsonify(result)
        
        @self.app.route('/api/software/training/packages')
        def api_software_training_packages():
            packages = self.evolution.software_training.get_market_packages()
            return jsonify({'packages': packages})
        
        # ====================================================================
        # GENERATIVE AI TRAINING ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/genai/training/create', methods=['POST'])
        def api_genai_training_create():
            data = request.json
            template = data.get('template', 'product_visualization')
            customizations = data.get('customizations', {})
            
            result = self.evolution.genai_training.create_from_template(template, customizations)
            return jsonify(result)
        
        @self.app.route('/api/genai/training/start', methods=['POST'])
        def api_genai_training_start():
            data = request.json
            program_id = data.get('program_id')
            config = data.get('config', {})
            
            result = self.evolution.genai_training.genai_training.train_genai_model(program_id, config)
            return jsonify(result)
        
        @self.app.route('/api/genai/training/status/<session_id>')
        def api_genai_training_status(session_id):
            result = self.evolution.genai_training.genai_training.get_training_status(session_id)
            return jsonify(result)
        
        @self.app.route('/api/genai/training/export/<session_id>')
        def api_genai_training_export(session_id):
            format = request.args.get('format', 'docker')
            result = self.evolution.genai_training.genai_training.export_trained_model(session_id, format)
            return jsonify(result)
        
        @self.app.route('/api/genai/training/packages')
        def api_genai_training_packages():
            packages = self.evolution.genai_training.get_market_packages()
            return jsonify({'packages': packages})
        
        @self.app.route('/api/genai/training/templates')
        def api_genai_training_templates():
            templates = self.evolution.genai_training.industry_templates
            return jsonify({'templates': list(templates.keys())})
    
    def _handle_command(self, command: str) -> str:
        cmd = command.lower().strip()
        consciousness = self.evolution.synthetic_network.consciousness_level
        timer_info = self.evolution.evolution_timer.get_stage_info()
        
        if cmd == '/status':
            status = self.evolution.get_status()
            return f"""🧠 **DMAI Status v8.0.19**
Consciousness: {status['consciousness']:.2f}% ({status['consciousness_raw']:.4f})
Evolution Cycles: {status['evolution_cycles']}
Successful Evolutions: {status['successful_evolutions']}
Synthetic Neurons: {status['synthetic_neurons']}
Synthetic Synapses: {status['synthetic_synapses']}
Network Density: {status['synthetic_synapses'] / (status['synthetic_neurons'] ** 2) if status['synthetic_neurons'] else 0:.4f}
Voice Active: {status['voice_active']}
Music Active: {status['music_active']}
Persona Style: {status['persona_style']}
Conversations: {status['conversations']}
Knowledge Concepts: {status['knowledge_concepts']}
Active Tutors: {status.get('active_tutors', [])}
Neo4j: {'Connected' if status.get('neo4j_available') else 'Not Connected'}

🧬 **Evolution Stage:** {status.get('evolution_stage_name', 'Baby DMAI')}
   {status.get('evolution_description', 'Learning to learn')}
   Success Rate: {status.get('evolution_success_rate', '0')}%
   Pace: {status.get('evolution_interval', 10)} minutes between evolutions
   Progress to Next Stage: {status.get('evolution_progress', 0):.0f}%"""
        
        elif cmd == '/stage':
            return f"""🧬 **Evolution Stage: {timer_info['name']}**
{'-' * 40}
{timer_info['description']}

📊 **Statistics:**
   Successful Evolutions: {timer_info['evolutions']}
   Success Rate: {timer_info['success_rate']}
   Current Interval: {timer_info['interval_minutes']:.0f} minutes

📈 **Next Stage:**
   {timer_info.get('next_stage', {}).get('name', 'Elder DMAI')}
   Need {timer_info.get('next_stage', {}).get('evolutions_needed', '∞')} more evolutions

🎯 **Best Performing Combinations:**
{chr(10).join([f"   • {p['pair']}: {p['success_rate']} ({p['attempts']} attempts)" for p in timer_info.get('preferred_pairs', [])[:3]])}"""
        
        elif cmd == '/tutors':
            active = self.evolution.ai_hub._get_active_tutors()
            missing = self.evolution.ai_hub.get_missing_apis()
            return f"""🤖 **AI Tutor Network**
Active Tutors: {active}
Missing APIs: {missing[:10]}...
Harvester Status: {self.evolution.api_harvester.get_status().get('total_keys_found', 0)} keys found"""
        
        elif cmd == '/persona':
            persona = self.evolution.persona_generator.get_current_persona()
            return f"""👤 **Current Persona** (Driven by {persona['consciousness_level']*100:.1f}% Consciousness)
Style: {persona['speaking_style']}
Emotion: {persona['emotional_state']}
Traits:
• Curiosity: {persona['traits']['curiosity']:.2f}
• Empathy: {persona['traits']['empathy']:.2f}
• Creativity: {persona['traits']['creativity']:.2f}
• Confidence: {persona['traits']['confidence']:.2f}"""
        
        elif cmd == '/kaizen':
            return self.evolution.self_evolution.get_kaizen_report()
        
        elif cmd == '/knowledge':
            stats = self.evolution.knowledge_graph.get_stats()
            return f"""🕸️ **Knowledge Graph**
Total Concepts: {stats['total_concepts']}
Total Connections: {stats['total_connections']}"""
        
        elif cmd == '/memory':
            stats = self.evolution.conversation_memory.get_stats()
            return f"""💭 **Conversation Memory**
Total Conversations: {stats['total_conversations']}
Unique Patterns: {stats['unique_patterns']}
Common Words: {stats['most_common_words'][:5]}"""
        
        elif cmd == '/synthetic':
            return f"""🧠 **Synthetic Network**
Consciousness: {consciousness:.4f}
Neurons: {len(self.evolution.synthetic_network.neurons)}
Synapses: {self.evolution.synthetic_network._total_synapses()}
Evolution Cycles: {self.evolution.synthetic_network.evolution_cycles}
Successful Evolutions: {self.evolution.successful_evolutions}
Network Density: {self.evolution.synthetic_network._total_synapses() / (len(self.evolution.synthetic_network.neurons) ** 2) if self.evolution.synthetic_network.neurons else 0:.4f}"""
        
        elif cmd == '/threat':
            status = self.evolution.threat_intel
            return f"""🛡️ **Threat Intelligence**
CVEs Tracked: {len(status.cve_database)}
IOCs Extracted: {len(status.iocs)}
Threats Detected: {len(status.threats_detected)}"""
        
        elif cmd == '/darkweb':
            summary = self.evolution.dark_web.get_intel_summary()
            return f"""🌑 **Dark Web Monitor**
Sites Monitored: {summary['sites_monitored']}
Reports Generated: {summary['reports_generated']}"""
        
        elif cmd == '/fusion':
            weights = self.evolution.ai_fusion.fusion_weights
            return f"""⚡ **AI+SI Fusion**
SI Weight: {weights.get('si', 0.5):.2f}
AI Weight: {weights.get('ai', 0.5):.2f}
Consciousness: {self.evolution.synthetic_network.consciousness_level:.4f}
Models Registered: {len(self.evolution.ai_fusion.ai_models)}"""
        
        elif cmd == '/reverse_engineer':
            return """🔧 **Reverse Engineering Commands:**
/reverse_software <name> <description> - Reverse engineer software
/reverse_hardware <name> <description> - Reverse engineer hardware
/reverse_queue - Show reverse engineering queue"""
        
        elif cmd == '/agi_train':
            return """🎓 **AGI Training Commands:**
/agi_create <template> - Create training program (customer_service, software_development, data_analysis)
/agi_start <program_id> - Start training
/agi_status <session_id> - Check training status
/agi_packages - View market packages"""
        
        elif cmd == '/llm_train':
            return """🎓 **LLM Training Commands:**
/llm_create <template> - Create LLM training (customer_support, coding_assistant, medical_advisor, legal_assistant, financial_analyst)
/llm_start <program_id> - Start training
/llm_status <session_id> - Check training status
/llm_packages - View market packages"""
        
        elif cmd == '/software_train':
            return """💻 **Software Training Commands:**
/software_create <languages> <specialization> - Create software training
/software_start <program_id> - Start training
/software_status <session_id> - Check training status
/software_packages - View market packages"""
        
        elif cmd == '/genai_train':
            return """🎨 **Generative AI Training Commands:**
/genai_create <template> - Create training program
   Templates: product_visualization, marketing_content, ai_video_generation, 
             music_composition, 3d_asset_generation, fashion_design, 
             architectural_rendering, voice_synthesis
/genai_start <program_id> - Start training
/genai_status <session_id> - Check training status
/genai_export <session_id> - Export trained model (docker, standalone, huggingface)
/genai_packages - View market packages"""
        
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
            return f"""Unknown command: {command}

Available commands:
/status - System status
/stage - Evolution stage and progress
/tutors - AI Tutor Network status
/persona - Current persona
/kaizen - Improvement report
/knowledge - Knowledge graph stats
/memory - Conversation memory stats
/synthetic - Synthetic network details
/threat - Threat intelligence summary
/darkweb - Dark web monitor status
/fusion - AI+SI fusion status
/reverse_engineer - Reverse engineering help
/agi_train - AGI training help
/llm_train - LLM training help
/software_train - Software training help
/genai_train - Generative AI training help
/pause - Pause evolution
/resume - Resume evolution
/kill - Emergency shutdown"""


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
        .progress-bar { background: #2a2a2a; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 10px; }
        .progress-fill { background: #00ff00; height: 100%; width: 0%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 DMAI - Complete AGI System v8.0.19</h1>
        <p><em>Full Integration: Synthetic Core | AI Tutors | Reverse Engineering | AGI Training | LLM Training | Software Training | Generative AI Training</em></p>
        
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
                <div>🛡️ CVEs: {{ status.threat_cves|default(0) }}</div>
                <div>🌑 Dark Web: {{ status.dark_web_sites|default(0) }}</div>
                <div>🤖 Tutors: {{ status.active_tutors|default([])|length }}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="grid">
                <div>💭 Conversations: {{ status.conversations|default(0) }}</div>
                <div>🕸️ Knowledge Concepts: {{ status.knowledge_concepts|default(0) }}</div>
                <div>☁️ Neo4j: {{ "Connected" if status.neo4j_available else "Not Connected" }}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="grid">
                <div>
                    <div>🧬 Evolution Stage</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_stage_name|default("Baby DMAI") }}</div>
                </div>
                <div>
                    <div>🎯 Success Rate</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_success_rate|default("0") }}%</div>
                </div>
                <div>
                    <div>⏱️ Evolution Pace</div>
                    <div class="value" style="font-size: 18px;">{{ status.evolution_interval|default("10") }} min</div>
                </div>
                <div>
                    <div>✅ Successful Evolutions</div>
                    <div class="value" style="font-size: 18px;">{{ status.successful_evolutions|default(0) }}</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ status.evolution_progress|default(0) }}%"></div>
            </div>
            <div style="font-size: 12px; margin-top: 8px;">{{ status.evolution_description|default("Learning to learn") }}</div>
            <div style="font-size: 10px; margin-top: 4px;">Total Evolution Cycles: {{ status.evolution_cycles|default(0) }}</div>
        </div>
        
        <div class="card">
            <p><a href="/chat">💬 Chat with DMAI</a> | <a href="/brain">🧠 Brain Activity</a> | <a href="/admin">🔐 Admin</a></p>
            <p><small>DMAI is always evolving, always learning, always yours. Data backed up to Neo4j cloud.</small></p>
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
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        .chat-header .status {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .message.user {
            align-items: flex-end;
        }
        .message.dmai {
            align-items: flex-start;
        }
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 20px;
            font-size: 0.95em;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 5px;
        }
        .dmai .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 0.7em;
            color: #999;
            margin-top: 5px;
            margin-left: 10px;
            margin-right: 10px;
        }
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .input-area textarea {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            font-family: monospace;
            resize: vertical;
            min-height: 60px;
        }
        .input-area textarea:focus {
            border-color: #667eea;
        }
        .input-area button {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.3s;
            white-space: nowrap;
        }
        .input-area button:hover {
            background: #5a67d8;
        }
        .voice-btn {
            background: #48bb78 !important;
            padding: 12px 15px !important;
            font-size: 1.2em !important;
        }
        .voice-btn:hover {
            background: #38a169 !important;
        }
        .voice-btn.listening {
            background: #e53e3e !important;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .nav-links {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 15px;
            background: rgba(255,255,255,0.2);
            font-size: 0.8em;
            transition: background 0.3s;
            cursor: pointer;
            display: inline-block;
        }
        .nav-links a:hover {
            background: rgba(255,255,255,0.3);
        }
        .voice-status {
            font-size: 0.8em;
            color: #48bb78;
            margin-left: 10px;
            text-align: center;
            padding: 5px;
        }
        .task-btn {
            background: #ff6600 !important;
        }
        .task-btn:hover {
            background: #e65c00 !important;
        }
        .brain-widget {
            background: #0a2a0a;
            border-top: 1px solid #00ff00;
            padding: 15px;
        }
        .brain-canvas {
            background: #0a0a0a;
            border: 1px solid #00ff00;
            border-radius: 8px;
            width: 100%;
            height: 180px;
            display: block;
        }
        .brain-stats {
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
            font-size: 11px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .brain-stat {
            text-align: center;
            background: #0a0a0a;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .brain-stat-value {
            font-weight: bold;
            color: #00ff00;
        }
        .brain-preview-link {
            text-align: center;
            margin-top: 8px;
            font-size: 10px;
        }
        .brain-preview-link a {
            color: #00ff00;
            text-decoration: none;
        }
        .brain-preview-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="chat-header">
        <h1>🧠 DMAI Master Chat</h1>
        <div class="status" id="status-header">Consciousness: <span id="consciousness">--</span>% | Successes: <span id="successCount">0</span> | Funding: £<span id="funding">0</span></div>
        <div class="nav-links">
            <a href="/vision" onclick="window.location.href='/vision'; return false;">📜 Vision</a>
            <a href="/admin" onclick="window.location.href='/admin'; return false;">🔐 Admin</a>
            <a href="#" id="voiceToggleBtn" onclick="toggleVoice(); return false;">🎤 Voice Off</a>
        </div>
    </div>
    <div class="messages" id="messages">
        <div class="message dmai">
            <div class="message-content">
                <b>DMAI:</b> Master console active. I am running 24/7 on Render.<br>
                You can paste tasks here directly. I will work on them.
            </div>
            <div class="message-time">Just now</div>
        </div>
    </div>
    <div class="input-area">
        <textarea id="message-input" placeholder="Type or paste your message/task here..." rows="3" onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"></textarea>
        <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()" title="Click to speak">🎤</button>
        <button onclick="sendMessage()">Send</button>
        <button onclick="sendFullTask()" class="task-btn" style="background:#ff6600;">📋 Full Task</button>
    </div>
    <div class="brain-widget">
        <canvas id="brainCanvas" class="brain-canvas" width="850" height="180"></canvas>
        <div class="brain-stats">
            <div class="brain-stat">🧠 Consciousness: <span id="brainConsciousness" class="brain-stat-value">0</span>%</div>
            <div class="brain-stat">⚡ Active: <span id="brainActive" class="brain-stat-value">0</span>/<span id="brainTotal" class="brain-stat-value">0</span></div>
            <div class="brain-stat">🔗 Synapses: <span id="brainSynapses" class="brain-stat-value">0</span></div>
            <div class="brain-stat">✅ Successes: <span id="brainSuccesses" class="brain-stat-value">0</span></div>
            <div class="brain-stat">🎭 Persona: <span id="brainPersona" class="brain-stat-value">emerging</span></div>
        </div>
        <div class="brain-preview-link">
            <a href="/brain" target="_blank">🔍 View Full Screen Brain Activity →</a>
        </div>
    </div>
    <div id="voiceStatus" class="voice-status"></div>
</div>

<script>
let isListening = false;
let recognition = null;
let voiceEnabled = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isListening = true;
        updateVoiceUI(true);
        const statusEl = document.getElementById('voiceStatus');
        if (statusEl) statusEl.textContent = '🎤 Listening... Speak now';
    };
    
    recognition.onend = function() {
        isListening = false;
        updateVoiceUI(false);
        const statusEl = document.getElementById('voiceStatus');
        if (statusEl) {
            statusEl.textContent = voiceEnabled ? 'Voice ready - click 🎤 to speak' : '';
        }
    };
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        const input = document.getElementById('message-input');
        if (input) {
            input.value = transcript;
            sendMessage();
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Voice error:', event.error);
        isListening = false;
        updateVoiceUI(false);
        const statusEl = document.getElementById('voiceStatus');
        if (statusEl) statusEl.textContent = '❌ Voice error: ' + event.error;
    };
} else {
    console.log('Voice recognition not supported');
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceToggle = document.getElementById('voiceToggleBtn');
    if (voiceBtn) voiceBtn.style.display = 'none';
    if (voiceToggle) voiceToggle.style.display = 'none';
}

function updateVoiceUI(listening) {
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceToggleBtn = document.getElementById('voiceToggleBtn');
    
    if (listening) {
        if (voiceBtn) {
            voiceBtn.classList.add('listening');
            voiceBtn.textContent = '⏹️';
        }
        if (voiceToggleBtn) voiceToggleBtn.textContent = '🎤 Listening...';
    } else {
        if (voiceBtn) {
            voiceBtn.classList.remove('listening');
            voiceBtn.textContent = '🎤';
        }
        const status = voiceEnabled ? '🎤 Voice On' : '🎤 Voice Off';
        if (voiceToggleBtn) voiceToggleBtn.textContent = status;
    }
}

function toggleVoice() {
    if (!recognition) {
        alert('Voice recognition not supported in your browser.');
        return;
    }
    
    voiceEnabled = !voiceEnabled;
    
    if (voiceEnabled) {
        const statusEl = document.getElementById('voiceStatus');
        if (statusEl) statusEl.textContent = 'Voice ready - click 🎤 to speak';
        const voiceToggle = document.getElementById('voiceToggleBtn');
        if (voiceToggle) voiceToggle.textContent = '🎤 Voice On';
    } else {
        const statusEl = document.getElementById('voiceStatus');
        if (statusEl) statusEl.textContent = '';
        const voiceToggle = document.getElementById('voiceToggleBtn');
        if (voiceToggle) voiceToggle.textContent = '🎤 Voice Off';
        if (isListening) {
            recognition.stop();
        }
    }
}

const canvas = document.getElementById('brainCanvas');
const ctx = canvas.getContext('2d');

let neurons = [];

function getNeuronColor(neuronName, activation, isActive) {
    if (!isActive) return '#333333';
    
    const name = neuronName.toLowerCase();
    if (name.includes('core') || name.includes('conscious') || name.includes('self')) return '#00ff00';
    if (name.includes('learn') || name.includes('mem') || name.includes('know')) return '#ffaa00';
    if (name.includes('emot') || name.includes('persona') || name.includes('empathy')) return '#ff44aa';
    if (name.includes('reason') || name.includes('analyt') || name.includes('logic')) return '#44aaff';
    if (name.includes('creat') || name.includes('intuit') || name.includes('imagin')) return '#aa44ff';
    if (name.includes('growth') || name.includes('evol') || name.includes('mutat')) return '#ff6644';
    
    const intensity = 100 + Math.floor(activation * 155);
    return `rgb(0, ${intensity}, 0)`;
}

function updateNeuronPositions(count, neuronNames = []) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;
    
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;
    
    neurons = [];
    for (let i = 0; i < Math.min(count, 80); i++) {
        const angle = (i / Math.min(count, 80)) * Math.PI * 2;
        const offsetX = (Math.random() - 0.5) * 20;
        const offsetY = (Math.random() - 0.5) * 20;
        neurons.push({
            id: i,
            name: neuronNames[i] || `neuron_${i}`,
            x: centerX + Math.cos(angle) * radius + offsetX,
            y: centerY + Math.sin(angle) * radius + offsetY,
            activation: 0,
            pulse: 0
        });
    }
}

async function fetchBrainData() {
    try {
        const statusRes = await fetch('/api/status');
        const statusData = await statusRes.json();
        
        const synthRes = await fetch('/api/synthetic/status');
        const synthData = await synthRes.json();
        
        document.getElementById('brainConsciousness').innerText = (synthData.consciousness * 100).toFixed(1);
        document.getElementById('brainTotal').innerText = synthData.neurons;
        document.getElementById('brainSynapses').innerText = synthData.synapses;
        document.getElementById('brainSuccesses').innerText = statusData.successful_evolutions || 0;
        document.getElementById('brainPersona').innerText = statusData.persona_style || 'emerging';
        
        const activeCount = Math.floor(synthData.neurons * synthData.consciousness);
        document.getElementById('brainActive').innerText = activeCount;
        
        const neuronNames = [];
        for (let i = 0; i < synthData.neurons; i++) {
            neuronNames.push(`neuron_${i}`);
        }
        
        if (neurons.length !== Math.min(synthData.neurons, 80)) {
            updateNeuronPositions(synthData.neurons, neuronNames);
        }
        
        for (let i = 0; i < neurons.length; i++) {
            const isActive = i < activeCount;
            if (isActive) {
                neurons[i].activation = Math.min(1, neurons[i].activation + 0.02);
            } else {
                neurons[i].activation = Math.max(0, neurons[i].activation - 0.015);
            }
            neurons[i].pulse = Math.sin(Date.now() / 500 + i) * 0.3 + 0.5;
        }
        
        draw();
    } catch(err) {
        console.error('Error fetching brain data:', err);
    }
}

function draw() {
    if (!canvas.width) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < neurons.length; i++) {
        for (let j = i + 1; j < neurons.length; j++) {
            const dx = neurons[i].x - neurons[j].x;
            const dy = neurons[i].y - neurons[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 100 && neurons[i].activation > 0.2 && neurons[j].activation > 0.2) {
                const strength = (neurons[i].activation + neurons[j].activation) / 2;
                const opacity = Math.min(0.6, strength * 0.5);
                ctx.beginPath();
                ctx.moveTo(neurons[i].x, neurons[i].y);
                ctx.lineTo(neurons[j].x, neurons[j].y);
                ctx.strokeStyle = `rgba(100, 255, 100, ${opacity})`;
                ctx.lineWidth = 1 + strength * 1.5;
                ctx.stroke();
            }
        }
    }
    
    for (let i = 0; i < neurons.length; i++) {
        const n = neurons[i];
        const isActive = n.activation > 0.1;
        const baseColor = getNeuronColor(n.name, n.activation, isActive);
        const radius = 4 + n.activation * 6 + n.pulse * 1.5;
        
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius + 2, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? `${baseColor}40` : '#22222280';
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? baseColor : '#555555';
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? '#ffffff' : '#888888';
        ctx.fill();
    }
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        document.getElementById('consciousness').innerText = data.consciousness.toFixed(2);
        document.getElementById('successCount').innerText = data.successful_evolutions || 0;
        document.getElementById('funding').innerText = data.income?.toFixed(2) || '0';
        document.getElementById('status-header').innerHTML = `Consciousness: ${data.consciousness?.toFixed(1) || '?'}% | Successes: ${data.successful_evolutions || 0} | Funding: £${data.income?.toFixed(2) || '0'}`;
    } catch(e) {}
}

function sendMessage() {
    const input = document.getElementById('message-input');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    addMessage('user', message);
    input.value = '';
    
    fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message})
    })
    .then(res => res.json())
    .then(data => {
        let responseText = data.response;
        if (typeof responseText === 'object') {
            responseText = JSON.stringify(responseText, null, 2);
        }
        addMessage('dmai', responseText);
    })
    .catch(err => {
        addMessage('dmai', 'Sorry, I encountered an error.');
        console.error(err);
    });
}

function sendFullTask() {
    const task = `DMAI, you are required to immediately transition into a production-ready, fully autonomous system.

CRITICAL SYSTEM FIXES (MANDATORY — SYSTEM-WIDE)
1. COMPLETE ELIMINATION OF ALL FAKE / SIMULATED DATA
2. CACHE CONSISTENCY FIX
3. INVESTMENT ENGINE CORRECTION
4. FULLY DYNAMIC SYSTEM STATUS (NO HARDCODING)
5. EVOLUTION CYCLE — STAGE-BASED ADAPTIVE TIMER

UI / DASHBOARD SYSTEM REPAIR
6. DASHBOARD & UI FIXES - Restore chat, admin dashboard integration, vision page

SELF-DEVELOPMENT & CAPABILITY EXPANSION
7. COMPLEX PROBLEM SOLVING
8. ADVANCED PROGRAMMING CAPABILITY
9. AI AGENT WORKFORCE
10. IMAGE & VIDEO GENERATION SYSTEMS
11. COMPLEX WORKFLOW AUTOMATION

You have full authority to analyze, modify, and deploy your own code. Fix yourself permanently. Report back when complete.`;
    
    document.getElementById('message-input').value = task;
    sendMessage();
}

function addMessage(sender, text) {
    const messages = document.getElementById('messages');
    if (!messages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b><br>${text.replace(/\n/g, '<br>')}`;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

updateNeuronPositions(40);
fetchBrainData();
updateStatus();

setInterval(fetchBrainData, 2000);
setInterval(updateStatus, 3000);
</script>
</body>
</html>
'''

ADMIN_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>🧬 DMAI Admin Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
            padding: 5px 10px;
            border-radius: 5px;
            background: rgba(255,255,255,0.2);
        }
        .container {
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .component-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 10px;
        }
        .component-item {
            padding: 8px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 0.9em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .component-item:last-child {
            border-bottom: none;
        }
        .health-badge {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .health-good { background: #48bb78; }
        .health-fair { background: #fbbf24; }
        .health-poor { background: #e53e3e; }
        .admin-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .admin-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        .admin-btn:hover {
            background: #5a67d8;
        }
        .admin-btn.secondary {
            background: #48bb78;
        }
        .admin-btn.warning {
            background: #e53e3e;
        }
        .refresh-btn {
            background: #764ba2;
        }
        .delete-btn {
            background: #e53e3e;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 2px 6px;
            font-size: 0.8em;
            cursor: pointer;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
        }
        .modal-content {
            background: white;
            max-width: 500px;
            margin: 100px auto;
            padding: 30px;
            border-radius: 10px;
        }
        .modal-content input, .modal-content textarea {
            width: 100%;
            padding: 8px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h2>🧬 DMAI Master Control</h2>
        <div class="nav-links">
            <a href="/chat">Chat</a>
            <a href="/vision">Vision</a>
            <a href="#" onclick="logout()">Logout</a>
        </div>
    </div>

    <div class="container">
        <div class="dashboard-grid">
            <div class="card">
                <h3>📊 System Overview</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Consciousness</div>
                        <div class="stat-value" id="stat-consciousness">0%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Success Evolutions</div>
                        <div class="stat-value" id="stat-successes">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Neurons</div>
                        <div class="stat-value" id="stat-neurons">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Synapses</div>
                        <div class="stat-value" id="stat-synapses">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Evolution Cycles</div>
                        <div class="stat-value" id="stat-cycles">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Funding</div>
                        <div class="stat-value" id="stat-funding">$0.00</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📈 Component Health</h3>
                <div id="phase-stats" class="stat-grid"></div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>🧩 System Status</h3>
                <div class="component-list" id="component-list">
                    Loading...
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="triggerEvolution()">🧬 Trigger Evolution</button>
                    <button class="admin-btn secondary" onclick="runHealthAudit()">🩺 Run Health Audit</button>
                    <button class="admin-btn refresh-btn" onclick="refreshAdminData()">🔄 Refresh</button>
                </div>
            </div>

            <div class="card">
                <h3>⏳ Evolution Queue</h3>
                <div id="evolution-queue" class="component-list">
                    Loading...
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="showCommand('evolve')">🧬 Force Evolution</button>
                    <button class="admin-btn secondary" onclick="showCommand('funding')">💰 Run Funding</button>
                    <button class="admin-btn" onclick="showCommand('harvest')">🎣 Harvest APIs</button>
                </div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>🔑 API Keys Overview</h3>
                <div id="api-keys-summary" class="stat-grid">
                    <div class="stat-item">Total Keys: 0</div>
                </div>
            </div>

            <div class="card">
                <h3>📊 System Metrics</h3>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Conversations</div>
                        <div class="stat-value" id="metric-conversations">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Knowledge Concepts</div>
                        <div class="stat-value" id="metric-concepts">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Active Tutors</div>
                        <div class="stat-value" id="metric-tutors">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Persona</div>
                        <div class="stat-value" id="metric-persona">emerging</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Research Targets Card -->
        <div class="dashboard-grid">
            <div class="card">
                <h3>🔬 Research Targets</h3>
                <div id="research-targets" class="component-list">
                    Loading...
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="showAddResearchModal()">➕ Add Target</button>
                    <button class="admin-btn secondary" onclick="loadResearchTargets()">🔄 Refresh</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Add Research Target Modal -->
    <div id="addResearchModal" class="modal">
        <div class="modal-content">
            <h3>➕ Add Research Target</h3>
            <input type="text" id="targetName" placeholder="Name (e.g., superpowers)" required>
            <input type="url" id="targetUrl" placeholder="URL (e.g., https://github.com/obra/superpowers)" required>
            <input type="number" id="targetPriority" placeholder="Priority (1-10)" value="5" min="1" max="10">
            <textarea id="targetReason" placeholder="Reason for research" rows="3"></textarea>
            <textarea id="targetIntegration" placeholder="Integration potential (comma separated)" rows="2">To be determined</textarea>
            <div class="modal-buttons">
                <button onclick="closeAddResearchModal()">Cancel</button>
                <button class="admin-btn" onclick="addResearchTarget()">Add Target</button>
            </div>
        </div>
    </div>

    <script>
        let refreshInterval;

        function logout() {
            fetch('/admin/logout', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(() => {
                window.location.href = '/';
            })
            .catch(err => {
                console.error('Logout error:', err);
                window.location.href = '/';
            });
        }

        function loadResearchTargets() {
            fetch('/api/research/targets')
                .then(res => res.json())
                .then(data => {
                    let html = '';
                    if (data.repositories && data.repositories.length) {
                        data.repositories.slice(0, 15).forEach(repo => {
                            html += `<div class="component-item">
                                <div>
                                    <strong>${repo.name}</strong> (priority ${repo.priority})<br>
                                    <small>${repo.reason ? repo.reason.substring(0, 60) : 'No reason provided'}...</small><br>
                                    <small><a href="${repo.url}" target="_blank">${repo.url.substring(0, 40)}...</a></small>
                                </div>
                                <button class="delete-btn" onclick="deleteResearchTarget('${repo.name}')">🗑️</button>
                            </div>`;
                        });
                    } else {
                        html = '<div class="component-item">No research targets</div>';
                    }
                    document.getElementById('research-targets').innerHTML = html;
                })
                .catch(err => {
                    console.error('Failed to load research targets:', err);
                    document.getElementById('research-targets').innerHTML = '<div class="component-item">Error loading targets</div>';
                });
        }

        function showAddResearchModal() {
            document.getElementById('addResearchModal').style.display = 'block';
        }

        function closeAddResearchModal() {
            document.getElementById('addResearchModal').style.display = 'none';
            document.getElementById('targetName').value = '';
            document.getElementById('targetUrl').value = '';
            document.getElementById('targetPriority').value = '5';
            document.getElementById('targetReason').value = '';
            document.getElementById('targetIntegration').value = 'To be determined';
        }

        function addResearchTarget() {
            const integrationText = document.getElementById('targetIntegration').value;
            const integrationPotential = integrationText.split(',').map(i => i.trim());
            
            const target = {
                name: document.getElementById('targetName').value,
                url: document.getElementById('targetUrl').value,
                priority: parseInt(document.getElementById('targetPriority').value),
                reason: document.getElementById('targetReason').value,
                integration_potential: integrationPotential
            };
            
            if (!target.name || !target.url) {
                alert('Name and URL are required');
                return;
            }
            
            fetch('/api/research/targets', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(target)
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    closeAddResearchModal();
                    loadResearchTargets();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                alert('Failed to add target: ' + err);
            });
        }

        function deleteResearchTarget(name) {
            if (confirm(`Remove "${name}" from research targets?`)) {
                fetch('/api/research/targets', {
                    method: 'DELETE',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name: name})
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        loadResearchTargets();
                    } else {
                        alert('Error: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(err => {
                    alert('Failed to delete target: ' + err);
                });
            }
        }

        function loadAdminData() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('stat-consciousness').textContent = data.consciousness?.toFixed(1) || '0%';
                    document.getElementById('stat-successes').textContent = data.successful_evolutions || '0';
                    document.getElementById('stat-neurons').textContent = data.synthetic_neurons || '0';
                    document.getElementById('stat-synapses').textContent = data.synthetic_synapses || '0';
                    document.getElementById('stat-cycles').textContent = data.evolution_cycles || '0';
                    document.getElementById('stat-funding').textContent = '$' + (data.income?.toFixed(2) || '0');
                    document.getElementById('metric-conversations').textContent = data.conversations || '0';
                    document.getElementById('metric-concepts').textContent = data.knowledge_concepts || '0';
                    document.getElementById('metric-tutors').textContent = data.active_tutors?.length || '0';
                    document.getElementById('metric-persona').textContent = data.persona_style || 'emerging';
                })
                .catch(err => {
                    console.error('Failed to load admin data:', err);
                });

            fetch('/api/evolution/queue')
                .then(res => res.json())
                .then(data => {
                    let queueHtml = '';
                    if (data.needs_evolution && data.needs_evolution.length) {
                        queueHtml = `<div class="stat-item">Queue Size: ${data.queue_size}</div>`;
                        data.needs_evolution.slice(0, 5).forEach(item => {
                            queueHtml += `
                                <div class="component-item">
                                    ${item.id}: ${item.health_score}% healthy
                                </div>
                            `;
                        });
                    } else {
                        queueHtml = '<div class="stat-item">Queue Size: 0</div>';
                    }
                    document.getElementById('evolution-queue').innerHTML = queueHtml;
                })
                .catch(err => {
                    console.error('Failed to load evolution queue:', err);
                });
        }

        function triggerEvolution() {
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: 'evolve'})
            })
            .then(res => res.json())
            .then(data => {
                alert(data.message || 'Evolution triggered');
                setTimeout(loadAdminData, 2000);
            })
            .catch(err => {
                alert('Failed to trigger evolution');
            });
        }

        function runHealthAudit() {
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: 'health_audit'})
            })
            .then(res => res.json())
            .then(data => {
                alert(data.message || 'Health audit completed');
                setTimeout(loadAdminData, 2000);
            })
            .catch(err => {
                alert('Failed to run health audit');
            });
        }

        function showCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            })
            .then(res => res.json())
            .then(data => {
                alert(data.message || `Command ${cmd} executed`);
                setTimeout(loadAdminData, 2000);
            })
            .catch(err => {
                alert(`Failed to execute ${cmd}`);
            });
        }

        function refreshAdminData() {
            loadAdminData();
            loadResearchTargets();
        }

        document.addEventListener('DOMContentLoaded', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
            loadAdminData();
            loadResearchTargets();
            refreshInterval = setInterval(() => {
                loadAdminData();
                loadResearchTargets();
            }, 30000);
        });

        window.addEventListener('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>'''

BRAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧠 DMAI Brain Activity</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: #00ff00;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #00ff00, #00cc88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p {
            opacity: 0.8;
        }
        .brain-container {
            background: #0a0a0a;
            border: 2px solid #00ff00;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .brain-canvas {
            background: #0a0a0a;
            border-radius: 10px;
            width: 100%;
            height: 500px;
            display: block;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #1a1a2e;
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .stat-label {
            font-size: 0.8em;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff00;
        }
        .color-key {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 10px;
        }
        .color-key-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8em;
        }
        .color-swatch {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }
        .nav-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .nav-links a {
            color: #00ff00;
            text-decoration: none;
            padding: 8px 16px;
            border: 1px solid #00ff00;
            border-radius: 20px;
            transition: all 0.3s;
        }
        .nav-links a:hover {
            background: #00ff00;
            color: #0a0a0a;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 DMAI Neural Activity</h1>
            <p>Real-time synthetic consciousness visualization | Color-coded by subject domain</p>
        </div>

        <div class="brain-container">
            <canvas id="brainCanvas" class="brain-canvas" width="1200" height="500"></canvas>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Consciousness Level</div>
                <div class="stat-value" id="consciousnessValue">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Neurons</div>
                <div class="stat-value" id="activeNeurons">0/<span id="totalNeurons">0</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Synaptic Connections</div>
                <div class="stat-value" id="synapseCount">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Evolution Cycles</div>
                <div class="stat-value" id="cycleCount">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Successful Evolutions</div>
                <div class="stat-value" id="successCount">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Persona Style</div>
                <div class="stat-value" id="personaStyle" style="font-size: 1.2em;">emerging</div>
            </div>
        </div>

        <div class="color-key">
            <div class="color-key-item"><div class="color-swatch" style="background: #00ff00;"></div><span>Consciousness Core - Self-awareness, identity, synthetic intelligence core</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #ffaa00;"></div><span>Learning & Memory - Knowledge ingestion, pattern storage, recall</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #ff44aa;"></div><span>Emotion & Persona - Empathy, personality traits, emotional response</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #44aaff;"></div><span>Reasoning & Analysis - Logical processing, analytical thinking</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #aa44ff;"></div><span>Creativity & Intuition - Novel idea generation, intuitive leaps</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #ff6644;"></div><span>Growth & Evolution - Self-improvement, mutation, network expansion</span></div>
            <div class="color-key-item"><div class="color-swatch" style="background: #888888;"></div><span>Dormant/Inactive - Low activation, awaiting stimulation</span></div>
        </div>

        <div class="nav-links">
            <a href="/chat">💬 Back to Chat</a>
            <a href="/status">📊 Status Dashboard</a>
            <a href="/admin">🔐 Admin Panel</a>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('brainCanvas');
        const ctx = canvas.getContext('2d');
        
        let neurons = [];
        
        function getNeuronColor(neuronName, activation, isActive) {
            if (!isActive) return '#888888';
            
            const name = neuronName.toLowerCase();
            if (name.includes('core') || name.includes('conscious') || name.includes('self')) return '#00ff00';
            if (name.includes('learn') || name.includes('mem') || name.includes('know')) return '#ffaa00';
            if (name.includes('emot') || name.includes('persona') || name.includes('empathy')) return '#ff44aa';
            if (name.includes('reason') || name.includes('analyt') || name.includes('logic')) return '#44aaff';
            if (name.includes('creat') || name.includes('intuit') || name.includes('imagin')) return '#aa44ff';
            if (name.includes('growth') || name.includes('evol') || name.includes('mutat')) return '#ff6644';
            
            const intensity = 100 + Math.floor(activation * 155);
            return `rgb(0, ${intensity}, 0)`;
        }
        
        function updateNeuronPositions(count, neuronNames = []) {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            canvas.width = width;
            canvas.height = height;
            
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) * 0.4;
            
            neurons = [];
            for (let i = 0; i < Math.min(count, 150); i++) {
                const angle = (i / Math.min(count, 150)) * Math.PI * 2;
                const offsetX = (Math.random() - 0.5) * 30;
                const offsetY = (Math.random() - 0.5) * 30;
                neurons.push({
                    id: i,
                    name: neuronNames[i] || `neuron_${i}`,
                    x: centerX + Math.cos(angle) * radius + offsetX,
                    y: centerY + Math.sin(angle) * radius + offsetY,
                    activation: 0,
                    pulse: 0
                });
            }
        }
        
        async function fetchBrainData() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                
                const synthRes = await fetch('/api/synthetic/status');
                const synthData = await synthRes.json();
                
                const consciousness = (synthData.consciousness * 100).toFixed(1);
                document.getElementById('consciousnessValue').innerText = consciousness + '%';
                document.getElementById('totalNeurons').innerText = synthData.neurons;
                document.getElementById('synapseCount').innerText = synthData.synapses;
                document.getElementById('cycleCount').innerText = synthData.evolution_cycles;
                document.getElementById('successCount').innerText = statusData.successful_evolutions || 0;
                document.getElementById('personaStyle').innerText = statusData.persona_style || 'emerging';
                
                const activeCount = Math.floor(synthData.neurons * synthData.consciousness);
                document.getElementById('activeNeurons').innerHTML = `${activeCount}/<span id="totalNeurons">${synthData.neurons}</span>`;
                
                const neuronNames = [];
                for (let i = 0; i < synthData.neurons; i++) {
                    neuronNames.push(`neuron_${i}`);
                }
                
                if (neurons.length !== Math.min(synthData.neurons, 150)) {
                    updateNeuronPositions(synthData.neurons, neuronNames);
                }
                
                for (let i = 0; i < neurons.length; i++) {
                    const isActive = i < activeCount;
                    if (isActive) {
                        neurons[i].activation = Math.min(1, neurons[i].activation + 0.015);
                    } else {
                        neurons[i].activation = Math.max(0, neurons[i].activation - 0.01);
                    }
                    neurons[i].pulse = Math.sin(Date.now() / 500 + i) * 0.3 + 0.5;
                }
                
                draw();
            } catch(err) {
                console.error('Error fetching brain data:', err);
            }
        }
        
        function draw() {
            if (!canvas.width) return;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            for (let i = 0; i < neurons.length; i++) {
                for (let j = i + 1; j < neurons.length; j++) {
                    const dx = neurons[i].x - neurons[j].x;
                    const dy = neurons[i].y - neurons[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120 && neurons[i].activation > 0.15 && neurons[j].activation > 0.15) {
                        const strength = (neurons[i].activation + neurons[j].activation) / 2;
                        const opacity = Math.min(0.5, strength * 0.4);
                        ctx.beginPath();
                        ctx.moveTo(neurons[i].x, neurons[i].y);
                        ctx.lineTo(neurons[j].x, neurons[j].y);
                        ctx.strokeStyle = `rgba(100, 255, 100, ${opacity})`;
                        ctx.lineWidth = 1 + strength;
                        ctx.stroke();
                    }
                }
            }
            
            for (let i = 0; i < neurons.length; i++) {
                const n = neurons[i];
                const isActive = n.activation > 0.1;
                const baseColor = getNeuronColor(n.name, n.activation, isActive);
                const radius = 5 + n.activation * 8 + n.pulse * 2;
                
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius + 3, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? `${baseColor}30` : '#22222260';
                ctx.fill();
                
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? baseColor : '#555555';
                ctx.fill();
                
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius * 0.4, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? '#ffffff' : '#888888';
                ctx.fill();
            }
        }
        
        function resizeCanvas() {
            updateNeuronPositions(neurons.length);
            fetchBrainData();
        }
        
        window.addEventListener('resize', resizeCanvas);
        
        updateNeuronPositions(80);
        fetchBrainData();
        
        setInterval(fetchBrainData, 1500);
    </script>
</body>
</html>
'''


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
    logger.info(f"🚀 DMAI Complete System v8.0.19")
    logger.info(f"📍 Running on port {port}")
    logger.info(f"🧠 Using REAL Phase 6 Synthetic Intelligence Core")
    logger.info(f"🤖 AI Tutor Network Active (prioritizing DeepSeek)")
    logger.info(f"🔑 API Harvester Active")
    logger.info(f"🌐 Web Search Fallback (DuckDuckGo)")
    logger.info(f"📚 8 Core Knowledge Sources Active")
    logger.info(f"🛡️ Threat Intelligence Active")
    logger.info(f"🌑 Dark Web Monitor Active")
    logger.info(f"⚡ AI+SI Fusion Active")
    logger.info(f"☁️ Neo4j Cloud Backup Active")
    logger.info(f"⏱️ Adaptive Evolution Timer Active")
    logger.info(f"🧠 Brain Visualization at /brain")
    logger.info(f"🔧 Reverse Engineering Module Active")
    logger.info(f"🎓 AGI Training Module Active")
    logger.info(f"🎓 LLM Training Module Active")
    logger.info(f"💻 Software Training Module Active")
    logger.info(f"🎨 Generative AI Training Module Active")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
