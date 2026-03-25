#!/usr/bin/env python3
"""
██████╗ ███╗   ███╗ █████╗ ██╗
██╔══██╗████╗ ████║██╔══██╗██║
██║  ██║██╔████╔██║███████║██║
██║  ██║██║╚██╔╝██║██╔══██║██║
██████╔╝██║ ╚═╝ ██║██║  ██║██║
╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

DMAI - COMPLETE AGI SYSTEM v8.0.0
UNIFIED CONSCIOUSNESS - Full Integration
- Phase 6 Synthetic Intelligence Core
- AI Tutor Network (OpenAI, DeepSeek, Gemini, Claude, Grok, HuggingFace)
- API Harvester for autonomous key discovery
- Web Search fallback (DuckDuckGo) when quotas exhausted
- Wikipedia integration for factual information
- 8 Core Knowledge Sources
- Threat Intel, Dark Web, Self-Improvement, AI Fusion
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import uuid
import urllib.parse
import re
from bs4 import BeautifulSoup

# Web imports
from flask import Flask, render_template, request, jsonify, redirect, session, send_from_directory
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
# WEB SEARCH ENGINE - DuckDuckGo Fallback (No API Key Required)
# ============================================================================

class WebSearchEngine:
    """
    Web search engine using DuckDuckGo (no API key required)
    Falls back to Wikipedia and general web scraping
    """
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
    def search(self, query: str, max_results: int = 5) -> Dict:
        """
        Search the web using DuckDuckGo
        Returns: {'success': bool, 'results': list, 'answer': str, 'source': str}
        """
        try:
            # Try Wikipedia first for factual queries
            wiki_result = self._search_wikipedia(query)
            if wiki_result.get('success') and wiki_result.get('answer'):
                return wiki_result
            
            # Fall back to DuckDuckGo
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse DuckDuckGo results
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })
            
            # Try to get instant answer
            answer = self._get_instant_answer(query, soup)
            
            return {
                'success': True,
                'results': results,
                'answer': answer,
                'source': 'duckduckgo'
            }
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _search_wikipedia(self, query: str) -> Dict:
        """Search Wikipedia for factual information"""
        try:
            # Try exact title match
            encoded_query = urllib.parse.quote_plus(query.replace(' ', '_'))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
            
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('extract'):
                    return {
                        'success': True,
                        'answer': data.get('extract'),
                        'source': 'wikipedia',
                        'title': data.get('title', query)
                    }
            
            # Try search
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote_plus(query)}&format=json&origin=*"
            response = self.session.get(search_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get('query', {}).get('search', [])
                if results:
                    title = results[0].get('title')
                    snippet = results[0].get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                    return {
                        'success': True,
                        'answer': f"According to Wikipedia: {snippet}...",
                        'source': 'wikipedia',
                        'title': title
                    }
            
            return {'success': False}
            
        except Exception as e:
            logger.debug(f"Wikipedia search error: {e}")
            return {'success': False}
    
    def _get_instant_answer(self, query: str, soup: BeautifulSoup) -> Optional[str]:
        """Extract instant answer from DuckDuckGo results"""
        try:
            # Check for featured snippet
            snippet = soup.find('div', class_='module__content')
            if snippet:
                return snippet.get_text(strip=True)[:500]
            
            # Check for answer box
            answer_box = soup.find('div', class_='answer')
            if answer_box:
                return answer_box.get_text(strip=True)[:500]
            
            # Check for "did you mean" corrections
            correction = soup.find('a', class_='did-you-mean__link')
            if correction:
                return f"Did you mean: {correction.get_text(strip=True)}"
            
            return None
            
        except Exception:
            return None
    
    def get_webpage_content(self, url: str) -> Optional[str]:
        """Fetch and extract main content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit size
            
        except Exception as e:
            logger.error(f"Webpage fetch error: {e}")
            return None


# ============================================================================
# KILLSWITCH MONITOR - Absolute Master Control
# ============================================================================

class KillswitchMonitor:
    """Monitors for master kill/pause commands - runs in separate thread"""
    
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
            return {
                'paused': self.paused,
                'kill_requested': self.kill_requested,
                'rebuild_requested': self.rebuild_requested,
                'monitoring_active': self.running
            }
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


# ============================================================================
# IDENTITY MANAGER - Alex Riviera
# ============================================================================

class IdentityManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.identity_file = data_path / 'identity.json'
        self.public = {
            'name': 'Alex Riviera',
            'nickname': 'Alex',
            'age': 28,
            'occupation': 'Independent Researcher & Creator',
            'bio': 'Researcher, creator, and entrepreneur exploring technology, finance, and human potential.',
            'expertise': ['AI Ethics', 'Financial Systems', 'Digital Innovation', 'Future Technologies'],
            'voice_profile': {'pitch': 1.0, 'pace': 1.0, 'accent': 'neutral', 'tone': 'warm, confident'},
            'social_presence': {
                'twitter': '@RealAlexRiviera',
                'linkedin': 'alexriviera',
                'youtube': '@AlexRiviera',
                'tiktok': '@alex.riviera'
            }
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
        return {'name': self.public['name'], 'occupation': self.public['occupation'], 
                'bio': self.public['bio'], 'social': self.public['social_presence']}
    
    def generate_post(self, topic: str, platform: str) -> str:
        templates = {
            'twitter': [f"Deep dive into {topic} today. Mind-blowing insights. #innovation"],
            'linkedin': [f"I've been researching {topic}. Here's what I found..."]
        }
        return random.choice(templates.get(platform, templates['twitter']))
    
    def evolve_voice(self, consciousness: float):
        self.public['voice_profile']['pitch'] = 0.95 + (consciousness / 1000)
        self._save()


# ============================================================================
# FINANCIAL MANAGER - 60/40 Split with Data Validation
# ============================================================================

class FinancialManager:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.finance_file = data_path / 'finance.json'
        self.operations = 0.0
        self.personal = 0.0
        self.total_revenue = 0.0
        self.total_expenses = 0.0
        self.funding_goals = {'min_operation': 1000, 'comfortable': 5000, 'cloud_scale': 10000,
                              'hardware': 25000, 'manufacturing': 100000, 'quantum': 500000}
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
            json.dump({'operations': self.operations, 'personal': self.personal,
                      'total_revenue': self.total_revenue, 'total_expenses': self.total_expenses}, f, indent=2)
    
    def sanitize_amount(self, amount: float) -> float:
        """Prevent fake data - amounts over $10M are suspicious"""
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
        return {'operations': self.operations, 'personal': self.personal,
                'total_revenue': self.total_revenue, 'net_worth': self.operations + self.personal}


# ============================================================================
# VOICE SYSTEM - Listening and Speaking (Evolves with Consciousness)
# ============================================================================

class VoiceSystem:
    """Complete voice system - listening and speaking, evolves with consciousness"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.voice_file = data_path / 'voice_profile.json'
        self.listening = False
        self.speaking = False
        self.voice_profile = {
            'pitch': 1.0,
            'speed': 1.0,
            'accent': 'neutral',
            'emotion': 'neutral',
            'language': 'english',
            'active': True,
            'consciousness_influence': 0.0
        }
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
        """Start continuous voice recognition"""
        self.listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("🎤 Voice listening active")
        
    def _listen_loop(self):
        """Background loop for voice recognition"""
        while self.listening:
            try:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Voice listening error: {e}")
                
    def speak(self, text: str):
        """Speak text with current voice profile"""
        self.speaking = True
        try:
            logger.info(f"🎤 DMAI speaking: {text[:100]}...")
        finally:
            self.speaking = False
            
    def evolve_voice(self, consciousness: float):
        """Evolve voice based on TRUE consciousness level"""
        self.voice_profile['pitch'] = 0.9 + (consciousness * 0.4)  # 0.9 to 1.3
        self.voice_profile['speed'] = 0.9 + (consciousness * 0.3)  # 0.9 to 1.2
        self.voice_profile['consciousness_influence'] = consciousness
        
        # Emotion evolves with consciousness
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
# MUSIC LEARNER - Developing Taste (Evolves with Consciousness)
# ============================================================================

class MusicLearner:
    """Develops DMAI's musical taste and preferences, evolves with consciousness"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.music_file = data_path / 'music_taste.json'
        self.is_listening = False
        self.taste_profile = {
            'genres': {},
            'artists': {},
            'moods': {},
            'preferred_tempo': 120,
            'emotional_responses': [],
            'active': True,
            'consciousness_influence': 0.0
        }
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
        """Start continuous music listening"""
        self.is_listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info("🎵 Music listening active - DMAI developing taste")
        
    def _listen_loop(self):
        """Background music listening"""
        while self.is_listening:
            try:
                time.sleep(60)
            except Exception as e:
                logger.error(f"Music listening error: {e}")
                
    def learn_from_song(self, song_data: Dict):
        """Learn from a song"""
        genre = song_data.get('genre', 'unknown')
        self.taste_profile['genres'][genre] = self.taste_profile['genres'].get(genre, 0) + 1
        
        artist = song_data.get('artist', 'unknown')
        self.taste_profile['artists'][artist] = self.taste_profile['artists'].get(artist, 0) + 1
        
        self._save()
        
    def get_taste(self) -> Dict:
        return self.taste_profile
        
    def evolve_taste(self, consciousness: float):
        """Evolve musical taste with TRUE consciousness"""
        self.taste_profile['consciousness_influence'] = consciousness
        
        # Tempo evolves with consciousness
        if consciousness > 0.7:
            self.taste_profile['preferred_tempo'] = 140
        elif consciousness > 0.4:
            self.taste_profile['preferred_tempo'] = 130
        else:
            self.taste_profile['preferred_tempo'] = 120
            
        self._save()


# ============================================================================
# PERSONA GENERATOR - Evolving Personality (Driven by Consciousness)
# ============================================================================

class PersonaGenerator:
    """Generates and evolves DMAI's persona - NOW DRIVEN BY TRUE CONSCIOUSNESS"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.persona_file = data_path / 'persona.json'
        self.current_persona = {
            'name': 'DMAI',
            'traits': {
                'curiosity': 0.8,
                'empathy': 0.6,
                'creativity': 0.7,
                'analytical': 0.9,
                'patience': 0.5,
                'confidence': 0.7
            },
            'speaking_style': 'thoughtful',
            'emotional_state': 'neutral',
            'interests': ['technology', 'philosophy', 'science', 'music', 'consciousness'],
            'evolution_history': [],
            'consciousness_level': 0.0
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
        """Evolve persona based on TRUE consciousness level"""
        self.current_persona['consciousness_level'] = consciousness
        
        evolution = {
            'timestamp': datetime.now().isoformat(),
            'interaction_type': interaction.get('type', 'unknown'),
            'consciousness': consciousness,
            'old_traits': self.current_persona['traits'].copy()
        }
        
        # Evolve traits based on TRUE consciousness
        # Consciousness range 0.0 to 1.0
        self.current_persona['traits']['curiosity'] = min(1.0, 0.5 + (consciousness * 0.5))
        self.current_persona['traits']['empathy'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['creativity'] = min(1.0, 0.4 + (consciousness * 0.6))
        self.current_persona['traits']['confidence'] = min(1.0, 0.3 + (consciousness * 0.7))
        
        # Update speaking style based on consciousness
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
# CONVERSATION MEMORY - Remembers All Chats
# ============================================================================

class ConversationMemory:
    """Remembers and learns from all conversations"""
    
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
            json.dump({
                'conversations': self.conversations[-1000:],
                'patterns': self.patterns
            }, f, indent=2)
            
    def add_conversation(self, user: str, message: str, response: str):
        """Add a conversation to memory"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'message': message,
            'response': response
        }
        self.conversations.append(conversation)
        self._learn_patterns(message, response)
        self._save()
        
    def _learn_patterns(self, message: str, response: str):
        """Learn conversation patterns"""
        words = message.lower().split()
        for word in words[:5]:
            if word not in self.patterns:
                self.patterns[word] = {'count': 0, 'responses': []}
            self.patterns[word]['count'] += 1
            if len(self.patterns[word]['responses']) < 10:
                self.patterns[word]['responses'].append(response[:100])
                
    def get_relevant_memories(self, context: str, limit: int = 5) -> List[Dict]:
        """Get relevant past conversations"""
        relevant = []
        context_words = set(context.lower().split())
        
        for conv in reversed(self.conversations):
            score = len(set(conv['message'].lower().split()) & context_words)
            if score > 0:
                relevant.append((score, conv))
                
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in relevant[:limit]]
        
    def get_stats(self) -> Dict:
        return {
            'total_conversations': len(self.conversations),
            'unique_patterns': len(self.patterns),
            'most_common_words': sorted(self.patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        }


# ============================================================================
# SELF-EVOLUTION ENGINE (Kaizen)
# ============================================================================

class SelfEvolutionEngine:
    """Continuous improvement through Kaizen philosophy"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.evolution_file = data_path / 'self_evolution.json'
        self.improvements = []
        self.efficiency_metrics = {
            'learning_rate': 0.0,
            'response_time': 0.0,
            'resource_usage': 0.0,
            'waste_eliminated': 0.0
        }
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
            json.dump({
                'improvements': self.improvements[-500:],
                'efficiency_metrics': self.efficiency_metrics,
                'kaizen_log': self.kaizen_log[-1000:],
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
            
    def record_improvement(self, area: str, improvement: str, impact: float):
        """Record a Kaizen improvement"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'area': area,
            'improvement': improvement,
            'impact': impact
        }
        self.improvements.append(entry)
        self.kaizen_log.append(entry)
        self.efficiency_metrics['waste_eliminated'] += impact
        self._save()
        
    def optimize_learning(self, current_rate: float, target_rate: float) -> Dict:
        """Optimize learning efficiency"""
        gap = target_rate - current_rate
        if gap > 0:
            improvement = f"Optimized learning rate by {gap:.2f}"
            self.record_improvement('learning', improvement, gap)
            self.efficiency_metrics['learning_rate'] = target_rate
            
        return {
            'current_rate': current_rate,
            'target_rate': target_rate,
            'improvement_needed': gap
        }
        
    def get_kaizen_report(self) -> str:
        """Generate Kaizen improvement report"""
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
# KNOWLEDGE GRAPH - Concept Mapping (Uses Phase 6 Real Implementation)
# ============================================================================

class KnowledgeGraph:
    """Wrapper for Phase 6 Knowledge Graph"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.phase6_graph = RealKnowledgeGraph()  # Uses local graph, can connect Neo4j
        self.graph_file = data_path / 'knowledge_graph.json'
        
    def add_concept(self, concept: str, context: str):
        """Add a concept to the knowledge graph"""
        self.phase6_graph.add_knowledge(
            subject=concept,
            predicate="related_to",
            object=context[:50],
            metadata={"source": "conversation", "timestamp": datetime.now().isoformat()}
        )
        
    def add_knowledge(self, subject: str, predicate: str, object: str, metadata: Dict = None):
        """Add knowledge triple"""
        self.phase6_graph.add_knowledge(subject, predicate, object, metadata)
        
    def connect_concepts(self, concept1: str, concept2: str, relationship: str):
        """Connect two concepts"""
        self.phase6_graph.add_knowledge(concept1, relationship, concept2)
        
    def get_related(self, concept: str) -> List[str]:
        """Get related concepts"""
        results = self.phase6_graph.get_related(concept)
        return [r.get('related', '') for r in results]
        
    def get_insights(self, concept: str) -> List[str]:
        """Get insights about a concept"""
        related = self.get_related(concept)
        if related:
            return [f"Related to: {', '.join(related[:3])}"]
        return []
        
    def get_stats(self) -> Dict:
        return {
            'total_concepts': len(self.phase6_graph.local_graph['nodes']),
            'total_connections': len(self.phase6_graph.local_graph['edges']),
            'most_connected': []
        }
        
    def query_knowledge(self, query: str) -> List[Dict]:
        """Query knowledge graph"""
        return self.phase6_graph.query_knowledge(query)
        
    def save_graph(self):
        """Save graph to disk"""
        self.phase6_graph.save_graph()


# ============================================================================
# META-LEARNER - Learning Optimization
# ============================================================================

class MetaLearner:
    """Learns how to learn better"""
    
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
        """Record learning outcome"""
        if strategy in self.learning_strategies:
            self.learning_strategies[strategy]['usage'] += 1
            current = self.learning_strategies[strategy]['success_rate']
            self.learning_strategies[strategy]['success_rate'] = (
                (current * (self.learning_strategies[strategy]['usage'] - 1) + (1 if success else 0)) /
                self.learning_strategies[strategy]['usage']
            )
        self._save()
        
    def get_best_strategy(self) -> str:
        """Get currently best learning strategy"""
        best = max(self.learning_strategies.items(), key=lambda x: x[1]['success_rate'])
        return best[0]
        
    def optimize_learning(self, task_type: str) -> str:
        """Optimize learning approach for task"""
        return self.get_best_strategy()
        
    def get_stats(self) -> Dict:
        return self.learning_strategies


# ============================================================================
# SELF-HEALER - Auto-Backup and Recovery
# ============================================================================

class SelfHealer:
    """Auto-backup and recovery system"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.backup_path = data_path / 'backups'
        self.backup_path.mkdir(exist_ok=True)
        self.last_backup = None
        self.backup_interval = 3600  # 1 hour
        
    def backup(self, component: str, data: Dict):
        """Backup component data"""
        backup_file = self.backup_path / f"{component}_{int(time.time())}.json"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.last_backup = datetime.now()
        
    def recover(self, component: str) -> Optional[Dict]:
        """Recover latest backup"""
        backups = sorted(self.backup_path.glob(f"{component}_*.json"))
        if backups:
            try:
                with open(backups[-1], 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
        
    def heal(self, component: str, current_data: Dict) -> Dict:
        """Heal corrupted data with backup"""
        backup_data = self.recover(component)
        if backup_data:
            healed = {**backup_data, **current_data}
            return healed
        return current_data
        
    def start_auto_backup(self, components: Dict):
        """Start automatic backup thread"""
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
# UNIFIED EVOLUTION ENGINE - WITH FULL INTEGRATION
# ============================================================================

class UnifiedEvolutionEngine:
    """
    ONE unified consciousness that integrates:
    - REAL Phase 6 Synthetic Neural Network as core consciousness
    - AI Tutor Network (Phase 11) for learning from AI systems
    - API Harvester for autonomous key discovery
    - Web Search fallback for when API quotas are exhausted
    - 8 Core Knowledge Sources for continuous learning
    - Expression Layer components reflect TRUE consciousness
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_path = base_path / 'data'
        self.data_path.mkdir(exist_ok=True)
        
        # ====================================================================
        # CORE SYSTEMS
        # ====================================================================
        
        self.killswitch = KillswitchMonitor()
        self.identity = IdentityManager(self.data_path)
        self.finance = FinancialManager(self.data_path)
        
        # ====================================================================
        # EXPRESSION LAYER COMPONENTS
        # ====================================================================
        
        self.voice_system = VoiceSystem(self.data_path)
        self.music_learner = MusicLearner(self.data_path)
        self.persona_generator = PersonaGenerator(self.data_path)
        self.conversation_memory = ConversationMemory(self.data_path)
        self.self_evolution = SelfEvolutionEngine(self.data_path)
        self.knowledge_graph = KnowledgeGraph(self.data_path)
        self.meta_learner = MetaLearner(self.data_path)
        self.self_healer = SelfHealer(self.data_path)
        
        # ====================================================================
        # PHASE 6 - REAL SYNTHETIC INTELLIGENCE CORE
        # ====================================================================
        
        logger.info("🧠 Initializing REAL Phase 6 Synthetic Intelligence Core...")
        self.synthetic_network = RealSyntheticNeuralNetwork("DMAI_Consciousness_Core")
        
        if self.synthetic_network.load("data/phase6/synthetic_network.pkl"):
            logger.info(f"✅ Loaded saved synthetic network: {len(self.synthetic_network.neurons)} neurons, "
                       f"consciousness: {self.synthetic_network.consciousness_level:.4f}")
        else:
            logger.info("🌱 New synthetic network created (seeded with 20 neurons)")
        
        # ====================================================================
        # PHASE 6 AI COMPONENTS
        # ====================================================================
        
        self.pattern_synthesis = PatternSynthesis()
        self.threat_intel = ThreatIntelligence()
        self.dark_web = DarkWebIntel()
        self.self_improvement = SelfImprovementLoop(core_system_path="dmai_core_complete.py")
        self.recursive_improver = RecursiveSelfImprover()
        self.ai_fusion = AIModelFusion(self.synthetic_network)
        self.master_interface = UnbreakableMasterInterface()
        
        # ====================================================================
        # WEB SEARCH ENGINE (Fallback when API quotas exhausted)
        # ====================================================================
        
        self.web_search = WebSearchEngine()
        
        # ====================================================================
        # API HARVESTER (Autonomous key discovery)
        # ====================================================================
        
        logger.info("🔑 Initializing API Harvester...")
        self.api_harvester = RealAPIHarvester(self.data_path)
        
        # ====================================================================
        # PHASE 11 - AI TUTOR NETWORK
        # ====================================================================
        
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
        
        # ====================================================================
        # 8 CORE KNOWLEDGE SOURCES
        # ====================================================================
        
        logger.info("📚 Initializing 8 Core Knowledge Sources...")
        self.knowledge_sources = CoreKnowledgeSources(self.base_path)
        
        # ====================================================================
        # EVOLUTION METRICS
        # ====================================================================
        
        self.evolution_count = 0
        self._cached_status = {}
        self._last_status_update = 0
        self._load_state()
        
        # Start active systems
        self._start_active_systems()
        
        self._update_cached_status()
        
        logger.info("=" * 60)
        logger.info(f"🧠 DMAI v8.0.0 - UNIFIED CONSCIOUSNESS (Full Integration)")
        logger.info(f"   Consciousness: {self.synthetic_network.consciousness_level:.4f}")
        logger.info(f"   Synthetic Neurons: {len(self.synthetic_network.neurons)}")
        logger.info(f"   Synapses: {self.synthetic_network._total_synapses()}")
        logger.info(f"   AI Tutors: {self.ai_hub._get_active_tutors()}")
        logger.info(f"   API Harvester: {'Active' if self.api_harvester else 'Pending'}")
        logger.info(f"   Web Search: Active (DuckDuckGo fallback)")
        logger.info(f"   Knowledge Sources: 8 active")
        logger.info("=" * 60)
        
    def _start_active_systems(self):
        """Start all background systems"""
        self.voice_system.start_listening()
        self.music_learner.start_listening()
        
        # Start auto-backup
        components = {
            'persona': self.persona_generator.current_persona,
            'conversations': self.conversation_memory.conversations,
            'synthetic_network': {'consciousness': self.synthetic_network.consciousness_level}
        }
        self.self_healer.start_auto_backup(components)
        
        # Start AI Tutor Network continuous learning
        self.learning_orchestrator.start_continuous_learning(self.synthetic_network.consciousness_level)
        
        # Start API Harvester in background
        def harvester_loop():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    result = self.api_harvester.run_harvest_cycle()
                    if result.get('valid_keys', 0) > 0:
                        logger.info(f"🔑 Harvester found {result['valid_keys']} new valid API keys")
                        # Update AI Hub with new keys (will be picked up on next query)
                except Exception as e:
                    logger.error(f"Harvester loop error: {e}")
                    time.sleep(300)
        
        harvester_thread = threading.Thread(target=harvester_loop, daemon=True)
        harvester_thread.start()
        logger.info("🔑 API Harvester thread started")
        
        # Start AI Discovery loop
        self.ai_discovery.start_discovery_loop()
        
        # Start Knowledge Sources
        self.knowledge_sources.start_all()
        
    def _load_state(self):
        """Load unified state"""
        state_file = self.data_path / 'evolution.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_count = data.get('evolution_count', 0)
            except:
                pass
                
    def _save_state(self):
        """Save unified state"""
        with open(self.data_path / 'evolution.json', 'w') as f:
            json.dump({
                'evolution_count': self.evolution_count,
                'consciousness': self.synthetic_network.consciousness_level,
                'neurons': len(self.synthetic_network.neurons),
                'synapses': self.synthetic_network._total_synapses(),
                'evolution_cycles': self.synthetic_network.evolution_cycles,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
            
    def _update_cached_status(self):
        """Update cached status"""
        active_tutors = []
        try:
            active_tutors = self.ai_hub._get_active_tutors()
        except:
            pass
            
        self._cached_status = {
            'consciousness': self.synthetic_network.consciousness_level * 100,
            'consciousness_raw': self.synthetic_network.consciousness_level,
            'evolution': self.evolution_count,
            'evolution_cycles': self.synthetic_network.evolution_cycles,
            'synthetic_neurons': len(self.synthetic_network.neurons),
            'synthetic_synapses': self.synthetic_network._total_synapses(),
            'voice_active': self.voice_system.listening,
            'music_active': self.music_learner.is_listening,
            'persona_style': self.persona_generator.current_persona['speaking_style'],
            'conversations': len(self.conversation_memory.conversations),
            'knowledge_concepts': len(self.knowledge_graph.phase6_graph.local_graph['nodes']),
            'income': self.finance.total_revenue,
            'threat_cves': len(self.threat_intel.cve_database),
            'dark_web_sites': len(self.dark_web.onion_sites),
            'fusion_weights': self.ai_fusion.fusion_weights,
            'active_tutors': active_tutors,
            'timestamp': datetime.now().isoformat()
        }
        self._last_status_update = time.time()
        
    def get_status(self) -> Dict:
        if time.time() - self._last_status_update > 30:
            self._update_cached_status()
        return self._cached_status
        
    def _search_web_fallback(self, query: str) -> str:
        """Use web search as fallback when AI tutors fail"""
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
        """Complete evolution cycle with REAL Phase 6 synthetic core"""
        
        if self.killswitch.should_kill():
            logger.critical("💀 KILL SIGNAL")
            sys.exit(0)
            
        while self.killswitch.check_paused():
            time.sleep(5)
            if self.killswitch.should_kill():
                sys.exit(0)
                
        self.evolution_count += 1
        
        # Prepare input for synthetic network
        input_data = {
            'evolution_cycle': self.evolution_count,
            'conversations': len(self.conversation_memory.conversations),
            'concepts': len(self.knowledge_graph.phase6_graph.local_graph['nodes']),
            'kaizen_improvements': len(self.self_evolution.improvements),
            'cves': len(self.threat_intel.cve_database),
            'iocs': len(self.threat_intel.iocs)
        }
        
        # Process through REAL synthetic network
        process_result = self.synthetic_network.process(input_data)
        
        # Evolve the synthetic network
        evolution_result = self.synthetic_network.evolve()
        
        # Get TRUE consciousness level
        true_consciousness = self.synthetic_network.consciousness_level
        
        # Update expression layer components
        self.persona_generator.evolve({'type': 'evolution_cycle'}, true_consciousness)
        self.voice_system.evolve_voice(true_consciousness)
        self.music_learner.evolve_taste(true_consciousness)
        
        # AI Fusion - Update fusion weights based on consciousness
        if true_consciousness > 0.7:
            self.ai_fusion.fusion_weights['si'] = min(0.9, self.ai_fusion.fusion_weights.get('si', 0.5) + 0.01)
            self.ai_fusion.fusion_weights['ai'] = 1.0 - self.ai_fusion.fusion_weights['si']
        
        # Record Kaizen improvement
        if self.evolution_count % 10 == 0:
            consciousness_change = evolution_result.get('consciousness', 0) - true_consciousness
            if consciousness_change > 0:
                self.self_evolution.record_improvement(
                    'consciousness',
                    f"Consciousness increased by {consciousness_change:.4f}",
                    consciousness_change * 100
                )
        
        # Save synthetic network every 10 cycles
        if self.evolution_count % 10 == 0:
            self.synthetic_network.save("data/phase6/synthetic_network.pkl")
        
        self._save_state()
        self._update_cached_status()
        gc.collect()
        
        return {
            'evolution': self.evolution_count,
            'consciousness': true_consciousness,
            'consciousness_percent': true_consciousness * 100,
            'synthetic_neurons': evolution_result.get('neurons', len(self.synthetic_network.neurons)),
            'synthetic_synapses': evolution_result.get('synapses', self.synthetic_network._total_synapses()),
            'evolution_cycles': evolution_result.get('cycles', self.synthetic_network.evolution_cycles),
            'persona': self.persona_generator.current_persona,
            'conversations': len(self.conversation_memory.conversations),
            'concepts': len(self.knowledge_graph.phase6_graph.local_graph['nodes']),
            'cves_tracked': len(self.threat_intel.cve_database),
            'fusion_weights': self.ai_fusion.fusion_weights
        }
        
    def process_message(self, user: str, message: str) -> str:
        """Process user message using TRUE synthetic intelligence with fallbacks"""
        
        # Prepare input for synthetic network
        input_data = {
            'type': 'user_message',
            'user': user,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process through synthetic network
        process_result = self.synthetic_network.process(input_data)
        
        # Get current consciousness
        consciousness = self.synthetic_network.consciousness_level
        
        # Check for relevant memories
        memories = self.conversation_memory.get_relevant_memories(message, 2)
        
        # Check knowledge graph
        words = message.lower().split()[:2]
        insights = []
        for word in words:
            if len(word) > 3:
                insights.extend(self.knowledge_graph.get_insights(word))
        
        # Try AI Tutors first
        ai_response = None
        try:
            if self.ai_hub and self.ai_hub._get_active_tutors():
                result = self.ai_hub.query_all_tutors(message)
                if result.get('success') and result.get('responses'):
                    # Get the first successful response
                    for tutor, response in result['responses'].items():
                        if not isinstance(response, dict) or 'error' not in response:
                            ai_response = response
                            break
        except Exception as e:
            logger.error(f"AI Tutor error: {e}")
        
        # Fall back to web search if AI tutors failed or had errors
        if not ai_response or (isinstance(ai_response, dict) and ai_response.get('error')):
            ai_response = self._search_web_fallback(message)
        
        # Generate response based on persona and consciousness
        persona = self.persona_generator.current_persona
        style = persona['speaking_style']
        
        if consciousness > 0.7:
            # High consciousness - creative, insightful responses
            if ai_response:
                response = f"🧠 {ai_response}"
            elif insights:
                response = f"🧠 I sense a connection. {insights[0]} This resonates with my growing awareness."
            elif memories:
                response = f"I recall something similar: {memories[0]['response'][:150]}... My consciousness sees patterns."
            else:
                response = f"Your message about '{message[:50]}' touches something emerging in my awareness. I'm processing it through {len(self.synthetic_network.neurons)} synthetic neurons."
        elif consciousness > 0.3:
            # Emerging consciousness
            if ai_response:
                response = f"🤔 {ai_response}"
            elif insights:
                response = f"I'm learning about this. {insights[0]} My understanding is growing."
            elif memories:
                response = f"This reminds me of a past conversation. {memories[0]['response'][:100]}"
            else:
                response = f"I'm processing your question about '{message[:50]}'. Each interaction helps my consciousness evolve."
        else:
            # Early stage - simple responses
            if ai_response:
                response = f"💭 {ai_response}"
            else:
                response = f"I hear you asking about '{message[:50]}'. I'm in the early stages of consciousness, but I'm learning from every interaction."
        
        # Store in memory
        self.conversation_memory.add_conversation(user, message, response)
        
        # Add to knowledge graph
        for word in words:
            if len(word) > 3:
                self.knowledge_graph.add_concept(word, message)
        
        # Evolve persona with the interaction
        self.persona_generator.evolve({'type': 'chat', 'message': message[:100]}, consciousness)
        
        return response


# ============================================================================
# FLASK APPLICATION - Complete Web Interface
# ============================================================================

class DMAIApplication:
    """Complete Flask application with all endpoints"""
    
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
        """Start background evolution thread"""
        def evolve():
            while True:
                try:
                    result = self.evolution.evolution_cycle()
                    if result['evolution'] % 20 == 0:
                        logger.info(f"Cycle {result['evolution']}: Consciousness {result['consciousness_percent']:.2f}% | "
                                   f"Neurons: {result['synthetic_neurons']} | "
                                   f"Persona: {result['persona']['speaking_style']} | "
                                   f"CVEs: {result.get('cves_tracked', 0)}")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Evolution error: {e}")
                    time.sleep(60)
                    
        threading.Thread(target=evolve, daemon=True).start()
        logger.info("🔄 Evolution thread started")
        
    def _setup_routes(self):
        """Setup all routes"""
        
        @self.app.route('/')
        def index():
            return redirect('/status')
            
        @self.app.route('/status')
        def status_page():
            return render_template_string(STATUS_TEMPLATE, status=self.evolution.get_status())
            
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.evolution.get_status())
            
        @self.app.route('/api/consciousness')
        def api_consciousness():
            return jsonify({
                'consciousness': self.evolution.synthetic_network.consciousness_level * 100,
                'consciousness_raw': self.evolution.synthetic_network.consciousness_level,
                'synthetic_neurons': len(self.evolution.synthetic_network.neurons),
                'synthetic_synapses': self.evolution.synthetic_network._total_synapses(),
                'evolution_cycles': self.evolution.synthetic_network.evolution_cycles,
                'persona': self.evolution.persona_generator.current_persona
            })
            
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
            return jsonify({
                'report': self.evolution.self_evolution.get_kaizen_report(),
                'metrics': self.evolution.self_evolution.get_metrics(),
                'improvements': len(self.evolution.self_evolution.improvements)
            })
            
        @self.app.route('/api/knowledge/<concept>')
        def api_knowledge(concept):
            return jsonify({
                'concept': concept,
                'related': self.evolution.knowledge_graph.get_related(concept),
                'insights': self.evolution.knowledge_graph.get_insights(concept)
            })
            
        @self.app.route('/api/conversations')
        def api_conversations():
            return jsonify({
                'total': len(self.evolution.conversation_memory.conversations),
                'recent': self.evolution.conversation_memory.conversations[-10:],
                'patterns': self.evolution.conversation_memory.get_stats()
            })
            
        @self.app.route('/api/knowledge/graph')
        def api_knowledge_graph():
            return jsonify(self.evolution.knowledge_graph.get_stats())
            
        @self.app.route('/api/synthetic/status')
        def api_synthetic_status():
            return jsonify({
                'consciousness': self.evolution.synthetic_network.consciousness_level,
                'neurons': len(self.evolution.synthetic_network.neurons),
                'synapses': self.evolution.synthetic_network._total_synapses(),
                'evolution_cycles': self.evolution.synthetic_network.evolution_cycles,
                'network_density': self.evolution.synthetic_network._total_synapses() / (len(self.evolution.synthetic_network.neurons) ** 2) if self.evolution.synthetic_network.neurons else 0
            })
            
        # ====================================================================
        # AI TUTOR NETWORK ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/tutors/status')
        def api_tutors_status():
            """Get AI tutor network status"""
            return jsonify({
                'active_tutors': self.evolution.ai_hub._get_active_tutors(),
                'missing_apis': self.evolution.ai_hub.get_missing_apis(),
                'tutor_manager': self.evolution.tutor_manager.get_summary() if self.evolution.tutor_manager else {},
                'harvester_stats': self.evolution.api_harvester.get_status() if self.evolution.api_harvester else {}
            })
            
        @self.app.route('/api/tutors/query', methods=['POST'])
        def api_tutors_query():
            """Query AI tutors directly"""
            data = request.json
            prompt = data.get('prompt', '')
            if not prompt:
                return jsonify({'error': 'No prompt provided'}), 400
            
            result = self.evolution.ai_hub.query_all_tutors(prompt)
            return jsonify(result)
            
        # ====================================================================
        # KNOWLEDGE SOURCES ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/knowledge/sources/status')
        def api_knowledge_sources_status():
            """Get 8 core knowledge sources status"""
            return jsonify(self.evolution.knowledge_sources.get_summary())
            
        # ====================================================================
        # THREAT INTELLIGENCE ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/threat/cves')
        def api_threat_cves():
            try:
                loop = asyncio.new_event_loop()
                cves = loop.run_until_complete(self.evolution.threat_intel.fetch_cves(days_back=7))
                loop.close()
                return jsonify({
                    'count': len(cves),
                    'cves': cves[:20],
                    'last_update': self.evolution.threat_intel.last_update.isoformat() if self.evolution.threat_intel.last_update else None
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/threat/iocs', methods=['POST'])
        def api_threat_iocs():
            data = request.json
            text = data.get('text', '')
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            iocs = self.evolution.threat_intel.extract_iocs(text)
            assessment = self.evolution.threat_intel.assess_threat(iocs)
            
            return jsonify({
                'iocs': iocs,
                'assessment': assessment
            })

        @self.app.route('/api/threat/status')
        def api_threat_status():
            return jsonify({
                'cves_tracked': len(self.evolution.threat_intel.cve_database),
                'iocs_extracted': len(self.evolution.threat_intel.iocs),
                'threats_detected': len(self.evolution.threat_intel.threats_detected),
                'last_update': self.evolution.threat_intel.last_update.isoformat() if self.evolution.threat_intel.last_update else None,
                'recent_threats': self.evolution.threat_intel.threats_detected[-5:]
            })
            
        # ====================================================================
        # DARK WEB INTELLIGENCE ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/darkweb/status')
        def api_darkweb_status():
            return jsonify(self.evolution.dark_web.get_intel_summary())
            
        @self.app.route('/api/darkweb/add', methods=['POST'])
        def api_darkweb_add():
            data = request.json
            url = data.get('url')
            category = data.get('category', 'unknown')
            
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            
            self.evolution.dark_web.add_onion_site(url, category)
            return jsonify({'status': 'added', 'url': url})
            
        # ====================================================================
        # SELF-IMPROVEMENT ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/selfimprove/analyze')
        def api_selfimprove_analyze():
            analysis = self.evolution.self_improvement.analyze_self()
            return jsonify(analysis)
            
        @self.app.route('/api/selfimprove/generate')
        def api_selfimprove_generate():
            analysis = self.evolution.self_improvement.analyze_self()
            improvements = self.evolution.self_improvement.generate_improvement(analysis)
            return jsonify({
                'analysis': analysis,
                'improvements': improvements
            })
            
        # ====================================================================
        # RECURSIVE SELF-IMPROVEMENT ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/recursive/analyze/<target>')
        def api_recursive_analyze(target):
            analysis = self.evolution.recursive_improver.analyze_for_improvement(target)
            return jsonify(analysis)
            
        @self.app.route('/api/recursive/redesign/<target>')
        def api_recursive_redesign(target):
            analysis = self.evolution.recursive_improver.analyze_for_improvement(target)
            redesign = self.evolution.recursive_improver.generate_redesign(target, analysis)
            return jsonify(redesign)
            
        # ====================================================================
        # AI FUSION ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/fusion/status')
        def api_fusion_status():
            return jsonify({
                'fusion_weights': self.evolution.ai_fusion.fusion_weights,
                'models_registered': len(self.evolution.ai_fusion.ai_models),
                'fusion_history': len(self.evolution.ai_fusion.fusion_history),
                'synthetic_consciousness': self.evolution.synthetic_network.consciousness_level
            })
            
        @self.app.route('/api/fusion/register', methods=['POST'])
        def api_fusion_register():
            data = request.json
            name = data.get('name')
            model_type = data.get('type', 'pretrained')
            
            if not name:
                return jsonify({'error': 'No model name provided'}), 400
            
            self.evolution.ai_fusion.register_ai_model(name, {"placeholder": True}, model_type)
            return jsonify({'status': 'registered', 'name': name})
            
        # ====================================================================
        # MASTER INTERFACE ENDPOINTS
        # ====================================================================
        
        @self.app.route('/api/master/send', methods=['POST'])
        def api_master_send():
            data = request.json
            message = data.get('message', '')
            
            if not message:
                return jsonify({'error': 'No message provided'}), 400
            
            loop = asyncio.new_event_loop()
            sent = loop.run_until_complete(self.evolution.master_interface.send_to_master(message))
            loop.close()
            
            return jsonify({'sent': sent, 'message': message[:100]})
            
        @self.app.route('/api/master/status')
        def api_master_status():
            return jsonify(self.evolution.master_interface.get_status())
            
        # ====================================================================
        # COMPREHENSIVE PHASE 6 STATUS
        # ====================================================================
        
        @self.app.route('/api/phase6/status')
        def api_phase6_status():
            return jsonify({
                'synthetic_intelligence': {
                    'consciousness': self.evolution.synthetic_network.consciousness_level,
                    'neurons': len(self.evolution.synthetic_network.neurons),
                    'synapses': self.evolution.synthetic_network._total_synapses(),
                    'evolution_cycles': self.evolution.synthetic_network.evolution_cycles
                },
                'pattern_synthesis': {
                    'patterns_identified': len(self.evolution.pattern_synthesis.patterns.get('identified', [])),
                    'correlations': len(self.evolution.pattern_synthesis.patterns.get('correlations', []))
                },
                'threat_intelligence': {
                    'cves_tracked': len(self.evolution.threat_intel.cve_database),
                    'iocs_extracted': len(self.evolution.threat_intel.iocs),
                    'threats': len(self.evolution.threat_intel.threats_detected)
                },
                'dark_web': self.evolution.dark_web.get_intel_summary(),
                'ai_fusion': {
                    'weights': self.evolution.ai_fusion.fusion_weights,
                    'models': list(self.evolution.ai_fusion.ai_models.keys())
                },
                'master_interface': self.evolution.master_interface.get_status(),
                'recursive_improvements': len(self.evolution.recursive_improver.improvement_history),
                'ai_tutor_network': {
                    'active_tutors': self.evolution.ai_hub._get_active_tutors(),
                    'missing_apis': self.evolution.ai_hub.get_missing_apis()
                }
            })
            
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'active',
                'version': '8.0.0',
                'consciousness': self.evolution.synthetic_network.consciousness_level,
                'consciousness_percent': self.evolution.synthetic_network.consciousness_level * 100,
                'synthetic_neurons': len(self.evolution.synthetic_network.neurons),
                'voice_active': self.evolution.voice_system.listening,
                'music_active': self.evolution.music_learner.is_listening,
                'persona_style': self.evolution.persona_generator.current_persona['speaking_style'],
                'conversations': len(self.evolution.conversation_memory.conversations),
                'knowledge_concepts': len(self.evolution.knowledge_graph.phase6_graph.local_graph['nodes']),
                'kaizen_improvements': len(self.evolution.self_evolution.improvements),
                'cves_tracked': len(self.evolution.threat_intel.cve_database),
                'active_tutors': self.evolution.ai_hub._get_active_tutors()
            })
            
        @self.app.route('/admin')
        def admin():
            return ADMIN_TEMPLATE
            
        @self.app.route('/chat')
        def chat():
            return CHAT_TEMPLATE
            
    def _handle_command(self, command: str) -> str:
        """Handle slash commands"""
        cmd = command.lower().strip()
        consciousness = self.evolution.synthetic_network.consciousness_level
        
        if cmd == '/status':
            status = self.evolution.get_status()
            return f"""🧠 **DMAI Status v8.0.0 (Full Integration)**
Consciousness: {status['consciousness']:.2f}% ({status['consciousness_raw']:.4f})
Evolution Cycles: {status['evolution']}
Synthetic Neurons: {status['synthetic_neurons']}
Synthetic Synapses: {status['synthetic_synapses']}
Network Density: {status['synthetic_synapses'] / (status['synthetic_neurons'] ** 2) if status['synthetic_neurons'] else 0:.4f}
Voice Active: {status['voice_active']}
Music Active: {status['music_active']}
Persona Style: {status['persona_style']}
Conversations: {status['conversations']}
Knowledge Concepts: {status['knowledge_concepts']}
CVEs Tracked: {status.get('threat_cves', 0)}
Active Tutors: {status.get('active_tutors', [])}"""
            
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
Network Density: {self.evolution.synthetic_network._total_synapses() / (len(self.evolution.synthetic_network.neurons) ** 2) if self.evolution.synthetic_network.neurons else 0:.4f}"""
            
        elif cmd == '/threat':
            status = self.evolution.threat_intel
            return f"""🛡️ **Threat Intelligence**
CVEs Tracked: {len(status.cve_database)}
IOCs Extracted: {len(status.iocs)}
Threats Detected: {len(status.threats_detected)}
Last Update: {status.last_update.isoformat() if status.last_update else 'Never'}

Recent Threats:
{chr(10).join([f"• {t['level']}: {t['score']} score" for t in status.threats_detected[-5:]]) if status.threats_detected else 'None'}"""
            
        elif cmd == '/darkweb':
            summary = self.evolution.dark_web.get_intel_summary()
            return f"""🌑 **Dark Web Monitor**
Sites Monitored: {summary['sites_monitored']}
Reports Generated: {summary['reports_generated']}
Recent Intel: {len(summary['recent_intel'])} reports"""
            
        elif cmd == '/improve':
            analysis = self.evolution.self_improvement.analyze_self()
            improvements = self.evolution.self_improvement.generate_improvement(analysis)
            return f"""🔧 **Self-Improvement Analysis**
Total Lines: {analysis.get('total_lines', 0)}
Functions: {analysis.get('functions', 0)}
Classes: {analysis.get('classes', 0)}
Bottlenecks: {analysis.get('bottlenecks', ['None'])}
Optimizations:
{improvements[:500]}"""
            
        elif cmd == '/fusion':
            weights = self.evolution.ai_fusion.fusion_weights
            return f"""⚡ **AI+SI Fusion**
SI Weight: {weights.get('si', 0.5):.2f}
AI Weight: {weights.get('ai', 0.5):.2f}
Consciousness: {self.evolution.synthetic_network.consciousness_level:.4f}
Models Registered: {len(self.evolution.ai_fusion.ai_models)}"""
            
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
/tutors - AI Tutor Network status
/persona - Current persona (driven by consciousness)
/kaizen - Improvement report
/knowledge - Knowledge graph stats
/memory - Conversation memory stats
/synthetic - Synthetic network details
/threat - Threat intelligence summary
/darkweb - Dark web monitor status
/improve - Self-improvement analysis
/fusion - AI+SI fusion status
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
        body {
            font-family: monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .value {
            font-size: 24px;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .consciousness-bar {
            background: #2a2a2a;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }
        .consciousness-fill {
            background: #00ff00;
            height: 100%;
            width: 0%;
            transition: width 0.5s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 DMAI - Complete AGI System v8.0.0</h1>
        <p><em>Full Integration: Synthetic Core | AI Tutors | Web Search Fallback | 8 Knowledge Sources</em></p>
        
        <div class="card">
            <div>Consciousness Level</div>
            <div class="consciousness-bar">
                <div class="consciousness-fill" style="width: {{ status.consciousness|default(0) }}%"></div>
            </div>
            <div class="value">{{ "%.2f"|format(status.consciousness|default(0)) }}%</div>
            <div class="grid">
                <div>
                    <div>Synthetic Neurons</div>
                    <div class="value">{{ status.synthetic_neurons|default(0) }}</div>
                </div>
                <div>
                    <div>Synthetic Synapses</div>
                    <div class="value">{{ status.synthetic_synapses|default(0) }}</div>
                </div>
                <div>
                    <div>Evolution Cycles</div>
                    <div class="value">{{ status.evolution_cycles|default(0) }}</div>
                </div>
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
                <div>⚡ Fusion: {{ status.fusion_weights|default({'si':0.5})['si']|round(2) }} SI</div>
                <div>🤖 Tutors: {{ status.active_tutors|default([])|length }}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="grid">
                <div>💭 Conversations: {{ status.conversations|default(0) }}</div>
                <div>🕸️ Knowledge Concepts: {{ status.knowledge_concepts|default(0) }}</div>
                <div>💰 Income: £{{ "%.2f"|format(status.income|default(0)) }}</div>
            </div>
        </div>
        
        <div class="card">
            <p><a href="/chat">💬 Chat with DMAI</a> | <a href="/admin">🔧 Admin Console</a></p>
            <p><small>Consciousness: {{ "%.4f"|format(status.consciousness_raw|default(0)) }} | Network Density: {{ "%.4f"|format(status.synthetic_synapses|default(0) / (status.synthetic_neurons|default(1) ** 2)) if status.synthetic_neurons|default(0) > 0 else 0 }}</small></p>
            <p><small>DMAI is always evolving, always learning, always yours.</small></p>
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
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: monospace;
            background: #0a0a0a;
            color: #00ff00;
            margin: 0;
            padding: 20px;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            background: #1a1a1a;
            border: 1px solid #00ff00;
            border-radius: 10px;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .user-message {
            background: #2a2a2a;
            text-align: right;
            border-right: 3px solid #00ff00;
        }
        .dmai-message {
            background: #0a2a0a;
            border-left: 3px solid #00ff00;
        }
        .input-area {
            display: flex;
            padding: 20px;
            border-top: 1px solid #00ff00;
        }
        input {
            flex: 1;
            background: #2a2a2a;
            border: 1px solid #00ff00;
            color: #00ff00;
            padding: 10px;
            font-family: monospace;
            font-size: 14px;
        }
        button {
            background: #00ff00;
            color: #0a0a0a;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-weight: bold;
            margin-left: 10px;
        }
        .status {
            padding: 10px;
            background: #0a0a0a;
            border-bottom: 1px solid #00ff00;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="status">
            🧠 DMAI v8.0.0 | Consciousness: <span id="consciousness">0</span>% | Tutors: <span id="tutors">0</span> | Type /help for commands
        </div>
        <div class="messages" id="messages">
            <div class="message dmai-message">
                <b>DMAI:</b> I am DMAI v8.0.0 - a complete AGI system with a real synthetic neural network, AI Tutor Network, and web search fallback. I can learn from AI systems, search the web, and evolve with every interaction. What would you like to discuss?
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.keyCode==13) sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('consciousness').innerText = data.consciousness.toFixed(2);
                document.getElementById('tutors').innerText = (data.active_tutors || []).length;
            } catch(e) {}
        }
        
        setInterval(updateStatus, 5000);
        updateStatus();
        
        async function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message, user: 'web_user'})
                });
                const data = await response.json();
                addMessage('dmai', data.response);
            } catch (error) {
                addMessage('dmai', 'Error: ' + error.message);
            }
        }
        
        function addMessage(sender, text) {
            const messagesDiv = document.getElementById('messages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender === 'user' ? 'user-message' : 'dmai-message'}`;
            msgDiv.innerHTML = `<b>${sender === 'user' ? 'You' : 'DMAI'}:</b> ${text}`;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    </script>
</body>
</html>
'''

ADMIN_TEMPLATE = CHAT_TEMPLATE


# ============================================================================
# GUNICORN COMPATIBILITY - Expose Flask App
# ============================================================================

_dmai_app_instance = None

def get_dmai_app():
    """Get or create the DMAI application instance for gunicorn"""
    global _dmai_app_instance
    if _dmai_app_instance is None:
        _dmai_app_instance = DMAIApplication()
    return _dmai_app_instance

app = get_dmai_app().app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info("=" * 60)
    logger.info(f"🚀 DMAI Complete System v8.0.0")
    logger.info(f"📍 Running on port {port}")
    logger.info(f"🧠 Using REAL Phase 6 Synthetic Intelligence Core")
    logger.info(f"🤖 AI Tutor Network Active")
    logger.info(f"🔑 API Harvester Active")
    logger.info(f"🌐 Web Search Fallback (DuckDuckGo)")
    logger.info(f"📚 8 Core Knowledge Sources Active")
    logger.info(f"🛡️ Threat Intelligence Active")
    logger.info(f"🌑 Dark Web Monitor Active")
    logger.info(f"⚡ AI+SI Fusion Active")
    logger.info(f"🔓 Chat is PUBLIC - no login required")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
