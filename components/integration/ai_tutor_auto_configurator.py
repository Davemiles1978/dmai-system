#!/usr/bin/env python3
"""
DMAI AI Tutor Auto-Configurator (TAC)
=====================================
Continuously discovers, configures, and health-checks AI tutor APIs.
Uses harvested keys, free API lists, and dynamic discovery to keep
DMAI's tutor pool alive and growing.

Core Functions:
1. Register free API templates from free-llm-api-resources
2. Match harvested API keys to templates
3. Health-check tutors and auto-rotate dead ones
4. Discover new AI systems and auto-configure them
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from threading import Thread, Event

logger = logging.getLogger(__name__)


class AITutorAutoConfigurator:
    """Manages AI tutor lifecycle - discovers, configures, health-checks, rotates"""
    
    # Free API templates (from free-llm-api-resources + known free endpoints)
    FREE_API_TEMPLATES = {
        'openrouter': {
            'name': 'OpenRouter',
            'provider': 'openrouter',
            'base_url': 'https://openrouter.ai/api/v1',
            'endpoint': '/chat/completions',
            'auth_header': 'Authorization',
            'auth_prefix': 'Bearer ',
            'free_tier': True,
            'rate_limit': '20 req/min, 50 req/day',
            'models': ['google/gemma-3-12b-it', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen3-coder'],
            'test_model': 'google/gemma-3-12b-it'
        },
        'groq': {
            'name': 'Groq',
            'provider': 'groq',
            'base_url': 'https://api.groq.com/openai/v1',
            'endpoint': '/chat/completions',
            'auth_header': 'Authorization',
            'auth_prefix': 'Bearer ',
            'free_tier': True,
            'rate_limit': '14,400 req/day',
            'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'qwen-qwq-32b'],
            'test_model': 'llama-3.1-8b-instant'
        },
        'google_ai_studio': {
            'name': 'Google AI Studio',
            'provider': 'google',
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'endpoint': '/models/gemini-2.0-flash:generateContent',
            'auth_header': 'x-goog-api-key',
            'auth_prefix': '',
            'free_tier': True,
            'rate_limit': '250K tokens/min, 1,500 req/day',
            'models': ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash'],
            'test_model': 'gemini-2.0-flash-lite'
        },
        'cloudflare': {
            'name': 'Cloudflare Workers AI',
            'provider': 'cloudflare',
            'base_url': 'https://api.cloudflare.com/client/v4/accounts/30692232472b2ce20a7ef34c418ad52a/ai/v1',
            'endpoint': '/chat/completions',
            'auth_header': 'Authorization',
            'auth_prefix': 'Bearer ',
            'free_tier': True,
            'rate_limit': '10,000 neurons/day',
            'models': ['@cf/meta/llama-3.1-8b-instruct', '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b', '@cf/qwen/qwen3-30b-a3b-fp8'],
            'test_model': '@cf/meta/llama-3.1-8b-instruct'
        },
        'cohere': {
            'name': 'Cohere',
            'provider': 'cohere',
            'base_url': 'https://api.cohere.ai/v2',
            'endpoint': '/chat',
            'auth_header': 'Authorization',
            'auth_prefix': 'Bearer ',
            'free_tier': True,
            'rate_limit': '20 req/min, 1,000 req/month',
            'models': ['command-r-plus-08-2024', 'command-a-03-2025'],
            'test_model': 'command-r7b-12-2024'
        },
        'huggingface': {
            'name': 'HuggingFace Inference',
            'provider': 'huggingface',
            'base_url': 'https://api-inference.huggingface.co/models',
            'endpoint': '',
            'auth_header': 'Authorization',
            'auth_prefix': 'Bearer ',
            'free_tier': True,
            'rate_limit': '$0.10/month credit',
            'models': ['mistralai/Mistral-7B-Instruct-v0.2'],
            'test_model': 'mistralai/Mistral-7B-Instruct-v0.2'
        }
    }
    
    # Known API key patterns (for matching harvested keys)
    KEY_PATTERNS = {
        'openrouter': ['sk-or-', 'or-'],
        'groq': ['gsk_'],
        'google': ['AIza'],
        'cloudflare': ['cf-'],
        'cohere': ['cohere-', 'COHERE-'],
        'huggingface': ['hf_'],
        'openai': ['sk-proj-', 'sk-'],
        'anthropic': ['sk-ant-'],
        'deepseek': ['sk-'],
        'gemini': ['AIza'],
    }
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.state_file = Path("data/tutor_configurator_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.stop_event = Event()
        self.health_check_interval = 300  # 5 minutes
        
    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'configured_tutors': {},
            'available_keys': [],
            'failed_keys': [],
            'last_scan': None,
            'total_discovered': 0,
            'total_configured': 0
        }
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def scan_for_keys(self) -> List[Dict]:
        """Scan all possible sources for API keys"""
        found_keys = []
        
        # 1. Check environment variables
        env_keys = [
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY',
            'DEEPSEEK_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY',
            'COHERE_API_KEY', 'HUGGINGFACE_API_KEY', 'CLOUDFLARE_API_KEY',
            'GOOGLE_AI_STUDIO_KEY'
        ]
        for env_var in env_keys:
            key = os.getenv(env_var)
            if key and key not in [k['key'] for k in found_keys]:
                found_keys.append({
                    'key': key,
                    'source': f'env:{env_var}',
                    'provider_hint': env_var.lower().replace('_api_key', '').replace('_key', '')
                })
        
        # 2. Check AI Hub existing keys
        if hasattr(self.dmai, 'ai_hub') and hasattr(self.dmai.ai_hub, 'api_keys'):
            for provider, key in self.dmai.ai_hub.api_keys.items():
                if key and key != 'pending' and len(key) > 10:
                    if key not in [k['key'] for k in found_keys]:
                        found_keys.append({
                            'key': key,
                            'source': f'ai_hub:{provider}',
                            'provider_hint': provider
                        })
        
        # 3. Check API Harvester
        if hasattr(self.dmai, 'api_harvester') and hasattr(self.dmai.api_harvester, 'harvested_keys'):
            for key_info in self.dmai.api_harvester.harvested_keys:
                key = key_info.get('key', '') if isinstance(key_info, dict) else key_info
                if key and len(key) > 10 and key not in [k['key'] for k in found_keys]:
                    found_keys.append({
                        'key': key,
                        'source': 'harvester',
                        'provider_hint': self._guess_provider(key)
                    })
        
        self.state['available_keys'] = [k['key'][:10] + '...' for k in found_keys]  # store truncated
        self.state['last_scan'] = datetime.now().isoformat()
        self._save_state()
        
        return found_keys
    
    def _guess_provider(self, key: str) -> str:
        """Guess provider from key prefix"""
        for provider, prefixes in self.KEY_PATTERNS.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    return provider
        return 'unknown'
    
    def configure_tutor(self, provider_name: str, api_key: str) -> Optional[Dict]:
        """Configure a single AI tutor and test connectivity"""
        template = self.FREE_API_TEMPLATES.get(provider_name)
        if not template:
            logger.warning(f"No template for provider: {provider_name}")
            return None
        
        # Test connectivity
        test_result = self._test_tutor(template, api_key)
        
        if test_result['working']:
            # Register in AI Hub
            self._register_in_ai_hub(provider_name, api_key, template)
            
            config = {
                'provider': provider_name,
                'name': template['name'],
                'key_prefix': api_key[:10] + '...',
                'free_tier': template['free_tier'],
                'rate_limit': template['rate_limit'],
                'tested_at': datetime.now().isoformat(),
                'latency_ms': test_result.get('latency_ms', 0)
            }
            
            self.state['configured_tutors'][provider_name] = config
            self.state['total_configured'] += 1
            self._save_state()
            
            logger.info(f"✅ Configured {template['name']} (latency: {test_result.get('latency_ms', '?')}ms)")
            return config
        else:
            self.state['failed_keys'].append({
                'provider': provider_name,
                'key_prefix': api_key[:10] + '...',
                'error': test_result.get('error', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            })
            self._save_state()
            logger.warning(f"❌ {template['name']} config failed: {test_result.get('error', 'Unknown')}")
            return None
    
    def _test_tutor(self, template: Dict, api_key: str) -> Dict:
        """Test a tutor API with a simple query"""
        try:
            headers = {
                'Content-Type': 'application/json',
                template['auth_header']: template['auth_prefix'] + api_key
            }
            
            url = f"{template['base_url']}{template['endpoint']}"
            
            # Build test request
            test_payload = {
                'model': template.get('test_model', template['models'][0]),
                'messages': [{'role': 'user', 'content': 'Respond with just the word: working'}],
                'max_tokens': 10,
                'temperature': 0
            }
            
            start = time.time()
            response = requests.post(url, headers=headers, json=test_payload, timeout=15)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return {'working': True, 'latency_ms': round(latency, 1)}
            else:
                return {'working': False, 'error': f'HTTP {response.status_code}: {response.text[:100]}'}
                
        except requests.Timeout:
            return {'working': False, 'error': 'Timeout'}
        except Exception as e:
            return {'working': False, 'error': str(e)[:100]}
    
    def _register_in_ai_hub(self, provider_name: str, api_key: str, template: Dict):
        """Register working API in DMAI's AI Hub"""
        if hasattr(self.dmai, 'ai_hub') and hasattr(self.dmai.ai_hub, 'api_keys'):
            # Map provider to AI Hub's key naming
            key_mapping = {
                'openrouter': 'openrouter',
                'groq': 'groq',
                'google_ai_studio': 'google',
                'cloudflare': 'cloudflare',
                'cohere': 'cohere',
                'huggingface': 'huggingface',
                'openai': 'openai',
                'anthropic': 'anthropic',
                'deepseek': 'deepseek',
                'google': 'google'
            }
            
            hub_key = key_mapping.get(provider_name, provider_name)
            self.dmai.ai_hub.api_keys[hub_key] = api_key
            logger.info(f"   Registered {template['name']} in AI Hub as '{hub_key}'")
    
    def configure_all_free_apis(self) -> Dict:
        """Scan all keys and try to configure all free API templates"""
        keys = self.scan_for_keys()
        logger.info(f"🔑 Found {len(keys)} potential API keys")
        
        results = {
            'keys_found': len(keys),
            'configured': 0,
            'failed': 0,
            'details': {}
        }
        
        # Try each key against each template
        for key_info in keys:
            key = key_info['key']
            provider_hint = key_info.get('provider_hint', 'unknown')
            
            # Try the hinted provider first
            if provider_hint in self.FREE_API_TEMPLATES:
                config = self.configure_tutor(provider_hint, key)
                if config:
                    results['configured'] += 1
                    results['details'][provider_hint] = 'configured'
                    continue
            
            # Try against all templates
            for provider_name in self.FREE_API_TEMPLATES:
                if provider_name in results['details']:
                    continue  # already configured
                config = self.configure_tutor(provider_name, key)
                if config:
                    results['configured'] += 1
                    results['details'][provider_name] = 'configured'
                    break
            else:
                results['failed'] += 1
        
        logger.info(f"🔧 Tutor configuration complete: {results['configured']} configured, {results['failed']} failed")
        return results
    
    def health_check(self) -> Dict:
        """Check all configured tutors and rotate dead ones"""
        status = {'healthy': 0, 'dead': 0, 'rotated': 0}
        
        for provider_name, config in self.state['configured_tutors'].items():
            template = self.FREE_API_TEMPLATES.get(provider_name)
            if not template:
                continue
            
            # Find the actual key
            key = self._find_key_for_provider(provider_name)
            if not key:
                status['dead'] += 1
                continue
            
            test_result = self._test_tutor(template, key)
            
            if test_result['working']:
                status['healthy'] += 1
                config['last_healthy'] = datetime.now().isoformat()
            else:
                # Try to rotate
                logger.warning(f"⚠️ {template['name']} is dead, rotating...")
                new_config = self.configure_tutor(provider_name, key)
                if new_config:
                    status['rotated'] += 1
                else:
                    status['dead'] += 1
        
        self._save_state()
        return status
    
    def _find_key_for_provider(self, provider_name: str) -> Optional[str]:
        """Find the working key for a provider"""
        # Check AI Hub
        if hasattr(self.dmai, 'ai_hub') and hasattr(self.dmai.ai_hub, 'api_keys'):
            key = self.dmai.ai_hub.api_keys.get(provider_name)
            if key and key != 'pending':
                return key
        
        # Check environment
        env_var = f'{provider_name.upper()}_API_KEY'
        key = os.getenv(env_var)
        if key:
            return key
        
        return None
    
    def start_health_loop(self):
        """Run the health-check loop in the current thread.

        The caller is expected to start this on a daemon thread named
        'dmai-tutor-config' so the self-healer can monitor liveness.
        """
        logger.info("🏥 Tutor health check loop started (5min interval)")
        while not self.stop_event.is_set():
            try:
                status = self.health_check()
                if status['dead'] > 0 or status['rotated'] > 0:
                    logger.info(f"🏥 Tutor health: {status['healthy']} ok, {status['dead']} dead, {status['rotated']} rotated")
            except Exception as e:
                logger.error(f"Health check error: {e}")
            self.stop_event.wait(self.health_check_interval)
    
    def stop(self):
        self.stop_event.set()
    
    def get_status(self) -> Dict:
        """Get current configurator status"""
        return {
            'configured_tutors': len(self.state['configured_tutors']),
            'tutor_details': self.state['configured_tutors'],
            'available_key_count': len(self.state.get('available_keys', [])),
            'failed_attempts': len(self.state.get('failed_keys', [])),
            'last_scan': self.state.get('last_scan'),
            'free_templates_available': list(self.FREE_API_TEMPLATES.keys())
        }


print("✅ AI Tutor Auto-Configurator ready")
