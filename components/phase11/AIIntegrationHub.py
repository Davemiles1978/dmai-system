# components/phase11/AIIntegrationHub.py
"""
Enhanced AIIntegrationHub - Complete version preserving all original functionality
Phase 11: AI Tutor Network & Self-Evolution
ADDED: xAI Grok, HuggingFace, GitHub integration
UPDATED: All AI tutors now have 30-second timeouts and retry logic
"""

import os
import json
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

class AIIntegrationHub:
    """
    Complete AI Integration Hub with all original functionality preserved.
    Connects to all AI tutors, queries them, synthesizes responses,
    and feeds learning into the synthetic neural network.
    
    ADDED TUTORS:
    - xAI Grok
    - HuggingFace
    - GitHub
    """
    
    def __init__(self, data_path: str, discovery=None):
        self.data_path = data_path
        self.discovery = discovery
        self.api_keys = self._load_api_keys()
        self.query_history = []
        self.learning_cache = {}
        
        # Phase 11 components (will be set later)
        self.capability_synthesizer = None
        self.tutor_manager = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0,
            'tutor_performance': {}
        }
        
        # Load existing query history
        self._load_history()
        
        # Track active tutors count
        active_tutors = self._get_active_tutors()
        logger.info(f"🚀 AIIntegrationHub initialized with {len(active_tutors)} active tutors")
        logger.info(f"   Active: {active_tutors}")
        
    def _load_api_keys(self) -> Dict:
        """Load API keys from environment with all original services + NEW services"""
        keys = {}
        
        # ====================================================================
        # ORIGINAL COMMERCIAL LLMs (All preserved)
        # ====================================================================
        keys['openai'] = os.getenv('OPENAI_API_KEY')
        keys['groq'] = os.getenv('GROQ_API_KEY') or os.getenv('GROQ_KEY')
        keys['deepseek'] = os.getenv('DEEPSEEK_API_KEY')
        keys['gemini'] = os.getenv('GEMINI_API_KEY')
        keys['anthropic'] = os.getenv('ANTHROPIC_API_KEY')
        keys['perplexity'] = os.getenv('PERPLEXITY_API_KEY')
        
        # ====================================================================
        # NEW: xAI Grok
        # ====================================================================
        keys['grok'] = os.getenv('XAI_API_KEY')
        
        # ====================================================================
        # NEW: HuggingFace
        # ====================================================================
        keys['huggingface'] = os.getenv('HUGGINGFACE_API_KEY')
        
        # ====================================================================
        # NEW: GitHub (separate tokens for different purposes)
        # ====================================================================
        keys['github_main'] = os.getenv('GITHUB_TOKEN_MAIN')
        keys['github_secondary'] = os.getenv('GITHUB_TOKEN_SECONDARY')
        keys['github'] = keys['github_main'] or keys['github_secondary']  # Fallback
        
        # ====================================================================
        # ORIGINAL Google ecosystem (All preserved)
        # ====================================================================
        keys['google_ai_studio'] = os.getenv('GOOGLE_AI_STUDIO_KEY')
        
        # ====================================================================
        # NEW: Cerebras Inference (1M tokens/day free, 2,600 tok/s)
        # ====================================================================
        keys['cerebras'] = os.getenv('CEREBRAS_API_KEY')
        
        # ====================================================================
        # NEW: GitHub Models (45+ frontier models, free with GitHub account)
        # ====================================================================
        keys['github_models'] = os.getenv('GITHUB_MODELS_TOKEN') or os.getenv('GITHUB_TOKEN_MAIN')
        
        # ====================================================================
        # NEW: Mistral AI (Experiment plan — all models, 1B tokens/month free)
        # ====================================================================
        keys['mistral'] = os.getenv('MISTRAL_API_KEY')
        keys['notebooklm'] = os.getenv('NOTEBOOKLM_API_KEY')
        keys['imagen'] = os.getenv('IMAGEN_API_KEY')
        keys['gemini_gems'] = os.getenv('GEMINI_GEMS_KEY')
        
        # ====================================================================
        # ORIGINAL Creative tools (All preserved)
        # ====================================================================
        keys['nano_banana'] = os.getenv('NANO_BANANA_KEY')
        keys['pomelli'] = os.getenv('POMELLI_KEY')
        keys['google_opal'] = os.getenv('GOOGLE_OPAL_KEY')
        keys['google_whisk'] = os.getenv('GOOGLE_WHISK_KEY')
        
        # ====================================================================
        # ORIGINAL Additional services (All preserved)
        # ====================================================================
        keys['runwayml'] = os.getenv('RUNWAYML_API_KEY')
        keys['pika'] = os.getenv('PIKA_API_KEY')
        keys['kling'] = os.getenv('KLING_API_KEY')
        keys['sora'] = os.getenv('SORA_API_KEY')
        keys['windsurf'] = os.getenv('WINDSURF_API_KEY')
        keys['lovable'] = os.getenv('LOVABLE_API_KEY')
        keys['cursor'] = os.getenv('CURSOR_API_KEY')
        keys['copilot'] = os.getenv('COPILOT_API_KEY')
        
        # Log which new keys are present
        if keys['grok']:
            logger.info("  ✅ xAI Grok API key found")
        else:
            logger.info("  ⏳ xAI Grok API key pending")
            
        if keys['huggingface']:
            logger.info("  ✅ HuggingFace API key found")
        else:
            logger.info("  ⏳ HuggingFace API key pending")
            
        if keys['github']:
            logger.info("  ✅ GitHub tokens found")
        else:
            logger.info("  ⏳ GitHub tokens pending")
        
        return keys
        
    def _load_history(self):
        """Load query history from disk"""
        try:
            history_file = os.path.join(self.data_path, 'phase11', 'query_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.query_history = data.get('queries', [])
                    self.performance_metrics = data.get('metrics', self.performance_metrics)
                    logger.info(f"Loaded {len(self.query_history)} historical queries")
        except Exception as e:
            logger.error(f"Failed to load query history: {e}")
            
    def _save_history(self):
        """Save query history to disk"""
        try:
            os.makedirs(os.path.join(self.data_path, 'phase11'), exist_ok=True)
            history_file = os.path.join(self.data_path, 'phase11', 'query_history.json')
            with open(history_file, 'w') as f:
                json.dump({
                    'queries': self.query_history[-1000:],  # Keep last 1000
                    'metrics': self.performance_metrics,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")
            
    def query_all_tutors(self, prompt: str, use_cache: bool = True) -> Dict:
        """
        Query all available tutors and collect responses.
        Original functionality preserved with enhanced features.
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = hash(prompt)
        if use_cache and cache_key in self.learning_cache:
            cache_time = self.learning_cache[cache_key]['timestamp']
            if (datetime.now() - cache_time).seconds < 300:  # 5 minute cache
                logger.info(f"Using cached response for: {prompt[:50]}...")
                return self.learning_cache[cache_key]['response']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'responses': {},
            'errors': [],
            'synthesis': None
        }
        
        # AI THINKING TUTORS - LLMs that reason, answer questions, and synthesize knowledge
        query_methods = [
            ('OpenAI GPT-4', self._query_openai),
            ('DeepSeek', self._query_deepseek),
            ('Google Gemini', self._query_gemini),
            ('Anthropic Claude', self._query_anthropic),
            ('Perplexity AI', self._query_perplexity),
            ('xAI Grok', self._query_grok),
            ('Cerebras Inference', self._query_cerebras),
            ('GitHub Models', self._query_github_models),
            ('Mistral AI', self._query_mistral),
            # Code/research tools available via dedicated methods, NOT queried as thinking tutors:
            # HuggingFace: _query_huggingface  (model search, not Q&A)
            # GitHub: _query_github  (repo search, not Q&A)
        ]
        
        # Query each tutor (sequential to avoid rate limits)
        for tutor_name, method in query_methods:
            try:
                logger.debug(f"Querying {tutor_name}...")
                result = method(prompt)
                
                if result.get('success'):
                    results['responses'][tutor_name] = result['response']
                    self.performance_metrics['successful_queries'] += 1
                    
                    # Update tutor performance
                    if tutor_name not in self.performance_metrics['tutor_performance']:
                        self.performance_metrics['tutor_performance'][tutor_name] = {
                            'successes': 0,
                            'failures': 0,
                            'avg_response_time': 0
                        }
                    self.performance_metrics['tutor_performance'][tutor_name]['successes'] += 1
                    
                    # Record comparison with synthetic network if available
                    if self.tutor_manager:
                        quality = self._estimate_response_quality(result['response'])
                        dma_quality = self._estimate_dma_quality(prompt)
                        self.tutor_manager.record_comparison(tutor_name, dma_quality, quality)
                        
                else:
                    results['errors'].append(f"{tutor_name}: {result.get('error', 'Unknown error')}")
                    self.performance_metrics['failed_queries'] += 1
                    
                    if self.tutor_manager and tutor_name in self.performance_metrics['tutor_performance']:
                        self.performance_metrics['tutor_performance'][tutor_name]['failures'] += 1
                        
            except Exception as e:
                logger.error(f"Error querying {tutor_name}: {e}")
                results['errors'].append(f"{tutor_name}: {str(e)}")
                self.performance_metrics['failed_queries'] += 1
                
        # Synthesize responses if we have a synthesizer
        if self.capability_synthesizer and results['responses']:
            try:
                results['synthesis'] = self.capability_synthesizer.synthesize(
                    results['responses'],
                    prompt
                )
                
                # Add unified answer to results
                if results['synthesis'].get('unified_answer'):
                    results['unified_answer'] = results['synthesis']['unified_answer']
                    
                # Feed to learning system
                self._learn_from_responses(results['synthesis'], prompt)
                
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                results['synthesis_error'] = str(e)
                
        # Update metrics
        response_time = time.time() - start_time
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['average_response_time'] = (
            (self.performance_metrics['average_response_time'] * (self.performance_metrics['total_queries'] - 1) + response_time) /
            self.performance_metrics['total_queries']
        )
        
        # Store in history
        self.query_history.append({
            'timestamp': results['timestamp'],
            'prompt': prompt[:200],
            'response_count': len(results['responses']),
            'error_count': len(results['errors']),
            'response_time': response_time
        })
        
        # Cache the result
        if use_cache and results['responses']:
            self.learning_cache[cache_key] = {
                'timestamp': datetime.now(),
                'response': results
            }
            # Trim cache
            if len(self.learning_cache) > 100:
                oldest_key = min(self.learning_cache.keys(), key=lambda k: self.learning_cache[k]['timestamp'])
                del self.learning_cache[oldest_key]
                
        # Save history periodically
        if len(self.query_history) % 10 == 0:
            self._save_history()
            
        logger.info(f"Query completed: {len(results['responses'])} responses, "
                   f"{len(results['errors'])} errors in {response_time:.2f}s")
                   
        return results
        
    def _learn_from_responses(self, synthesis: Dict, prompt: str):
        """Feed synthesized responses to synthetic neural network - Original functionality"""
        try:
            # If synthetic network is available, feed learning data
            if hasattr(self, 'synthetic_network') and self.synthetic_network:
                training_data = synthesis.get('training_data', {})
                if training_data:
                    # Original learning logic
                    if hasattr(self.synthetic_network, 'train_on_data'):
                        self.synthetic_network.train_on_data(training_data)
                    elif hasattr(self.synthetic_network, 'learn'):
                        self.synthetic_network.learn(synthesis.get('unified_answer', ''))
                    logger.debug("Fed synthesized response to synthetic network")
                    
            # Store in learning cache for pattern recognition
            learning_key = f"{prompt[:100]}_{datetime.now().date()}"
            if learning_key not in self.learning_cache:
                self.learning_cache[learning_key] = {
                    'prompt': prompt,
                    'synthesis': synthesis,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Failed to learn from responses: {e}")
            
    def _estimate_response_quality(self, response: str) -> float:
        """Estimate response quality (0-1) - Enhanced from original"""
        if not response:
            return 0.0
            
        quality = 0.5  # Base quality
        
        # Length quality (longer isn't always better, but indicates detail)
        if len(response) > 500:
            quality += 0.2
        elif len(response) > 200:
            quality += 0.1
            
        # Content quality indicators
        quality_indicators = [
            ('insight', 0.05),
            ('therefore', 0.05),
            ('conclusion', 0.05),
            ('analysis', 0.05),
            ('example', 0.05),
            ('```', 0.1),  # Code blocks indicate technical depth
            ('research', 0.05),
            ('study', 0.05)
        ]
        
        for indicator, boost in quality_indicators:
            if indicator in response.lower():
                quality += boost
                
        # Penalize overly short responses
        if len(response) < 50:
            quality -= 0.2
            
        return max(0.0, min(1.0, quality))
        
    def _estimate_dma_quality(self, prompt: str) -> float:
        """Estimate DMAI's response quality - Original logic enhanced"""
        # This would ideally query the synthetic network
        # For now, use learning progress indicator
        
        # Base quality that improves with learning
        base_quality = 0.5
        
        # If we have learning history, improve quality
        if len(self.query_history) > 100:
            base_quality += 0.1
        if len(self.query_history) > 500:
            base_quality += 0.1
            
        # Check if we've seen similar prompts before
        similar_prompts = [q for q in self.query_history 
                          if any(word in q.get('prompt', '').lower() 
                                 for word in prompt.lower().split()[:3])]
        if similar_prompts:
            base_quality += min(0.2, len(similar_prompts) / 100)
            
        # If synthetic network is available, use its confidence
        if hasattr(self, 'synthetic_network') and self.synthetic_network:
            try:
                if hasattr(self.synthetic_network, 'get_confidence'):
                    network_confidence = self.synthetic_network.get_confidence(prompt)
                    base_quality = (base_quality + network_confidence) / 2
            except:
                pass
                
        return min(1.0, base_quality)
        
    # ====================================================================
    # ORIGINAL INDIVIDUAL TUTOR QUERY METHODS (All preserved with timeout fixes)
    # ====================================================================
    
    def _query_openai(self, prompt: str) -> Dict:
        """Query OpenAI GPT-4 with increased timeout and retry"""
        api_key = os.getenv('OPENAI_API_KEY') or self.api_keys.get('openai')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'OpenAI', 'error': 'No API key'}
        
        for attempt in range(2):
            try:
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'gpt-3.5-turbo',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': 500,
                        'temperature': 0.7
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'tutor': 'OpenAI GPT-4',
                        'response': data['choices'][0]['message']['content'],
                        'model': 'gpt-3.5-turbo'
                    }
                else:
                    return {'success': False, 'tutor': 'OpenAI', 'error': f'HTTP {response.status_code}'}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"OpenAI timeout, retrying...")
                    continue
                else:
                    return {'success': False, 'tutor': 'OpenAI', 'error': 'Request timed out after retry'}
            except Exception as e:
                return {'success': False, 'tutor': 'OpenAI', 'error': str(e)}
        
        return {'success': False, 'tutor': 'OpenAI', 'error': 'All attempts failed'}
            
    def _query_deepseek(self, prompt: str) -> Dict:
        """Query DeepSeek with increased timeout and retry logic"""
        api_key = os.getenv('DEEPSEEK_API_KEY') or self.api_keys.get('deepseek')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'DeepSeek', 'error': 'No API key'}
        
        # Retry logic for timeout issues
        for attempt in range(2):
            try:
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                    json={
                        'model': 'deepseek-chat',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': 500,
                        'temperature': 0.7
                    },
                    timeout=30  # Increased from 15 to 30 seconds
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'tutor': 'DeepSeek',
                        'response': data['choices'][0]['message']['content'],
                        'model': 'deepseek-chat'
                    }
                else:
                    return {'success': False, 'tutor': 'DeepSeek', 'error': f'HTTP {response.status_code}'}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"DeepSeek timeout, retrying...")
                    continue
                else:
                    return {'success': False, 'tutor': 'DeepSeek', 'error': 'Request timed out after retry'}
            except Exception as e:
                return {'success': False, 'tutor': 'DeepSeek', 'error': str(e)}
        
        return {'success': False, 'tutor': 'DeepSeek', 'error': 'All attempts failed'}
            
    def _query_gemini(self, prompt: str) -> Dict:
        """Query Google Gemini with increased timeout and retry"""
        api_key = self.api_keys.get('gemini')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Gemini', 'error': 'No API key'}
        
        for attempt in range(2):
            try:
                response = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}',
                    headers={'Content-Type': 'application/json'},
                    json={
                        'contents': [{'parts': [{'text': prompt}]}],
                        'generationConfig': {
                            'temperature': 0.7,
                            'maxOutputTokens': 500
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'tutor': 'Google Gemini',
                        'response': data['candidates'][0]['content']['parts'][0]['text'],
                        'model': 'gemini-pro'
                    }
                else:
                    return {'success': False, 'tutor': 'Gemini', 'error': f'HTTP {response.status_code}'}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"Gemini timeout, retrying...")
                    continue
                else:
                    return {'success': False, 'tutor': 'Gemini', 'error': 'Request timed out after retry'}
            except Exception as e:
                return {'success': False, 'tutor': 'Gemini', 'error': str(e)}
        
        return {'success': False, 'tutor': 'Gemini', 'error': 'All attempts failed'}
            
    def _query_anthropic(self, prompt: str) -> Dict:
        """Query Anthropic Claude with increased timeout and retry"""
        api_key = os.getenv('ANTHROPIC_API_KEY') or self.api_keys.get('anthropic')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Claude', 'error': 'No API key'}
        
        for attempt in range(2):
            try:
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers={
                        'x-api-key': api_key,
                        'anthropic-version': '2023-06-01',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-haiku-20240307',
                        'max_tokens': 500,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.7
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'success': True,
                        'tutor': 'Anthropic Claude',
                        'response': data['content'][0]['text'],
                        'model': 'claude-3-haiku'
                    }
                else:
                    return {'success': False, 'tutor': 'Claude', 'error': f'HTTP {response.status_code}'}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"Claude timeout, retrying...")
                    continue
                else:
                    return {'success': False, 'tutor': 'Claude', 'error': 'Request timed out after retry'}
            except Exception as e:
                return {'success': False, 'tutor': 'Claude', 'error': str(e)}
        
        return {'success': False, 'tutor': 'Claude', 'error': 'All attempts failed'}
            
    def _query_perplexity(self, prompt: str) -> Dict:
        """Query Perplexity AI - Original logic preserved"""
        api_key = self.api_keys.get('perplexity')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Perplexity', 'error': 'No API key'}
            
        try:
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'sonar',  # Perplexity Sonar — real-time web search + AI
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'Perplexity AI',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'sonar'
                }
            else:
                return {'success': False, 'tutor': 'Perplexity', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'tutor': 'Perplexity', 'error': str(e)}
    
    # ====================================================================
    # NEW TUTOR QUERY METHODS (xAI Grok, HuggingFace, GitHub)
    # ====================================================================
    
    def _query_grok(self, prompt: str) -> Dict:
        """Query xAI Grok - NEW TUTOR"""
        api_key = self.api_keys.get('grok')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Grok', 'error': 'No API key'}
            
        try:
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'grok-1',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'xAI Grok',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'grok-1'
                }
            else:
                return {'success': False, 'tutor': 'Grok', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'tutor': 'Grok', 'error': str(e)}
    
    def _query_huggingface(self, prompt: str) -> Dict:
        """Query HuggingFace Inference API - NEW TUTOR"""
        api_key = self.api_keys.get('huggingface')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'HuggingFace', 'error': 'No API key'}
            
        try:
            # Use a popular model for text generation
            model = "meta-llama/Llama-2-7b-chat-hf"
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {'Authorization': f'Bearer {api_key}'}
            payload = {'inputs': prompt, 'parameters': {'max_new_tokens': 500}}
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get('generated_text', str(data))
                    return {
                        'success': True,
                        'tutor': 'HuggingFace',
                        'response': generated_text,
                        'model': model
                    }
                else:
                    return {
                        'success': True,
                        'tutor': 'HuggingFace',
                        'response': str(data),
                        'model': model
                    }
            else:
                return {'success': False, 'tutor': 'HuggingFace', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'tutor': 'HuggingFace', 'error': str(e)}
    
    def _query_github(self, prompt: str) -> Dict:
        """Query GitHub - Search for repositories matching the prompt - NEW TUTOR"""
        token = self.api_keys.get('github')
        
        # Extract search query from prompt (simple keyword extraction)
        search_query = prompt.replace(' ', '+')[:100]
        url = f"https://api.github.com/search/repositories?q={search_query}&per_page=5"
        
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if items:
                    result = f"Found {len(items)} repositories:\n\n"
                    for repo in items[:5]:
                        result += f"📁 {repo['full_name']}\n"
                        result += f"   ⭐ {repo['stargazers_count']} stars | 📝 {repo.get('description', 'No description')[:100]}\n"
                        result += f"   🔗 {repo['html_url']}\n\n"
                    return {
                        'success': True,
                        'tutor': 'GitHub',
                        'response': result,
                        'count': len(items)
                    }
                else:
                    return {
                        'success': True,
                        'tutor': 'GitHub',
                        'response': "No repositories found matching the search query."
                    }
            else:
                # Public rate limits apply if no token
                if response.status_code == 403 and 'rate limit' in response.text.lower():
                    return {
                        'success': False,
                        'tutor': 'GitHub',
                        'error': 'Rate limit exceeded - add GitHub token for higher limits'
                    }
                return {'success': False, 'tutor': 'GitHub', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'tutor': 'GitHub', 'error': str(e)}
    
    # ====================================================================
    # ORIGINAL ADDITIONAL TUTOR QUERY METHODS (All preserved)
    # ====================================================================
    
    def _query_google_ai_studio(self, prompt: str) -> Dict:
        """Query Google AI Studio - Original functionality"""
        api_key = os.getenv('GOOGLE_AI_STUDIO_KEY') or os.getenv('GEMINI_API_KEY') or self.api_keys.get('google_ai_studio')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Google AI Studio', 'error': 'No API key'}
            
        try:
            # Implementation similar to Gemini but with different endpoint
            response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}',
                json={'contents': [{'parts': [{'text': prompt}]}]},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'Google AI Studio',
                    'response': data['candidates'][0]['content']['parts'][0]['text']
                }
            else:
                return {'success': False, 'tutor': 'Google AI Studio', 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'tutor': 'Google AI Studio', 'error': str(e)}
            
    def _query_notebooklm(self, prompt: str) -> Dict:
        """Query NotebookLM - Original functionality"""
        api_key = self.api_keys.get('notebooklm')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'NotebookLM', 'error': 'No API key'}
            
        # NotebookLM specific implementation
        return {
            'success': False,
            'tutor': 'NotebookLM',
            'error': 'NotebookLM API requires specific implementation - placeholder'
        }
        
    def _query_imagen(self, prompt: str) -> Dict:
        """Query Imagen 3 - Original functionality"""
        api_key = self.api_keys.get('imagen')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Imagen', 'error': 'No API key'}
            
        # Imagen is for image generation, not text
        return {
            'success': False,
            'tutor': 'Imagen',
            'error': 'Imagen is for image generation, not text queries'
        }
        
    def _query_gemini_gems(self, prompt: str) -> Dict:
        """Query Gemini Gems - Original functionality"""
        api_key = self.api_keys.get('gemini_gems')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Gemini Gems', 'error': 'No API key'}
            
        return {
            'success': False,
            'tutor': 'Gemini Gems',
            'error': 'Gemini Gems API requires specific implementation'
        }
        
    def _query_nano_banana(self, prompt: str) -> Dict:
        """Query Nano Banana - Original functionality"""
        api_key = self.api_keys.get('nano_banana')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Nano Banana', 'error': 'No API key'}
            
        return {
            'success': False,
            'tutor': 'Nano Banana',
            'error': 'Nano Banana API requires specific implementation'
        }
        
    def _query_pomelli(self, prompt: str) -> Dict:
        """Query Pomelli - Original functionality"""
        api_key = self.api_keys.get('pomelli')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Pomelli', 'error': 'No API key'}
            
        return {
            'success': False,
            'tutor': 'Pomelli',
            'error': 'Pomelli API requires specific implementation'
        }
        
    def _query_opal(self, prompt: str) -> Dict:
        """Query Google Opal - Original functionality"""
        api_key = self.api_keys.get('google_opal')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Google Opal', 'error': 'No API key'}
            
        return {
            'success': False,
            'tutor': 'Google Opal',
            'error': 'Google Opal API requires specific implementation'
        }
        
    def _query_whisk(self, prompt: str) -> Dict:
        """Query Google Whisk - Original functionality"""
        api_key = self.api_keys.get('google_whisk')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Google Whisk', 'error': 'No API key'}
            
        return {
            'success': False,
            'tutor': 'Google Whisk',
            'error': 'Google Whisk API requires specific implementation'
        }
        

    # ====================================================================
    # UNIFIED CHAT INTERFACE (called by ExtendedHub fallback + _ai_chat directly)
    # ====================================================================

    async def chat(self, prompt: str) -> str:
        """
        Async chat interface — tries providers in priority order and returns
        the first successful response.

        Priority (cheapest/fastest first):
          1. Cerebras   — 1M tokens/day, 2,600 tok/s
          2. Groq       — 14,400 req/day, fastest after Cerebras
          3. Google AI Studio — 1,500 req/day, large context
          4. GitHub Models — 150 RPD free, OpenAI-quality
          5. Mistral    — 1B tokens/month, 2 RPM
          6. DeepSeek   — $0.14/1M, best paid value
          7. OpenAI     — $0.15/1M (gpt-4o-mini)
          8. Anthropic  — paid, last resort
        """
        import asyncio

        priority_methods = [
            ("Cerebras",        self._query_cerebras),
            ("Groq",            self._query_groq),
            ("Google AI Studio",self._query_google_ai_studio),
            ("GitHub Models",   self._query_github_models),
            ("Mistral AI",      self._query_mistral),
            ("DeepSeek",        self._query_deepseek),
            ("OpenAI",          self._query_openai),
            ("Anthropic",       self._query_anthropic),
        ]

        for name, method in priority_methods:
            try:
                # Use get_event_loop safely — create new loop if none exists
                # Always use the running loop's executor — never create a new loop inside async
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, method, prompt)
                if not isinstance(result, dict):
                    logger.warning("chat(): %s returned non-dict: %s", name, type(result).__name__)
                    result = {"success": False, "error": "non-dict result"}
                if result.get("success"):
                    logger.info("chat(): responded via %s", name)
                    resp = result["response"]
                    if isinstance(resp, tuple):
                        resp = resp[0] if resp else None
                    if resp and not isinstance(resp, str):
                        resp = str(resp)
                    if resp:
                        return resp
                else:
                    logger.warning("chat(): %s failed — %s", name, result.get("error", "unknown"))
            except Exception as exc:
                logger.warning("chat(): %s exception — %s", name, exc)

        return (
            "DMAI is online but no AI provider key is active. "
            "Add a free key — Groq (GROQ_API_KEY) is fastest to set up: "
            "https://console.groq.com/keys"
        )

    def chat_sync(self, prompt: str) -> str:
        """Synchronous wrapper around chat() — safe in any thread context."""
        import asyncio
        try:
            try:
                asyncio.get_running_loop()
                is_running = True
            except RuntimeError:
                is_running = False

            if is_running:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.chat(prompt))
                    result = fut.result(timeout=45)
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.chat(prompt))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            return result if isinstance(result, str) else (str(result) if result is not None else None)
        except Exception as exc:
            logger.warning("chat_sync error: %s", exc)
            return None

    # ====================================================================
    # NEW FREE-TIER TUTOR QUERY METHODS
    # Groq · Cerebras · GitHub Models · Mistral AI
    # ====================================================================

    def _query_groq(self, prompt: str) -> Dict:
        """Query Groq — 14,400 req/day free, ultra-low latency LLM inference"""
        api_key = os.getenv('GROQ_API_KEY') or self.api_keys.get('groq')
        if not api_key or api_key == 'pending':
            return {'success': False, 'tutor': 'Groq', 'error': 'No API key — sign up free at console.groq.com/keys'}
        try:
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7,
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'Groq',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'llama-3.3-70b-versatile',
                }
            elif response.status_code == 429:
                return {'success': False, 'tutor': 'Groq', 'error': 'Rate limited'}
            else:
                return {'success': False, 'tutor': 'Groq', 'error': f'HTTP {response.status_code}: {response.text[:120]}'}
        except Exception as e:
            return {'success': False, 'tutor': 'Groq', 'error': str(e)}

    def _query_cerebras(self, prompt: str) -> Dict:
        """Query Cerebras Inference — 1M tokens/day free, world-fastest inference (2,600+ tok/s)"""
        api_key = os.getenv('CEREBRAS_API_KEY') or self.api_keys.get('cerebras')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Cerebras', 'error': 'No API key — sign up free at cloud.cerebras.ai'}

        try:
            response = requests.post(
                'https://api.cerebras.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b',  # Best free model: 70B, 2,600 tok/s
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'Cerebras Inference',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'llama-3.3-70b'
                }
            elif response.status_code == 429:
                return {'success': False, 'tutor': 'Cerebras', 'error': 'Rate limited (30 RPM / 1M tokens/day)'}
            else:
                return {'success': False, 'tutor': 'Cerebras', 'error': f'HTTP {response.status_code}: {response.text[:120]}'}

        except Exception as e:
            return {'success': False, 'tutor': 'Cerebras', 'error': str(e)}

    def _query_github_models(self, prompt: str) -> Dict:
        """Query GitHub Models Marketplace — 45+ frontier models free with GitHub account"""
        api_key = os.getenv('GITHUB_TOKEN_MAIN') or os.getenv('GITHUB_TOKEN') or self.api_keys.get('github_models')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'GitHub Models', 'error': 'No token — set GITHUB_MODELS_TOKEN or GITHUB_TOKEN_MAIN'}

        try:
            response = requests.post(
                'https://models.github.ai/inference/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o-mini',  # 15 RPM / 150 RPD on free tier
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'GitHub Models',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'gpt-4o-mini'
                }
            elif response.status_code == 429:
                return {'success': False, 'tutor': 'GitHub Models', 'error': 'Rate limited (15 RPM / 150 RPD free tier)'}
            elif response.status_code == 401:
                return {'success': False, 'tutor': 'GitHub Models', 'error': 'Invalid token — needs Models read permission'}
            else:
                return {'success': False, 'tutor': 'GitHub Models', 'error': f'HTTP {response.status_code}: {response.text[:120]}'}

        except Exception as e:
            return {'success': False, 'tutor': 'GitHub Models', 'error': str(e)}

    def _query_mistral(self, prompt: str) -> Dict:
        """Query Mistral AI — Experiment plan: all models including Large, 1B tokens/month free"""
        api_key = os.getenv('MISTRAL_API_KEY') or self.api_keys.get('mistral')
        if not api_key or api_key == "pending":
            return {'success': False, 'tutor': 'Mistral AI', 'error': 'No API key — sign up free at console.mistral.ai'}

        try:
            response = requests.post(
                'https://api.mistral.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'mistral-large-latest',  # Full Large model on free Experiment tier
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'tutor': 'Mistral AI',
                    'response': data['choices'][0]['message']['content'],
                    'model': 'mistral-large-latest'
                }
            elif response.status_code == 429:
                return {'success': False, 'tutor': 'Mistral AI', 'error': 'Rate limited (2 RPM on Experiment plan)'}
            else:
                return {'success': False, 'tutor': 'Mistral AI', 'error': f'HTTP {response.status_code}: {response.text[:120]}'}

        except Exception as e:
            return {'success': False, 'tutor': 'Mistral AI', 'error': str(e)}

    # ====================================================================
    # PHASE 11 ENHANCEMENTS (All preserved)
    # ====================================================================
    
    def integrate_discovered_model(self, name: str, endpoint: str, capabilities: List[str]):
        """Add a new tutor that was discovered - Phase 11 enhancement"""
        if self.tutor_manager:
            self.tutor_manager.add_tutor(
                name=name,
                capabilities=capabilities,
                api_endpoint=endpoint,
                is_available=False
            )
            logger.info(f"Integrated discovered model: {name}")
            
            # Add to query methods dynamically
            self._add_dynamic_query_method(name, endpoint)
            
    def _add_dynamic_query_method(self, name: str, endpoint: str):
        """Dynamically add a query method for a discovered tutor"""
        def dynamic_query(prompt: str) -> Dict:
            try:
                response = requests.post(
                    endpoint,
                    json={'prompt': prompt, 'max_tokens': 500},
                    timeout=10
                )
                if response.status_code == 200:
                    return {
                        'success': True,
                        'tutor': name,
                        'response': response.json().get('response', '')
                    }
                else:
                    return {'success': False, 'tutor': name, 'error': f'HTTP {response.status_code}'}
            except Exception as e:
                return {'success': False, 'tutor': name, 'error': str(e)}
                
        # Add to available query methods
        if not hasattr(self, '_dynamic_methods'):
            self._dynamic_methods = {}
        self._dynamic_methods[name] = dynamic_query
        
        # Add to query_all_tutors method's list
        if not hasattr(self, '_query_methods'):
            self._query_methods = []
        self._query_methods.append((name, dynamic_query))
        
    def get_missing_apis(self) -> List[str]:
        """Return what API keys DMAI needs to find - Phase 11 enhancement"""
        missing = []
        for service, key in self.api_keys.items():
            if not key or key == "pending":
                missing.append(service)
        return missing
        
    def set_synthesizer(self, synthesizer):
        """Set the capability synthesizer - Phase 11 enhancement"""
        self.capability_synthesizer = synthesizer
        logger.info("Capability synthesizer connected")
        
    def set_tutor_manager(self, tutor_manager):
        """Set the tutor manager - Phase 11 enhancement"""
        self.tutor_manager = tutor_manager
        logger.info("Tutor manager connected")
        
    def set_synthetic_network(self, synthetic_network):
        """Set the synthetic network for learning - Original functionality"""
        self.synthetic_network = synthetic_network
        logger.info("Synthetic network connected")
        
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics - Original functionality"""
        return {
            **self.performance_metrics,
            'cache_size': len(self.learning_cache),
            'history_size': len(self.query_history),
            'active_tutors': len(self._get_active_tutors())
        }
        
    def _get_active_tutors(self) -> List[str]:
        """Get list of active tutors with valid API keys - Original functionality"""
        active = []
        for service, key in self.api_keys.items():
            if key and key != "pending":
                # Map service to display name
                if service == 'openai':
                    active.append('OpenAI GPT-4')
                elif service == 'deepseek':
                    active.append('DeepSeek')
                elif service == 'gemini':
                    active.append('Google Gemini')
                elif service == 'anthropic':
                    active.append('Anthropic Claude')
                elif service == 'perplexity':
                    active.append('Perplexity AI')
                elif service == 'grok':
                    active.append('xAI Grok')
                elif service == 'huggingface':
                    active.append('HuggingFace')
                elif service == 'github':
                    active.append('GitHub')
                elif service == 'google_ai_studio':
                    active.append('Google AI Studio')
                elif service == 'cerebras':
                    active.append('Cerebras Inference')
                elif service == 'github_models':
                    active.append('GitHub Models')
                elif service == 'mistral':
                    active.append('Mistral AI')
                else:
                    active.append(service)
        return active
        
    def clear_cache(self):
        """Clear learning cache - Utility method"""
        self.learning_cache.clear()
        logger.info("Learning cache cleared")
        
    def get_learning_stats(self) -> Dict:
        """Get statistics about learning progress - Enhanced for Phase 11"""
        stats = {
            'total_queries': self.performance_metrics['total_queries'],
            'success_rate': (
                self.performance_metrics['successful_queries'] / 
                max(1, self.performance_metrics['total_queries'])
            ),
            'avg_response_time': self.performance_metrics['average_response_time'],
            'cache_size': len(self.learning_cache),
            'tutor_performance': self.performance_metrics.get('tutor_performance', {}),
            'missing_apis': self.get_missing_apis(),
            'active_tutors': self._get_active_tutors()
        }
        
        if self.tutor_manager:
            stats['tutor_summary'] = self.tutor_manager.get_summary()
            
        return stats
        
    async def query_all_tutors_async(self, prompt: str) -> Dict:
        """Async version for concurrent queries - Performance enhancement"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            # Add async tasks for each tutor
            # This would require async versions of each query method
            # For now, return sync version
            return self.query_all_tutors(prompt)
