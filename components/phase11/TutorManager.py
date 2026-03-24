# components/phase11/TutorManager.py
"""
Manages the lifecycle of all AI tutors - tracks capabilities, performance, and when to discard.
ENHANCED: Added support for new tutors (Grok, HuggingFace, GitHub), better integration with AI Hub,
          performance metrics storage, and fixed discarded_tutors structure.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Tutor:
    """Represents an AI tutor with its capabilities and performance history"""
    
    def __init__(self, name: str, capabilities: List[str], api_endpoint: str = None, 
                 api_key_name: str = None, is_available: bool = False, model: str = None):
        self.name = name
        self.capabilities = capabilities
        self.api_endpoint = api_endpoint
        self.api_key_name = api_key_name
        self.model = model
        self.is_available = is_available
        self.added_date = datetime.now()
        self.performance_history = []  # List of (dma_quality, tutor_quality)
        self.query_count = 0
        self.successful_queries = 0
        self.avg_response_time = 0.0
        self.last_used = None
        
    def record_comparison(self, dma_quality: float, tutor_quality: float, response_time: float = None):
        """Record a performance comparison"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'dma_quality': dma_quality,
            'tutor_quality': tutor_quality,
            'response_time': response_time
        })
        self.query_count += 1
        if tutor_quality > 0:
            self.successful_queries += 1
            
        # Update average response time
        if response_time:
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (self.avg_response_time * (self.query_count - 1) + response_time) / self.query_count
                
        self.last_used = datetime.now()
            
    def get_average_performance(self) -> Dict:
        """Calculate average performance metrics"""
        if not self.performance_history:
            return {'dma_quality': 0, 'tutor_quality': 0, 'gap': 0, 'samples': 0, 'success_rate': 0}
        
        avg_dma = sum(p['dma_quality'] for p in self.performance_history) / len(self.performance_history)
        avg_tutor = sum(p['tutor_quality'] for p in self.performance_history) / len(self.performance_history)
        
        return {
            'dma_quality': avg_dma,
            'tutor_quality': avg_tutor,
            'gap': avg_dma - avg_tutor,
            'samples': len(self.performance_history),
            'success_rate': self.successful_queries / self.query_count if self.query_count > 0 else 0,
            'avg_response_time': self.avg_response_time,
            'total_queries': self.query_count
        }
    
    def get_trend(self) -> str:
        """Get performance trend direction"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent = self.performance_history[-3:]
        gaps = [p['dma_quality'] - p['tutor_quality'] for p in recent]
        
        if all(g > 0 for g in gaps):
            if gaps[-1] > gaps[0]:
                return "improving"
            return "stable"
        elif any(g < 0 for g in gaps):
            return "declining"
        return "stable"
    
    def to_dict(self):
        return {
            'name': self.name,
            'capabilities': self.capabilities,
            'api_endpoint': self.api_endpoint,
            'api_key_name': self.api_key_name,
            'model': self.model,
            'is_available': self.is_available,
            'added_date': self.added_date.isoformat(),
            'performance_history': self.performance_history[-20:],  # Last 20 for context
            'query_count': self.query_count,
            'successful_queries': self.successful_queries,
            'avg_response_time': self.avg_response_time,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


class TutorManager:
    """
    Manages the lifecycle of all AI tutors
    Features:
    - Track tutor performance over time
    - Determine when DMAI surpasses a tutor
    - Discard/archive surpassed tutors
    - Provide progress metrics for master
    """
    
    def __init__(self, data_path: str = None):
        self.active_tutors = {}  # name -> Tutor object
        self.discarded_tutors = []  # list of discarded tutor records (dicts)
        self.surpass_threshold = 0.9  # 90% quality gap to consider surpassed
        self.min_queries_for_comparison = 5  # Need at least 5 comparisons to discard
        self.data_path = data_path or 'data/phase11'
        self._load_state()
        
        # Initialize default tutors (including new ones)
        self._init_default_tutors()
        
    def _init_default_tutors(self):
        """Initialize known AI tutors that can be used (including new ones)"""
        default_tutors = [
            # Original tutors
            Tutor("OpenAI GPT-4", 
                  ["text_generation", "code", "analysis", "reasoning"],
                  "https://api.openai.com/v1/chat/completions",
                  "OPENAI_API_KEY",
                  False,
                  "gpt-4-turbo"),
            Tutor("DeepSeek",
                  ["text_generation", "reasoning", "code"],
                  "https://api.deepseek.com/v1/chat/completions",
                  "DEEPSEEK_API_KEY",
                  False,
                  "deepseek-chat"),
            Tutor("Google Gemini",
                  ["text_generation", "multimodal", "analysis"],
                  "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                  "GEMINI_API_KEY",
                  False,
                  "gemini-pro"),
            Tutor("Anthropic Claude",
                  ["text_generation", "safety", "analysis"],
                  "https://api.anthropic.com/v1/messages",
                  "ANTHROPIC_API_KEY",
                  False,
                  "claude-3-sonnet"),
            Tutor("Perplexity AI",
                  ["web_search", "citations", "research"],
                  "https://api.perplexity.ai/chat/completions",
                  "PERPLEXITY_API_KEY",
                  False,
                  "llama-3.1-sonar"),
            Tutor("Google AI Studio",
                  ["model_prototyping", "development"],
                  "https://generativelanguage.googleapis.com/v1beta",
                  "GOOGLE_AI_STUDIO_KEY",
                  False,
                  "gemini-pro"),
            
            # NEW: xAI Grok
            Tutor("xAI Grok",
                  ["text_generation", "real_time", "personality", "conversation"],
                  "https://api.x.ai/v1/chat/completions",
                  "XAI_API_KEY",
                  False,
                  "grok-1"),
            
            # NEW: HuggingFace
            Tutor("HuggingFace",
                  ["model_inference", "embeddings", "classification", "model_hub"],
                  "https://api-inference.huggingface.co/models/",
                  "HUGGINGFACE_API_KEY",
                  False,
                  "meta-llama/Llama-2-7b-chat-hf"),
            
            # NEW: GitHub
            Tutor("GitHub",
                  ["code_analysis", "repository_search", "trending", "version_control"],
                  "https://api.github.com",
                  "GITHUB_TOKEN_MAIN",
                  True,  # GitHub is always available (public rate limits)
                  "code_analysis"),
        ]
        
        for tutor in default_tutors:
            if tutor.name not in self.active_tutors:
                self.active_tutors[tutor.name] = tutor
                logger.debug(f"Initialized default tutor: {tutor.name}")
                
        logger.info(f"Initialized {len(self.active_tutors)} default tutors")
        
    def _save_state(self):
        """Save tutor state to disk"""
        try:
            os.makedirs(self.data_path, exist_ok=True)
            state = {
                'active_tutors': {name: tutor.to_dict() for name, tutor in self.active_tutors.items()},
                'discarded_tutors': self.discarded_tutors,
                'surpass_threshold': self.surpass_threshold,
                'min_queries_for_comparison': self.min_queries_for_comparison,
                'last_updated': datetime.now().isoformat()
            }
            with open(os.path.join(self.data_path, 'tutor_state.json'), 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tutor state: {e}")
            
    def _load_state(self):
        """Load tutor state from disk"""
        try:
            state_path = os.path.join(self.data_path, 'tutor_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    
                # Load active tutors
                for name, tutor_data in state.get('active_tutors', {}).items():
                    tutor = Tutor(
                        name=tutor_data['name'],
                        capabilities=tutor_data['capabilities'],
                        api_endpoint=tutor_data.get('api_endpoint'),
                        api_key_name=tutor_data.get('api_key_name'),
                        is_available=tutor_data.get('is_available', False),
                        model=tutor_data.get('model')
                    )
                    tutor.added_date = datetime.fromisoformat(tutor_data['added_date'])
                    tutor.performance_history = tutor_data.get('performance_history', [])
                    tutor.query_count = tutor_data.get('query_count', 0)
                    tutor.successful_queries = tutor_data.get('successful_queries', 0)
                    tutor.avg_response_time = tutor_data.get('avg_response_time', 0.0)
                    if tutor_data.get('last_used'):
                        tutor.last_used = datetime.fromisoformat(tutor_data['last_used'])
                    self.active_tutors[name] = tutor
                    
                self.discarded_tutors = state.get('discarded_tutors', [])
                self.surpass_threshold = state.get('surpass_threshold', 0.9)
                self.min_queries_for_comparison = state.get('min_queries_for_comparison', 5)
                
        except Exception as e:
            logger.error(f"Failed to load tutor state: {e}")
            
    def add_tutor(self, name: str, capabilities: List[str], api_endpoint: str = None,
                  api_key_name: str = None, is_available: bool = True, model: str = None):
        """Add a new tutor to learn from"""
        if name in self.active_tutors:
            logger.info(f"Tutor {name} already active")
            return
            
        # Check if it was discarded and reactivate
        for i, discarded in enumerate(self.discarded_tutors):
            if discarded.get('name') == name:
                logger.info(f"Tutor {name} was discarded, reactivating")
                self.discarded_tutors.pop(i)
                break
            
        tutor = Tutor(name, capabilities, api_endpoint, api_key_name, is_available, model)
        self.active_tutors[name] = tutor
        self._save_state()
        logger.info(f"Added new tutor: {name} with capabilities: {capabilities}")
        
    def discard_tutor(self, name: str, reason: str = "surpassed"):
        """Remove a tutor - DMAI has surpassed it"""
        if name not in self.active_tutors:
            logger.warning(f"Tutor {name} not active")
            return
            
        # Archive before discarding
        tutor = self.active_tutors[name]
        discard_record = {
            'name': name,
            'reason': reason,
            'discarded_date': datetime.now().isoformat(),
            'final_performance': tutor.get_average_performance(),
            'capabilities': tutor.capabilities,
            'total_queries': tutor.query_count,
            'was_available': tutor.is_available
        }
        self.discarded_tutors.append(discard_record)
        
        # Remove from active
        del self.active_tutors[name]
        self._save_state()
        logger.info(f"🎉 Discarded tutor {name} - DMAI has {reason}")
        
    def record_comparison(self, tutor_name: str, dma_quality: float, tutor_quality: float, response_time: float = None):
        """Track how DMAI compares to each tutor"""
        if tutor_name in self.active_tutors:
            self.active_tutors[tutor_name].record_comparison(dma_quality, tutor_quality, response_time)
            self._save_state()
        else:
            logger.debug(f"Tutor {tutor_name} not active, skipping comparison")
            
    def should_discard_tutor(self, tutor_name: str) -> Tuple[bool, str]:
        """
        Determine if DMAI has surpassed a tutor
        Returns: (should_discard, reason)
        """
        if tutor_name not in self.active_tutors:
            return False, "not_active"
            
        tutor = self.active_tutors[tutor_name]
        perf = tutor.get_average_performance()
        
        # Need enough samples for reliable comparison
        if perf['samples'] < self.min_queries_for_comparison:
            return False, f"insufficient_samples ({perf['samples']}/{self.min_queries_for_comparison})"
            
        # Check if DMAI consistently outperforms tutor
        if perf['gap'] > self.surpass_threshold:
            return True, f"DMAI quality ({perf['dma_quality']:.2f}) exceeds tutor ({perf['tutor_quality']:.2f}) by {perf['gap']:.2f}"
            
        return False, f"gap ({perf['gap']:.2f}) below threshold ({self.surpass_threshold})"
        
    def get_surpass_progress(self) -> Dict:
        """Show how close DMAI is to surpassing each tutor"""
        progress = {}
        for name, tutor in self.active_tutors.items():
            perf = tutor.get_average_performance()
            if perf['samples'] > 0:
                progress_percent = min(100, (perf['gap'] / self.surpass_threshold) * 100) if self.surpass_threshold > 0 else 0
                progress[name] = {
                    'dma_quality': perf['dma_quality'],
                    'tutor_quality': perf['tutor_quality'],
                    'gap': perf['gap'],
                    'threshold': self.surpass_threshold,
                    'progress_percent': max(0, progress_percent),
                    'samples': perf['samples'],
                    'success_rate': perf['success_rate'],
                    'avg_response_time': perf.get('avg_response_time', 0),
                    'total_queries': perf.get('total_queries', 0),
                    'trend': tutor.get_trend()
                }
            else:
                progress[name] = {
                    'status': 'not_yet_evaluated',
                    'capabilities': tutor.capabilities,
                    'is_available': tutor.is_available
                }
                
        return progress
        
    def get_active_tutors(self) -> List[str]:
        """Get list of active tutor names"""
        return list(self.active_tutors.keys())
        
    def get_active_tutors_with_status(self) -> List[Dict]:
        """Get list of active tutors with their status"""
        tutors = []
        for name, tutor in self.active_tutors.items():
            perf = tutor.get_average_performance()
            tutors.append({
                'name': name,
                'capabilities': tutor.capabilities,
                'is_available': tutor.is_available,
                'samples': perf['samples'],
                'success_rate': perf['success_rate'],
                'avg_response_time': perf['avg_response_time']
            })
        return tutors
        
    def get_tutor_capabilities(self, tutor_name: str) -> List[str]:
        """Get capabilities of a specific tutor"""
        if tutor_name in self.active_tutors:
            return self.active_tutors[tutor_name].capabilities
        return []
        
    def update_availability(self, tutor_name: str, is_available: bool):
        """Update whether a tutor's API is available"""
        if tutor_name in self.active_tutors:
            self.active_tutors[tutor_name].is_available = is_available
            self._save_state()
            
    def get_summary(self) -> Dict:
        """Get summary of all tutors - used by AIIntegrationHub"""
        active_tutors_list = []
        for name, tutor in self.active_tutors.items():
            perf = tutor.get_average_performance()
            active_tutors_list.append({
                'name': name,
                'capabilities': tutor.capabilities,
                'is_available': tutor.is_available,
                'samples': perf['samples'],
                'gap': perf['gap'],
                'success_rate': perf['success_rate']
            })
            
        return {
            'active_count': len(self.active_tutors),
            'discarded_count': len(self.discarded_tutors),
            'active_tutors': active_tutors_list,
            'discarded_tutors': [d.get('name') for d in self.discarded_tutors],
            'surpass_progress': self.get_surpass_progress(),
            'thresholds': {
                'surpass_threshold': self.surpass_threshold,
                'min_queries': self.min_queries_for_comparison
            }
        }
        
    def get_performance_history(self, tutor_name: str, limit: int = 20) -> List[Dict]:
        """Get performance history for a specific tutor"""
        if tutor_name in self.active_tutors:
            return self.active_tutors[tutor_name].performance_history[-limit:]
        return []
        
    def set_surpass_threshold(self, threshold: float):
        """Set the threshold for surpassing a tutor (0-1)"""
        self.surpass_threshold = max(0.0, min(1.0, threshold))
        self._save_state()
        logger.info(f"Surpass threshold set to {self.surpass_threshold}")
        
    def set_min_queries(self, min_queries: int):
        """Set minimum queries required before discarding"""
        self.min_queries_for_comparison = max(1, min_queries)
        self._save_state()
        
    def get_discarded_tutors(self) -> List[Dict]:
        """Get list of discarded tutors with their final stats"""
        return self.discarded_tutors


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("Tutor Manager Test")
    print("=" * 60)
    
    manager = TutorManager()
    
    print("\nActive Tutors:")
    for name, tutor in manager.active_tutors.items():
        print(f"  - {name}: {tutor.capabilities}")
        print(f"    API: {tutor.api_endpoint}")
        print(f"    Available: {tutor.is_available}")
    
    print("\nSummary:")
    print(json.dumps(manager.get_summary(), indent=2, default=str))
    
    print("\nSurpass Progress:")
    progress = manager.get_surpass_progress()
    for name, data in progress.items():
        if 'dma_quality' in data:
            print(f"  - {name}: {data['progress_percent']:.1f}% complete (gap: {data['gap']:.3f})")
        else:
            print(f"  - {name}: {data.get('status', 'unknown')}")
    
    print("\n✅ Tutor Manager ready")
