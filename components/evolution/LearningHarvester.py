# ============================================================================
# LEARNING HARVESTER
# ============================================================================
"""
Handles harvesting knowledge from multiple sources for DMAI's learning
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LearningHarvester:
    """
    Harvests knowledge from AI tutors, web, papers, and other sources
    """
    
    def __init__(self, data_path: Path, ai_hub, knowledge_graph):
        self.data_path = data_path
        self.ai_hub = ai_hub
        self.knowledge_graph = knowledge_graph
        self.harvest_dir = data_path / 'learning' / 'harvests'
        self.harvest_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🌾 LearningHarvester initialized")
    
    def harvest_from_ai_tutors(self, topic: str, stage: str) -> List[Dict]:
        """Harvest knowledge from AI tutors"""
        harvested = []
        
        if not self.ai_hub:
            return harvested
        
        prompt = f"""
DMAI is in the {stage} stage of development.
Learn about: {topic}

Provide specific, actionable knowledge that DMAI can directly use to improve herself.
Focus on implementation details, code examples, and practical applications.
"""
        
        try:
            result = self.ai_hub.query_all_tutors(prompt)
            if result.get('responses'):
                for tutor, response in result.get('responses', {}).items():
                    if response and isinstance(response, str) and len(response) > 50:
                        harvested.append({
                            'source': tutor,
                            'content': response[:2000],
                            'topic': topic,
                            'stage': stage,
                            'harvested_at': datetime.now().isoformat()
                        })
        except Exception as e:
            logger.error(f"AI tutor harvest failed: {e}")
        
        return harvested
    
    def harvest_from_web(self, topic: str) -> List[Dict]:
        """Harvest knowledge from web search"""
        # Placeholder - would integrate with web search API
        return []
    
    def harvest_from_papers(self, topic: str) -> List[Dict]:
        """Harvest from ArXiv and research papers"""
        # Placeholder - would integrate with ArXiv API
        return []
    
    def save_harvest(self, topic: str, harvested: List[Dict]) -> str:
        """Save harvested knowledge to disk"""
        harvest_file = self.harvest_dir / f"{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(harvest_file, 'w') as f:
            json.dump({
                'topic': topic,
                'harvested_at': datetime.now().isoformat(),
                'items': harvested
            }, f, indent=2)
        
        return str(harvest_file)
