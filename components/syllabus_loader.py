#!/usr/bin/env python3
"""
SyllabusLoader - Creates macro/micro neurons from the DMAI Evolutionary Learning Syllabus
Total: 140 topics across 5 stages (Baby, Toddler, Child, Teen, Adult)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SyllabusLoader:
    """Loads syllabus topics into SI Core as macro neurons with micro neuron seeds"""
    
    def __init__(self, si_core, data_dir: str = "data"):
        self.si_core = si_core
        self.data_dir = Path(data_dir)
        self.syllabus_path = self.data_dir / "syllabus_topics.json"
        self.state_path = self.data_dir / "syllabus_loader_state.json"
        self.topics_processed = 0
        self.macros_created = 0
        self.micros_created = 0
        self.synapses_created = 0
        
    def load_syllabus(self) -> Optional[Dict]:
        """Load the syllabus JSON file"""
        if not self.syllabus_path.exists():
            logger.error(f"Syllabus file not found: {self.syllabus_path}")
            return None
        
        with open(self.syllabus_path, 'r') as f:
            return json.load(f)
    
    def get_processed_topics(self) -> set:
        """Get set of already processed topic IDs"""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                    return set(state.get('processed_topics', []))
            except:
                pass
        return set()
    
    def save_processed_topic(self, topic_id: str):
        """Mark a topic as processed"""
        processed = self.get_processed_topics()
        processed.add(topic_id)
        
        state = {
            'processed_topics': list(processed),
            'last_updated': datetime.now().isoformat(),
            'total_processed': len(processed),
            'macros_created': self.macros_created,
            'micros_created': self.micros_created,
            'synapses_created': self.synapses_created
        }
        
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_micro_seeds(self, topic: str, category: str) -> List[str]:
        """Generate micro neuron seeds for a topic"""
        
        seeds_by_category = {
            "Core": [
                f"Core principles of {topic}",
                f"Key concepts in {topic}",
                f"Fundamental techniques for {topic}",
                f"Common applications of {topic}",
                f"Best practices for {topic}"
            ],
            "Artistic": [
                f"Aesthetic foundations of {topic}",
                f"Creative techniques in {topic}",
                f"Style variations in {topic}",
                f"Emotional expression through {topic}",
                f"Composition principles for {topic}"
            ],
            "Wealth": [
                f"Revenue models for {topic}",
                f"Market analysis for {topic}",
                f"Monetization strategies for {topic}",
                f"Risk management in {topic}",
                f"Scaling techniques for {topic}"
            ],
            "Reverse": [
                f"Analysis methodologies for {topic}",
                f"Deconstruction techniques for {topic}",
                f"Pattern identification in {topic}",
                f"System mapping for {topic}",
                f"Extraction methods for {topic}"
            ],
            "Accelerator": [
                f"Optimization strategies for {topic}",
                f"Growth acceleration via {topic}",
                f"Efficiency gains through {topic}",
                f"Evolution mechanisms in {topic}",
                f"Consciousness impact of {topic}"
            ]
        }
        
        return seeds_by_category.get(category, [f"Introduction to {topic}", f"Advanced concepts in {topic}"])[:5]
    
    def process_topic(self, topic_data: Dict, all_topics: List[Dict]) -> Optional[str]:
        """Create macro neuron and seed micro neurons for a single topic"""
        
        topic = topic_data['topic']
        category = topic_data['category']
        stage = topic_data['stage']
        why_important = topic_data.get('why_important', '')
        
        # Create MACRO neuron
        try:
            macro_id = self.si_core.add_insight(
                insight_text=f"[{stage}] {topic}: {why_important}",
                entity_type="syllabus_topic",
                entities=[topic, category, stage],
                relationship="foundational_knowledge",
                source_topic="syllabus",
                target_topic=category.lower(),
                confidence=1.0,
                source_title=f"DMAI Evolutionary Syllabus - {stage} Stage",
                source_type="syllabus_loader",
                neuron_level='macro',
                is_visible_at_top_level=True
            )
        except Exception as e:
            logger.error(f"Failed to create macro neuron for {topic}: {e}")
            return None
        
        if not macro_id:
            logger.warning(f"No macro ID returned for: {topic}")
            return None
        
        self.macros_created += 1
        logger.info(f"📚 Created macro neuron: {topic} ({category})")
        
        # Create MICRO neurons (seeds)
        micro_seeds = self.get_micro_seeds(topic, category)
        for seed in micro_seeds:
            try:
                micro_id = self.si_core.add_insight(
                    insight_text=f"{topic} - {seed}",
                    entity_type="syllabus_micro",
                    entities=[topic, seed[:30]],
                    relationship="detailed_knowledge",
                    source_topic=topic,
                    target_topic=category.lower(),
                    confidence=0.8,
                    source_type="syllabus_loader_seed",
                    neuron_level='micro',
                    cluster_id=macro_id,
                    parent_macro_id=macro_id,
                    is_visible_at_top_level=False
                )
                if micro_id:
                    self.micros_created += 1
            except Exception as e:
                logger.debug(f"Failed to create micro neuron for seed '{seed}': {e}")
        
        return macro_id
    
    def process_all_topics(self, limit_per_stage: int = None) -> Dict:
        """Process all unprocessed syllabus topics"""
        
        syllabus = self.load_syllabus()
        if not syllabus:
            return {'error': 'Syllabus not found'}
        
        processed = self.get_processed_topics()
        all_topics = syllabus['all_topics']
        
        stages = ['Baby', 'Toddler', 'Child', 'Teen', 'Adult']
        
        for stage in stages:
            stage_topics = [t for t in all_topics if t['stage'] == stage]
            stage_topics.sort(key=lambda x: x['priority'])
            
            if limit_per_stage:
                stage_topics = stage_topics[:limit_per_stage]
            
            for topic_data in stage_topics:
                topic_id = topic_data['id']
                
                if topic_id in processed:
                    continue
                
                macro_id = self.process_topic(topic_data, all_topics)
                
                if macro_id:
                    self.save_processed_topic(topic_id)
                    self.topics_processed += 1
        
        return {
            'topics_processed': self.topics_processed,
            'macros_created': self.macros_created,
            'micros_created': self.micros_created,
            'synapses_created': self.synapses_created,
            'remaining': len(all_topics) - len(processed)
        }
    
    def get_status(self) -> Dict:
        """Get current loader status"""
        syllabus = self.load_syllabus()
        processed = self.get_processed_topics()
        
        if syllabus:
            total = len(syllabus['all_topics'])
            return {
                'total_topics': total,
                'processed': len(processed),
                'remaining': total - len(processed),
                'progress': f"{(len(processed)/total)*100:.1f}%" if total > 0 else "0%",
                'macros_created': self.macros_created,
                'micros_created': self.micros_created,
                'synapses_created': self.synapses_created
            }
        
        return {'error': 'Syllabus not loaded'}


def initialize_syllabus(si_core, data_dir: str = "data", process_limit: int = None):
    """
    Initialize syllabus loader and process topics.
    
    Args:
        si_core: SyntheticIntelligenceCore instance
        data_dir: Data directory path
        process_limit: Max topics to process per stage (None = all)
    
    Returns:
        Dict with processing results
    """
    loader = SyllabusLoader(si_core, data_dir)
    result = loader.process_all_topics(limit_per_stage=process_limit)
    logger.info(f"✅ Syllabus initialization complete: {result}")
    return result


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dmai_core_complete import SyntheticIntelligenceCore
    
    # Initialize SI Core
    si_core = SyntheticIntelligenceCore(data_dir="data/synthetic")
    
    # Initialize syllabus (process 3 topics per stage for testing)
    result = initialize_syllabus(si_core, data_dir="data", process_limit=3)
    
    print(json.dumps(result, indent=2))
