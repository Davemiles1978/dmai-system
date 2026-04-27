"""
Response Quality Trainer - Teaches DMAI how to answer questions by studying
how other AI systems structure their responses.

Phase 1: Generate a Q&A dataset by extracting real knowledge topics from
DMAI's micro-neurons and querying external AIs for benchmark answers.
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseQualityTrainer:
    """Builds a training dataset of high-quality Q&A pairs from multiple AIs."""
    
    def __init__(self, db_path: str, ai_hub=None, data_dir: str = "data/training"):
        self.db_path = Path(db_path)
        self.ai_hub = ai_hub
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qa_pairs = []
        
    def extract_topics_from_micros(self, limit: int = 50) -> List[Dict]:
        """Extract meaningful topics from micro-level neurons across categories."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get diverse topics from article micros (richest content)
        cursor.execute("""
            SELECT i.insight_text, i.entity_type, i.source_title, i.source_url,
                   i.parent_macro_id, m.insight_text as macro_label
            FROM insights i
            LEFT JOIN insights m ON i.parent_macro_id = m.id
            WHERE i.neuron_level = 'micro'
              AND i.entity_type = 'article_micro'
              AND i.source_title IS NOT NULL
              AND i.source_title != ''
            ORDER BY i.created_at DESC
            LIMIT ?
        """, (limit,))
        
        topics = []
        for row in cursor.fetchall():
            # Extract the actual article title from the insight text
            insight_text = row[0]
            source_title = row[2] if row[2] else insight_text
            
            # Clean up the title
            if source_title.startswith("Article: "):
                source_title = source_title[9:]
            
            topics.append({
                'title': source_title[:200],
                'category': row[5] if row[5] else row[1],
                'source_url': row[3] if row[3] else '',
                'type': 'article'
            })
        
        # Add research paper topics
        cursor.execute("""
            SELECT i.insight_text, i.source_title, i.source_url
            FROM insights i
            WHERE i.entity_type = 'research_paper'
              AND i.source_title IS NOT NULL
            ORDER BY i.created_at DESC
            LIMIT 20
        """)
        
        for row in cursor.fetchall():
            title = row[1] if row[1] else row[0]
            if title.startswith("Paper: "):
                title = title[7:]
            topics.append({
                'title': title[:200],
                'category': 'Research Paper',
                'source_url': row[2] if row[2] else '',
                'type': 'paper'
            })
        
        # Add genealogy topics (AI system knowledge)
        cursor.execute("""
            SELECT v.version_name || ' - ' || s.name, s.name, v.architecture
            FROM system_versions v
            JOIN ai_systems s ON v.system_id = s.id
            ORDER BY v.release_date DESC
            LIMIT 15
        """)
        
        for row in cursor.fetchall():
            topics.append({
                'title': row[0][:200],
                'category': f'AI System: {row[1]}',
                'architecture': row[2] if row[2] else '',
                'type': 'ai_system'
            })
        
        conn.close()
        logger.info(f"Extracted {len(topics)} topics from micro-neurons and genealogy data")
        return topics
    
    def generate_questions_from_topics(self, topics: List[Dict]) -> List[Dict]:
        """Generate meaningful questions from extracted topics."""
        questions = []
        
        question_templates = [
            "Explain '{title}' and why it matters for AI development.",
            "What are the key concepts behind '{title}'?",
            "How does '{title}' relate to machine learning and artificial intelligence?",
            "What can DMAI learn from '{title}'?",
            "Summarize the main insights from '{title}' in simple terms.",
        ]
        
        for topic in topics[:30]:  # Generate 30 questions initially, expand later
            title = topic['title'][:150]
            if len(title) < 10:
                continue
                
            for template in question_templates[:2]:  # 2 questions per topic = 60 total
                q = template.replace('{title}', title)
                questions.append({
                    'question': q,
                    'topic': topic,
                    'category': topic.get('category', 'General'),
                })
        
        logger.info(f"Generated {len(questions)} questions from {len(topics)} topics")
        return questions
    
    def query_all_ais(self, questions: List[Dict]) -> List[Dict]:
        """Query multiple AI systems for each question and collect responses."""
        if not self.ai_hub:
            logger.error("No AI Hub available for querying")
            return []
        
        # AI tutors to query (thinking AIs only, not code search tools)
        ai_tutors = ['_query_openai', '_query_anthropic', '_query_gemini', '_query_deepseek']
        
        dataset = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question']
            logger.info(f"Querying AIs for question {i+1}/{len(questions)}: {question[:80]}...")
            
            qa_entry = {
                'question': question,
                'topic': q_data['topic']['title'],
                'category': q_data['category'],
                'answers': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for tutor_method_name in ai_tutors:
                try:
                    method = getattr(self.ai_hub, tutor_method_name, None)
                    if not method:
                        continue
                    
                    tutor_name = tutor_method_name.replace('_query_', '').replace('_', ' ').title()
                    result = method(question)
                    
                    if result.get('success') and result.get('response'):
                        qa_entry['answers'][tutor_name] = {
                            'response': result['response'][:1000],
                            'model': result.get('model', 'unknown')
                        }
                        logger.debug(f"  {tutor_name}: Got response ({len(result['response'])} chars)")
                    else:
                        qa_entry['answers'][tutor_name] = {
                            'error': result.get('error', 'No response')
                        }
                    
                    time.sleep(0.5)  # Rate limiting between tutors
                    
                except Exception as e:
                    qa_entry['answers'][tutor_method_name] = {'error': str(e)}
            
            dataset.append(qa_entry)
            
            # Save incrementally
            self._save_dataset(dataset)
            time.sleep(1)  # Rate limiting between questions
        
        return dataset
    
    def analyze_response_patterns(self, dataset: List[Dict]) -> Dict:
        """Analyze how different AIs structure their answers."""
        analysis = {
            'total_questions': len(dataset),
            'tutors_analyzed': set(),
            'avg_response_lengths': {},
            'common_structures': {},
            'sample_answers': {}
        }
        
        tutor_lengths = {}
        
        for entry in dataset:
            for tutor_name, answer_data in entry['answers'].items():
                analysis['tutors_analyzed'].add(tutor_name)
                
                if 'response' in answer_data:
                    response = answer_data['response']
                    if tutor_name not in tutor_lengths:
                        tutor_lengths[tutor_name] = []
                    tutor_lengths[tutor_name].append(len(response))
                    
                    # Collect sample answers (first 2 per tutor)
                    if tutor_name not in analysis['sample_answers']:
                        analysis['sample_answers'][tutor_name] = []
                    if len(analysis['sample_answers'][tutor_name]) < 2:
                        analysis['sample_answers'][tutor_name].append({
                            'question': entry['question'][:100],
                            'answer_preview': response[:300]
                        })
        
        # Calculate average response lengths
        for tutor, lengths in tutor_lengths.items():
            analysis['avg_response_lengths'][tutor] = {
                'avg': sum(lengths) / len(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'count': len(lengths)
            }
        
        analysis['tutors_analyzed'] = list(analysis['tutors_analyzed'])
        
        return analysis
    
    def _save_dataset(self, dataset: List[Dict]):
        """Save dataset to disk."""
        filepath = self.data_dir / 'qa_training_dataset.json'
        with open(filepath, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_entries': len(dataset),
                'dataset': dataset
            }, f, indent=2)
    
    def build_synthesis_prompt(self, dataset: List[Dict]) -> str:
        """Build a synthesis prompt template based on what we learned from other AIs."""
        analysis = self.analyze_response_patterns(dataset)
        
        prompt_template = f"""You are DMAI, an evolving AGI system. When answering questions, follow these guidelines learned from studying {len(analysis['tutors_analyzed'])} AI systems:

RESPONSE STRUCTURE:
1. Start with a clear, direct answer to the question
2. Provide context and background in 2-3 sentences
3. Include a concrete example or application
4. Connect to broader implications for AI/AGI development
5. End with an actionable takeaway or follow-up question

QUALITY STANDARDS (from benchmark analysis):
- Aim for {int(analysis['avg_response_lengths'].get('Openai', {}).get('avg', 300))}-{int(analysis['avg_response_lengths'].get('Anthropic', {}).get('avg', 500))} characters
- Use beginner-friendly language but don't oversimplify
- Cross-reference related DMAI knowledge domains when possible
- Be specific, not generic - use real examples from ingested research

KNOWLEDGE SYNTHESIS RULES:
- Draw from DMAI's micro-neuron knowledge base first
- Connect the topic to at least one other domain DMAI has studied
- If the answer requires external knowledge, acknowledge the boundary
- Never fabricate information - cite sources when available

This template was generated from {len(dataset)} Q&A pairs across {len(analysis['tutors_analyzed'])} AI systems.
"""
        
        return prompt_template
    
    def run_full_pipeline(self, num_topics: int = 30) -> Dict:
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING RESPONSE QUALITY TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Extract topics
        logger.info("Step 1: Extracting topics from micro-neurons...")
        topics = self.extract_topics_from_micros(limit=num_topics)
        
        # Step 2: Generate questions
        logger.info("Step 2: Generating questions...")
        questions = self.generate_questions_from_topics(topics)
        
        # Step 3: Query AIs
        logger.info("Step 3: Querying AI systems for benchmark answers...")
        dataset = self.query_all_ais(questions)
        
        # Step 4: Analyze patterns
        logger.info("Step 4: Analyzing response patterns...")
        analysis = self.analyze_response_patterns(dataset)
        
        # Step 5: Build synthesis prompt
        logger.info("Step 5: Building synthesis prompt template...")
        synthesis_prompt = self.build_synthesis_prompt(dataset)
        
        # Save everything
        self._save_dataset(dataset)
        
        with open(self.data_dir / 'response_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        with open(self.data_dir / 'synthesis_prompt_template.txt', 'w') as f:
            f.write(synthesis_prompt)
        
        logger.info(f"Pipeline complete: {len(dataset)} Q&A pairs generated")
        logger.info(f"Data saved to {self.data_dir}/")
        
        return {
            'topics_extracted': len(topics),
            'questions_generated': len(questions),
            'qa_pairs_created': len(dataset),
            'tutors_queried': analysis['tutors_analyzed'],
            'avg_response_lengths': analysis['avg_response_lengths'],
            'data_dir': str(self.data_dir)
        }


# Singleton for DMAI integration
_trainer_instance = None

def get_trainer(db_path: str = None, ai_hub=None) -> ResponseQualityTrainer:
    global _trainer_instance
    if _trainer_instance is None and db_path and ai_hub:
        _trainer_instance = ResponseQualityTrainer(db_path, ai_hub)
    return _trainer_instance
