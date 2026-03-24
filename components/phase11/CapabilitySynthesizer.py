# components/phase11/CapabilitySynthesizer.py
"""
Synthesizes new capabilities from multiple tutor responses.
Enhanced with concept extraction, pattern detection, and novel insight generation.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class CapabilitySynthesizer:
    """
    Takes responses from multiple tutors and creates new insights.
    Features:
    - Unified answer synthesis
    - Concept extraction with confidence scoring
    - Pattern detection across responses
    - Novel insight generation
    - Knowledge gap identification
    - Training data preparation for SI network
    """
    
    def __init__(self):
        self.synthesis_history = []
        
    def synthesize(self, responses: Dict, prompt: str) -> Dict:
        """
        Combine multiple tutor responses into:
        - Unified knowledge
        - Cross-reference validation
        - Novel insights
        - Training data for SI network
        
        Args:
            responses: Dict mapping tutor names to response strings or response dicts
            prompt: Original prompt that generated these responses
            
        Returns:
            Dict containing synthesized knowledge
        """
        if not responses:
            return {
                'success': False,
                'unified_answer': "No responses from tutors available.",
                'insights': [],
                'gaps': [],
                'confidence': 0.0,
                'extracted_concepts': {},
                'patterns': [],
                'novel_insights': [],
                'training_data': None
            }
            
        # Extract text responses (handle both string and dict formats)
        text_responses = {}
        errors = []
        
        for tutor, response in responses.items():
            if isinstance(response, dict):
                if response.get('success') and response.get('response'):
                    text_responses[tutor] = response['response']
                elif response.get('error'):
                    errors.append(f"{tutor}: {response['error']}")
            elif isinstance(response, str):
                text_responses[tutor] = response
            else:
                text_responses[tutor] = str(response)
                
        if not text_responses:
            return {
                'success': False,
                'unified_answer': "All tutors failed to respond.",
                'insights': [],
                'gaps': errors,
                'confidence': 0.0,
                'extracted_concepts': {},
                'patterns': [],
                'novel_insights': [],
                'training_data': None,
                'errors': errors
            }
            
        # ====================================================================
        # NEW: Extract concepts from all responses
        # ====================================================================
        extracted_concepts = self._extract_concepts_from_all(text_responses)
        
        # ====================================================================
        # NEW: Detect patterns across responses
        # ====================================================================
        patterns = self._extract_patterns(text_responses)
        
        # ====================================================================
        # ORIGINAL: Extract best patterns from each tutor
        # ====================================================================
        best_patterns = self.extract_best_patterns(text_responses)
        
        # ====================================================================
        # ORIGINAL: Identify gaps in knowledge
        # ====================================================================
        gaps = self.identify_gaps(text_responses, prompt)
        
        # ====================================================================
        # NEW: Find novel insights (information unique to a response)
        # ====================================================================
        novel_insights = self._find_novel_insights(text_responses)
        
        # ====================================================================
        # ORIGINAL: Create unified answer
        # ====================================================================
        unified_answer = self._create_unified_answer(text_responses, prompt)
        
        # ====================================================================
        # ORIGINAL: Calculate confidence based on agreement
        # ====================================================================
        confidence = self._calculate_confidence(text_responses)
        
        # ====================================================================
        # ORIGINAL: Create training data for SI network
        # ====================================================================
        training_data = self.create_training_data({
            'prompt': prompt,
            'responses': text_responses,
            'unified_answer': unified_answer,
            'insights': novel_insights,
            'extracted_concepts': extracted_concepts,
            'patterns': patterns
        })
        
        # ====================================================================
        # Build complete result (ORIGINAL fields + NEW fields)
        # ====================================================================
        result = {
            'success': True,
            'unified_answer': unified_answer,
            'best_patterns': best_patterns,
            'gaps': gaps,
            'insights': novel_insights,  # Original field name
            'novel_insights': novel_insights,  # New field name for clarity
            'confidence': confidence,
            'training_data': training_data,
            'extracted_concepts': extracted_concepts,
            'patterns': patterns,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
        
        self.synthesis_history.append(result)
        
        # Keep history manageable
        if len(self.synthesis_history) > 100:
            self.synthesis_history = self.synthesis_history[-100:]
            
        return result
    
    # ====================================================================
    # NEW METHODS (Added to enhance functionality)
    # ====================================================================
    
    def _extract_concepts_from_all(self, responses: Dict[str, str]) -> Dict[str, Dict]:
        """
        Extract key concepts from all responses with confidence scores.
        
        Returns:
            Dict mapping concept names to metadata including occurrences, sources, confidence
        """
        concepts = {}
        
        for tutor, response in responses.items():
            # Extract capitalized terms (potential proper nouns/concepts)
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
            # Extract quoted terms (explicitly named concepts)
            quoted = re.findall(r'"([^"]+)"', response)
            # Extract technical terms (words with numbers or special patterns)
            technical = re.findall(r'\b[a-z]+[0-9]+\b|\b[0-9]+[a-z]+\b', response.lower())
            # Extract terms in parentheses (often definitions)
            parenthetical = re.findall(r'\(([^)]+)\)', response)
            
            all_terms = set(capitalized + quoted + technical + parenthetical)
            
            for term in all_terms:
                if len(term) > 3:  # Ignore short terms
                    if term not in concepts:
                        concepts[term] = {
                            "occurrences": 0,
                            "sources": [],
                            "confidence": 0.0,
                            "insights": []
                        }
                    
                    concepts[term]["occurrences"] += 1
                    concepts[term]["sources"].append(tutor)
                    
                    # Extract context around term (50 chars before and after)
                    context_match = re.search(rf'.{{0,50}}{re.escape(term)}.{{0,50}}', response, re.IGNORECASE)
                    if context_match:
                        concepts[term]["insights"].append(context_match.group())
        
        # Calculate confidence based on occurrences across tutors
        total_tutors = len(responses)
        for term in concepts:
            concepts[term]["confidence"] = min(1.0, concepts[term]["occurrences"] / max(1, total_tutors))
            # Limit insights to 3 per concept
            concepts[term]["insights"] = concepts[term]["insights"][:3]
        
        return concepts
    
    def _extract_patterns(self, responses: Dict[str, str]) -> List[Dict]:
        """
        Extract common patterns across multiple responses.
        
        Returns:
            List of pattern dictionaries with type, term, frequency
        """
        patterns = []
        
        # Collect all sentences from all responses
        all_sentences = []
        for tutor, response in responses.items():
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Only consider substantial sentences
                    all_sentences.append((tutor, sentence))
        
        # Find common words across responses
        common_words = Counter()
        for tutor, sentence in all_sentences:
            words = set(re.findall(r'\b[a-z]{4,}\b', sentence.lower()))
            common_words.update(words)
        
        # Extract patterns from common words
        for word, count in common_words.most_common(10):
            if count > 1:
                patterns.append({
                    "type": "common_term",
                    "term": word,
                    "frequency": count,
                    "occurrences_across_tutors": count
                })
        
        # Check for common phrases (simple 2-word phrases)
        all_phrases = []
        for tutor, sentence in all_sentences:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                all_phrases.append(phrase)
        
        common_phrases = Counter(all_phrases)
        for phrase, count in common_phrases.most_common(5):
            if count > 1 and len(phrase) > 10:
                patterns.append({
                    "type": "common_phrase",
                    "term": phrase,
                    "frequency": count,
                    "occurrences_across_tutors": count
                })
        
        return patterns
    
    def _find_novel_insights(self, responses: Dict[str, str]) -> List[Dict]:
        """
        Find insights that are unique to a single tutor response.
        
        Returns:
            List of insight dictionaries with insight text, source, and confidence
        """
        novel_insights = []
        
        # Extract key sentences from each response
        for tutor, response in responses.items():
            sentences = re.split(r'[.!?]+', response)
            # Filter for substantial sentences (20-200 chars, not too short or long)
            significant_sentences = [
                s.strip() for s in sentences 
                if 30 < len(s.strip()) < 300 and 
                not s.strip().startswith("I don't know") and
                not s.strip().startswith("I cannot")
            ]
            
            for sentence in significant_sentences:
                # Check if this idea appears in other responses
                appears_elsewhere = False
                for other_tutor, other_response in responses.items():
                    if other_tutor != tutor:
                        # Calculate word overlap
                        words = set(sentence.lower().split())
                        other_words = set(other_response.lower().split())
                        overlap = len(words & other_words) / max(1, len(words))
                        
                        # If more than 30% word overlap, likely similar idea
                        if overlap > 0.3:
                            appears_elsewhere = True
                            break
                
                if not appears_elsewhere:
                    novel_insights.append({
                        "insight": sentence,
                        "source": tutor,
                        "confidence": 0.7  # Base confidence for novel insights
                    })
        
        # Limit to top 10 insights
        return novel_insights[:10]
    
    # ====================================================================
    # ORIGINAL METHODS (Preserved and enhanced)
    # ====================================================================
    
    def _create_unified_answer(self, responses: Dict[str, str], prompt: str) -> str:
        """Create a coherent unified answer from multiple responses"""
        if not responses:
            return "No responses available to synthesize."
        
        # Find the most comprehensive response
        best_response = max(responses.values(), key=lambda x: len(x.split()))
        
        # If multiple responses, combine key points
        if len(responses) > 1:
            key_points = []
            seen_points = set()
            
            for tutor, response in responses.items():
                # Extract key sentences (simple heuristic)
                sentences = re.split(r'[.!?]+', response)
                for sentence in sentences[:3]:  # Top 3 sentences
                    sentence = sentence.strip()
                    if len(sentence) > 20 and sentence not in seen_points:
                        seen_points.add(sentence)
                        key_points.append(sentence)
            
            if key_points:
                return "Based on synthesis of multiple AI tutors:\n\n" + "\n\n".join(key_points[:5])
                
        return best_response
        
    def extract_best_patterns(self, responses: Dict[str, str]) -> List[Dict]:
        """Identify the best approaches from each tutor"""
        patterns = []
        
        for tutor, response in responses.items():
            # Extract patterns based on response characteristics
            patterns.append({
                'tutor': tutor,
                'style': self._analyze_style(response),
                'strengths': self._identify_strengths(response),
                'response_length': len(response)
            })
            
        return patterns
        
    def _analyze_style(self, response: str) -> str:
        """Analyze response style"""
        if len(response) < 100:
            return "concise"
        elif "first" in response.lower() and "second" in response.lower():
            return "structured"
        elif "```" in response:
            return "technical"
        elif "?" in response:
            return "interactive"
        else:
            return "narrative"
            
    def _identify_strengths(self, response: str) -> List[str]:
        """Identify strengths in the response"""
        strengths = []
        
        if len(response) > 500:
            strengths.append("detailed")
        if "```" in response:
            strengths.append("code_examples")
        if any(word in response.lower() for word in ["therefore", "thus", "consequently"]):
            strengths.append("reasoning")
        if "example" in response.lower():
            strengths.append("examples")
            
        return strengths if strengths else ["general"]
        
    def identify_gaps(self, responses: Dict[str, str], prompt: str) -> List[str]:
        """Find what no tutor could answer"""
        gaps = []
        
        # Check for common gap indicators
        gap_indicators = [
            "I don't know",
            "I cannot",
            "I'm not able",
            "unable to answer",
            "no information",
            "not available",
            "I don't have",
            "cannot answer"
        ]
        
        for tutor, response in responses.items():
            response_lower = response.lower()
            for indicator in gap_indicators:
                if indicator in response_lower:
                    gaps.append(f"{tutor} indicated: {indicator}")
                    
        # If all tutors responded with very short answers, likely a gap
        if all(len(r) < 50 for r in responses.values()):
            gaps.append("All tutors gave minimal responses - potential knowledge gap")
            
        return gaps
        
    def _calculate_confidence(self, responses: Dict[str, str]) -> float:
        """Calculate confidence based on agreement and response quality"""
        if len(responses) == 0:
            return 0.0
            
        # Base confidence on number of successful responses
        confidence = min(1.0, len(responses) / 5)  # Max confidence at 5+ tutors
        
        # Adjust for response consistency
        response_lengths = [len(r) for r in responses.values()]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            # Longer responses often indicate more confidence
            confidence += min(0.3, avg_length / 2000)  # Cap at 0.3
            
        return min(1.0, confidence)
        
    def create_training_data(self, synthesized: Dict) -> Dict:
        """Format for feeding into synthetic neural network"""
        training_data = {
            'input': {
                'prompt': synthesized.get('prompt', ''),
                'responses': synthesized.get('responses', {})
            },
            'output': {
                'unified_answer': synthesized.get('unified_answer', ''),
                'insights': synthesized.get('insights', []),
                'extracted_concepts': synthesized.get('extracted_concepts', {}),
                'patterns': synthesized.get('patterns', [])
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_tutors': len(synthesized.get('responses', {})),
                'confidence': synthesized.get('confidence', 0),
                'num_concepts': len(synthesized.get('extracted_concepts', {})),
                'num_patterns': len(synthesized.get('patterns', []))
            }
        }
        
        return training_data
        
    def get_synthesis_stats(self) -> Dict:
        """Get statistics about synthesis operations"""
        if not self.synthesis_history:
            return {'total_syntheses': 0}
        
        total_insights = 0
        total_concepts = 0
        total_patterns = 0
        
        for s in self.synthesis_history:
            total_insights += len(s.get('novel_insights', []))
            total_concepts += len(s.get('extracted_concepts', {}))
            total_patterns += len(s.get('patterns', []))
            
        return {
            'total_syntheses': len(self.synthesis_history),
            'avg_confidence': sum(s.get('confidence', 0) for s in self.synthesis_history) / len(self.synthesis_history),
            'total_insights': total_insights,
            'total_concepts': total_concepts,
            'total_patterns': total_patterns,
            'latest_synthesis': self.synthesis_history[-1].get('timestamp') if self.synthesis_history else None
        }
        
    def get_synthesis_history(self, limit: int = 10) -> List[Dict]:
        """Get recent synthesis history"""
        return self.synthesis_history[-limit:]
        
    def identify_gaps_enhanced(self, responses: Dict[str, str], prompt: str) -> List[str]:
        """Enhanced gap identification (alias for identify_gaps)"""
        return self.identify_gaps(responses, prompt)


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("Capability Synthesizer Test")
    print("=" * 60)
    
    synthesizer = CapabilitySynthesizer()
    
    # Test with mock responses
    mock_responses = {
        "openai": "Artificial General Intelligence (AGI) is a theoretical AI system that can understand, learn, and apply intelligence across a wide range of tasks, matching or exceeding human capabilities. Key approaches include neural networks and reinforcement learning.",
        "deepseek": "AGI represents the next frontier in AI research. It would possess human-like reasoning abilities and could transfer knowledge between domains. Current research focuses on scalable architectures and emergent capabilities.",
        "gemini": "The path to AGI involves creating systems that can generalize across domains. Important research areas include few-shot learning, meta-learning, and reasoning capabilities. Safety is a critical consideration."
    }
    
    result = synthesizer.synthesize(mock_responses, "What is AGI?")
    
    print(f"\n✅ Synthesis successful: {result.get('success', False)}")
    print(f"   Confidence: {result.get('confidence', 0):.2f}")
    print(f"   Extracted Concepts: {len(result.get('extracted_concepts', {}))}")
    for concept, data in list(result.get('extracted_concepts', {}).items())[:5]:
        print(f"     - {concept}: confidence {data['confidence']:.2f}")
    
    print(f"\n   Patterns Detected: {len(result.get('patterns', []))}")
    for pattern in result.get('patterns', [])[:3]:
        print(f"     - {pattern.get('type')}: '{pattern.get('term')}' (freq {pattern.get('frequency')})")
    
    print(f"\n   Novel Insights: {len(result.get('novel_insights', []))}")
    for insight in result.get('novel_insights', [])[:2]:
        print(f"     - {insight.get('insight')[:80]}...")
    
    print(f"\n   Gaps: {result.get('gaps', [])}")
    
    print("\n" + "=" * 60)
    print("Synthesis Stats:")
    print(json.dumps(synthesizer.get_synthesis_stats(), indent=2))
