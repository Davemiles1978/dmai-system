"""
Content Validator - Checks for plagiarism and copyright issues
Every work must be verified before submission
"""

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ContentValidator:
    """Validate all content before submission"""
    
    def __init__(self):
        self.verified_works = {}
        self.rejected_works = {}
        self.plagiarism_db = self._load_plagiarism_db()
        self._load_state()
    
    def _load_plagiarism_db(self):
        """Load known copyrighted phrases/titles"""
        return {
            'titles': [
                'the da vinci code', 'gone girl', 'the girl on the train',
                'where the crawdads sing', 'the silent patient', 'project hail mary'
            ],
            'phrases': [
                'it was the best of times', 'call me ishmael', 'once upon a time',
                'they lived happily ever after', 'in a galaxy far far away'
            ]
        }
    
    def _load_state(self):
        """Load validation state"""
        state_file = Path("data/validation/verified_works.json")
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                self.verified_works = data.get('verified', {})
                self.rejected_works = data.get('rejected', {})
    
    def _save_state(self):
        """Save validation state"""
        state_file = Path("data/validation/verified_works.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump({
                'verified': self.verified_works,
                'rejected': self.rejected_works,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def validate_book(self, title: str, synopsis: str, chapters: List[Dict]) -> Dict:
        """Validate a book before submission"""
        
        issues = []
        warnings = []
        
        # Check title against known books
        title_lower = title.lower()
        for known_title in self.plagiarism_db['titles']:
            if known_title in title_lower or self._similarity(title_lower, known_title) > 0.7:
                issues.append(f"Title similar to existing book: {known_title}")
        
        # Check synopsis for copied phrases
        synopsis_lower = synopsis.lower()
        for phrase in self.plagiarism_db['phrases']:
            if phrase in synopsis_lower:
                warnings.append(f"Familiar phrase detected: '{phrase}' - consider rephrasing")
        
        # Check each chapter
        for i, chapter in enumerate(chapters):
            content = chapter.get('content', '')
            content_lower = content.lower()
            
            # Check for common phrases
            for phrase in self.plagiarism_db['phrases']:
                if phrase in content_lower:
                    warnings.append(f"Chapter {i+1}: Familiar phrase '{phrase}' detected")
            
            # Basic uniqueness check (simplified)
            word_count = len(content.split())
            unique_words = len(set(content.split()))
            uniqueness_ratio = unique_words / word_count if word_count > 0 else 1
            
            if uniqueness_ratio < 0.3:  # Less than 30% unique words
                warnings.append(f"Chapter {i+1}: Low uniqueness ratio ({uniqueness_ratio:.0%})")
        
        # Determine if valid
        is_valid = len(issues) == 0
        needs_review = len(warnings) > 0
        
        result = {
            'title': title,
            'is_valid': is_valid,
            'needs_review': needs_review,
            'issues': issues,
            'warnings': warnings,
            'uniqueness_score': self._calculate_uniqueness_score(synopsis, chapters),
            'validated_at': datetime.now().isoformat()
        }
        
        if is_valid and not needs_review:
            self.verified_works[title] = result
        else:
            self.rejected_works[title] = result
        
        self._save_state()
        return result
    
    def validate_tv_series(self, title: str, logline: str, overview: str) -> Dict:
        """Validate a TV series before submission"""
        
        issues = []
        warnings = []
        
        # Check against known TV shows
        known_shows = ['stranger things', 'wednesday', 'the last of us', 'succession', 'the crown']
        title_lower = title.lower()
        for known in known_shows:
            if known in title_lower or self._similarity(title_lower, known) > 0.7:
                issues.append(f"Title similar to existing series: {known}")
        
        # Check logline for originality
        common_logline_starters = [
            'when a', 'after a', 'a young', 'in a world', 'a group of'
        ]
        logline_lower = logline.lower()
        for starter in common_logline_starters:
            if logline_lower.startswith(starter):
                warnings.append(f"Logline starts with common phrase '{starter}' - consider making more distinctive")
        
        result = {
            'title': title,
            'is_valid': len(issues) == 0,
            'needs_review': len(warnings) > 0,
            'issues': issues,
            'warnings': warnings,
            'validated_at': datetime.now().isoformat()
        }
        
        if result['is_valid'] and not result['needs_review']:
            self.verified_works[title] = result
        
        self._save_state()
        return result
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple similarity check"""
        if not a or not b:
            return 0
        set_a = set(a.split())
        set_b = set(b.split())
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union) if union else 0
    
    def _calculate_uniqueness_score(self, synopsis: str, chapters: List[Dict]) -> float:
        """Calculate overall uniqueness score"""
        all_text = synopsis + " " + " ".join([c.get('content', '') for c in chapters])
        words = all_text.split()
        if not words:
            return 1.0
        unique_words = len(set(words))
        return min(1.0, unique_words / len(words) * 2)  # Normalize
    
    def get_verification_status(self, title: str) -> Dict:
        """Get verification status for a work"""
        if title in self.verified_works:
            return {'status': 'verified', 'details': self.verified_works[title]}
        elif title in self.rejected_works:
            return {'status': 'needs_revision', 'details': self.rejected_works[title]}
        return {'status': 'not_checked'}
