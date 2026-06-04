"""
Content Generator - Alex Riviera as author
Includes plagiarism check before returning content
"""

import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from .identity import ALEX_RIVIERA, BOOK_GENRES, TV_GENRES

# Import plagiarism checker
import sys
sys.path.insert(0, '/Users/davidmiles/Desktop/dmai-system')
from components.plagiarism.ContentValidator import ContentValidator

class AlexRivieraContent:
    """Content generator - Alex Riviera as the author"""
    
    def __init__(self, ai_hub=None):
        self.ai_hub = ai_hub
        self.creator = ALEX_RIVIERA
        self.validator = ContentValidator()
        self.generated_works = []
    
    def generate_and_validate_book(self, genre: str = None, subgenre: str = None) -> Tuple[Dict, Dict]:
        """Generate a book and validate it before returning"""
        
        book = self._generate_book(genre, subgenre)
        
        # Create chapters for validation
        chapters = [{'content': book['synopsis']}]  # Simplified for validation
        
        # Validate
        validation = self.validator.validate_book(
            book['title'],
            book['synopsis'],
            chapters
        )
        
        # Record the work
        self.generated_works.append({
            'type': 'book',
            'title': book['title'],
            'genre': book['genre'],
            'validation': validation,
            'created_at': datetime.now().isoformat()
        })
        
        return book, validation
    
    def generate_and_validate_tv_series(self, genre: str = None, subgenre: str = None) -> Tuple[Dict, Dict]:
        """Generate a TV series and validate it before returning"""
        
        series = self._generate_tv_series(genre, subgenre)
        
        # Validate
        validation = self.validator.validate_tv_series(
            series['title'],
            series['logline'],
            series['overview']
        )
        
        # Record the work
        self.generated_works.append({
            'type': 'tv_series',
            'title': series['title'],
            'genre': series['genre'],
            'validation': validation,
            'created_at': datetime.now().isoformat()
        })
        
        return series, validation
    
    def _generate_book(self, genre: str = None, subgenre: str = None) -> Dict:
        """Generate a book (internal)"""
        
        if not genre:
            genre = random.choice(list(BOOK_GENRES.keys()))
        
        subgenres = BOOK_GENRES.get(genre, [genre.capitalize()])
        if not subgenre:
            subgenre = random.choice(subgenres)
        
        titles = self._get_titles(genre)
        selected_title = random.choice(titles)
        
        return {
            'author': self.creator['name'],
            'author_email': self.creator['email'],
            'title': selected_title,
            'genre': genre,
            'subgenre': subgenre,
            'logline': self._get_logline(genre, selected_title),
            'synopsis': self._get_synopsis(selected_title, genre),
            'word_count': random.randint(60000, 100000),
            'chapters': random.randint(12, 20),
            'completion_date': datetime.now().strftime('%B %Y'),
            'target_audience': self._get_audience(genre),
            'exclusive_days': 60
        }
    
    def _generate_tv_series(self, genre: str = None, subgenre: str = None) -> Dict:
        """Generate a TV series (internal)"""
        
        if not genre:
            genre = random.choice(list(TV_GENRES.keys()))
        
        subgenres = TV_GENRES.get(genre, [genre.capitalize()])
        if not subgenre:
            subgenre = random.choice(subgenres)
        
        titles = self._get_tv_titles(genre)
        selected_title = random.choice(titles)
        
        return {
            'creator': self.creator['name'],
            'creator_email': self.creator['email'],
            'title': selected_title,
            'genre': genre,
            'subgenre': subgenre,
            'logline': self._get_tv_logline(genre, selected_title),
            'overview': self._get_tv_overview(selected_title, genre),
            'episodes': random.randint(6, 12),
            'episode_length': '30 minutes' if genre == 'sitcom' else '60 minutes',
            'market_fit': self._get_market_fit(genre)
        }
    
    def _get_titles(self, genre: str) -> List[str]:
        """Get book titles by genre"""
        titles = {
            'fiction': ['The Last Chapter', 'Where Rivers Meet', 'The Memory Keeper', 'Falling Together'],
            'non_fiction': ['The Creative Path', 'Finding Your Voice', 'The Reset', 'Breaking Through'],
            'comedy': ['The Worst Best Year', 'Dating Disasters', 'My So-Called Midlife Crisis'],
            'sci_fi': ['The Quantum Divide', 'Echo Protocol', 'The Last Colony', 'Neural Threads'],
            'childrens': ['Penny the Brave', 'The Little Star That Could', 'Benny and the Big Feelings'],
            'mystery': ['The Lake House Secret', 'Three Days Gone', 'The Witness'],
            'horror': ['The Hollow Place', 'Whisper Ridge', 'The Dark Between'],
            'romance': ['The Summer of Second Chances', 'Love in the Afternoon', 'The Wedding Agreement']
        }
        return titles.get(genre, ['The Untitled Project'])
    
    def _get_tv_titles(self, genre: str) -> List[str]:
        """Get TV titles by genre"""
        titles = {
            'sitcom': ['Roommates', 'The Break Room', 'Modern Family Ties', 'Next Door'],
            'thriller': ['The Cover-up', 'Point of No Return', 'The Watcher', 'Deep State'],
            'drama': ['Legacy', 'The Heights', 'Crossroads', 'Shattered Glass'],
            'sci_fi': ['The Signal', 'New Eden', 'Resistance', 'The Collective'],
            'mystery': ['Cold River', 'The Keeper', 'Shadow Lane', 'Unsolved'],
            'horror': ['The Hollow', 'Nightfall', 'Harvest', 'The Uninvited']
        }
        return titles.get(genre, ['Untitled Series'])
    
    def _get_logline(self, genre: str, title: str) -> str:
        """Get logline by genre"""
        loglines = {
            'fiction': f"When a young archivist discovers a hidden collection of letters, she uncovers a love story that changes everything.",
            'sci_fi': f"In a future where memories can be bought and sold, one woman fights to recover what was taken from her.",
            'mystery': f"A cold case detective gets a second chance when new evidence emerges in the disappearance that has haunted her for a decade.",
            'romance': f"A travel writer and a small-town bookstore owner discover that home isn't a place - it's a person."
        }
        return loglines.get(genre, f"{title} is a compelling story that will keep readers turning pages.")
    
    def _get_synopsis(self, title: str, genre: str) -> str:
        """Get synopsis"""
        return f"""{title} follows the journey of an ordinary person in extraordinary circumstances.

Set in present-day America, this {genre} explores themes of identity, belonging, and purpose.

Alex Riviera's writing brings fresh perspective to the {genre} genre."""
    
    def _get_audience(self, genre: str) -> str:
        """Get target audience"""
        audiences = {
            'fiction': 'Adults 25-55, book clubs',
            'sci_fi': 'Adults 20-50, sci-fi enthusiasts',
            'mystery': 'Adults 25-65, mystery lovers',
            'romance': 'Adults 18-50, romance readers'
        }
        return audiences.get(genre, 'General adult audience')
    
    def _get_tv_logline(self, genre: str, title: str) -> str:
        """Get TV logline"""
        loglines = {
            'sitcom': f"When a group of twenty-somethings navigates dating and careers, chaos and comedy ensue.",
            'thriller': f"A journalist uncovers a conspiracy that threatens everything - and everyone - she loves.",
            'drama': f"Three generations of a family struggle with secrets, ambition, and forgiveness.",
            'sci_fi': f"In a future where technology has failed, one woman fights for humanity's survival."
        }
        return loglines.get(genre, f"{title} is a compelling new series.")
    
    def _get_tv_overview(self, title: str, genre: str) -> str:
        """Get TV overview"""
        return f"""{title} is a {genre} series created by Alex Riviera.

The series follows compelling characters through unexpected twists and turns.

Alex Riviera brings a fresh voice to the {genre} genre, with character-driven storytelling."""
    
    def _get_market_fit(self, genre: str) -> str:
        """Get market fit - no AI mentions"""
        return f"""Audiences are hungry for fresh {genre} content. Recent successes demonstrate strong appetite for original voices in this space."""
    
    def get_verification_summary(self) -> Dict:
        """Get summary of all verified works"""
        verified = [w for w in self.generated_works if w['validation'].get('is_valid', False)]
        needs_revision = [w for w in self.generated_works if w['validation'].get('needs_review', False)]
        
        return {
            'total_generated': len(self.generated_works),
            'verified_ready': len(verified),
            'needs_revision': len(needs_revision),
            'verified_list': [w['title'] for w in verified],
            'revision_list': [w['title'] for w in needs_revision]
        }
