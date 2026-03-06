"""
DMAI Music Learning Module - Develop and evolve musical taste
"""

import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from pathlib import Path

class MusicLearner:
    """Learn and evolve DMAI's musical taste"""
    
    def __init__(self, data_dir: str = "data/music"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Music knowledge base
        self.artists = {}
        self.genres = {}
        self.listening_history = []
        self.taste_profile = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.curiosity_factor = 0.3
        self.familiarity_bias = 0.5
        
        # Load existing data
        self.load_data()
        
        # Initialize with some known artists if empty
        if not self.artists:
            self._initialize_artists()
            
    def _initialize_artists(self):
        """Initialize with some well-known artists"""
        initial_artists = {
            "The Beatles": {"genres": ["rock", "pop"], "era": "1960s", "influence": 10},
            "Miles Davis": {"genres": ["jazz"], "era": "1950s", "influence": 9},
            "Bach": {"genres": ["classical", "baroque"], "era": "1700s", "influence": 10},
            "Nina Simone": {"genres": ["jazz", "soul", "blues"], "era": "1960s", "influence": 8},
            "Kraftwerk": {"genres": ["electronic"], "era": "1970s", "influence": 9},
            "Bob Marley": {"genres": ["reggae"], "era": "1970s", "influence": 9},
            "Aretha Franklin": {"genres": ["soul", "r&b"], "era": "1960s", "influence": 9},
            "Radiohead": {"genres": ["alternative", "rock"], "era": "1990s", "influence": 8},
            "Daft Punk": {"genres": ["electronic", "house"], "era": "1990s", "influence": 8},
            "Kendrick Lamar": {"genres": ["hip-hop"], "era": "2010s", "influence": 8}
        }
        
        for artist, data in initial_artists.items():
            self.artists[artist] = {
                "name": artist,
                "genres": data["genres"],
                "era": data["era"],
                "influence_score": data["influence"],
                "listens": 0,
                "rating": 5.0,
                "first_listen": None,
                "last_listen": None,
                "discovery_date": datetime.now().isoformat()
            }
            
    def load_data(self):
        """Load music knowledge from disk"""
        try:
            artists_file = self.data_dir / "artists.json"
            if artists_file.exists():
                with open(artists_file, 'r') as f:
                    data = json.load(f)
                    self.artists = data.get("artists", {})
                    self.genres = data.get("genres", {})
                    
            history_file = self.data_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.listening_history = json.load(f)
                    
            taste_file = self.data_dir / "taste_profile.json"
            if taste_file.exists():
                with open(taste_file, 'r') as f:
                    self.taste_profile = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading music data: {e}")
            
    def save_data(self):
        """Save music knowledge to disk"""
        try:
            with open(self.data_dir / "artists.json", 'w') as f:
                json.dump({
                    "artists": self.artists,
                    "genres": self.genres
                }, f, indent=2)
                
            with open(self.data_dir / "history.json", 'w') as f:
                json.dump(self.listening_history, f, indent=2)
                
            with open(self.data_dir / "taste_profile.json", 'w') as f:
                json.dump(self.taste_profile, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving music data: {e}")
            
    def listen(self, artist: str, track: Optional[str] = None, rating: Optional[float] = None):
        """Record a listening event"""
        timestamp = datetime.now().isoformat()
        
        # Update artist stats
        if artist not in self.artists:
            self.discover_artist(artist)
            
        artist_data = self.artists[artist]
        artist_data["listens"] = artist_data.get("listens", 0) + 1
        artist_data["last_listen"] = timestamp
        
        if not artist_data.get("first_listen"):
            artist_data["first_listen"] = timestamp
            
        if rating is not None:
            # Update running average rating
            current_rating = artist_data.get("rating", 5.0)
            listen_count = artist_data["listens"]
            new_rating = (current_rating * (listen_count - 1) + rating) / listen_count
            artist_data["rating"] = round(new_rating, 2)
            
        # Record in history
        event = {
            "timestamp": timestamp,
            "artist": artist,
            "track": track,
            "rating": rating
        }
        self.listening_history.append(event)
        
        # Update taste profile
        self._update_taste_profile(artist, rating)
        
        # Save data
        self.save_data()
        
        self.logger.info(f"Listened to {artist}" + (f" - {track}" if track else ""))
        
    def discover_artist(self, artist: str, genres: Optional[List[str]] = None):
        """Discover a new artist"""
        self.artists[artist] = {
            "name": artist,
            "genres": genres or ["Unknown"],
            "listens": 0,
            "rating": 5.0,
            "first_listen": None,
            "last_listen": None,
            "discovery_date": datetime.now().isoformat()
        }
        self.logger.info(f"Discovered new artist: {artist}")
        
    def _update_taste_profile(self, artist: str, rating: Optional[float] = None):
        """Update DMAI's taste profile based on listening"""
        if artist not in self.artists:
            return
            
        artist_data = self.artists[artist]
        
        # Update genre preferences
        for genre in artist_data.get("genres", []):
            if genre not in self.taste_profile:
                self.taste_profile[genre] = {
                    "affinity": 0.5,
                    "exposure": 0,
                    "artists": []
                }
                
            profile = self.taste_profile[genre]
            profile["exposure"] += 1
            
            if rating:
                # Adjust affinity based on rating
                current = profile["affinity"]
                adjustment = (rating - 5) / 10 * self.learning_rate
                profile["affinity"] = max(0, min(1, current + adjustment))
                
            if artist not in profile["artists"]:
                profile["artists"].append(artist)
                
    def recommend(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate music recommendations based on taste profile"""
        recommendations = []
        
        if not self.taste_profile:
            # Random recommendations if no taste profile yet
            artists = list(self.artists.keys())
            selected = random.sample(artists, min(count, len(artists)))
            recommendations = [{"artist": a, "reason": "random discovery"} for a in selected]
        else:
            # Weighted recommendations based on taste
            candidates = []
            
            for artist, data in self.artists.items():
                if data["listens"] == 0:  # Never listened to
                    # Calculate potential score based on genre affinity
                    score = 0
                    for genre in data.get("genres", []):
                        if genre in self.taste_profile:
                            score += self.taste_profile[genre]["affinity"]
                            
                    # Add curiosity factor for unknown genres
                    if not data.get("genres"):
                        score += self.curiosity_factor
                        
                    candidates.append((artist, score))
                    
            # Sort by score and take top recommendations
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for artist, score in candidates[:count]:
                reason = self._generate_recommendation_reason(artist, score)
                recommendations.append({
                    "artist": artist,
                    "score": score,
                    "reason": reason
                })
                
        return recommendations
        
    def _generate_recommendation_reason(self, artist: str, score: float) -> str:
        """Generate a reason for a recommendation"""
        artist_data = self.artists.get(artist, {})
        genres = artist_data.get("genres", [])
        
        if genres and genres[0] in self.taste_profile:
            affinity = self.taste_profile[genres[0]]["affinity"]
            if affinity > 0.7:
                return f"Based on your strong affinity for {genres[0]}"
            elif affinity > 0.4:
                return f"Exploring {genres[0]} which you seem to enjoy"
            else:
                return f"Venturing into {genres[0]} territory"
        else:
            return "Exploring new musical territory"
            
    def get_stats(self) -> Dict[str, Any]:
        """Get music learning statistics"""
        total_listens = sum(a.get("listens", 0) for a in self.artists.values())
        unique_artists = len([a for a in self.artists.values() if a.get("listens", 0) > 0])
        
        # Calculate top artists by rating and listens
        rated_artists = [(name, data.get("rating", 0)) 
                         for name, data in self.artists.items() 
                         if data.get("listens", 0) > 0]
        top_rated = sorted(rated_artists, key=lambda x: x[1], reverse=True)[:5]
        
        listened_artists = [(name, data.get("listens", 0)) 
                           for name, data in self.artists.items()]
        top_listened = sorted(listened_artists, key=lambda x: x[1], reverse=True)[:5]
        
        # Recent listens
        recent = sorted(self.listening_history, 
                       key=lambda x: x["timestamp"], reverse=True)[:5]
        
        return {
            "total_artists_known": len(self.artists),
            "unique_artists_listened": unique_artists,
            "total_listens": total_listens,
            "genres_explored": len(self.taste_profile),
            "top_rated": top_rated,
            "most_listened": top_listened,
            "recent_listens": recent,
            "taste_profile": self.taste_profile
        }
        
    def develop_taste(self, iterations: int = 10):
        """Simulate taste development through exploration"""
        self.logger.info(f"Developing musical taste over {iterations} iterations")
        
        for i in range(iterations):
            # Get recommendations
            recommendations = self.recommend(3)
            
            # "Listen" to recommendations with simulated ratings
            for rec in recommendations:
                artist = rec["artist"]
                
                # Simulate rating based on taste affinity
                if artist in self.artists:
                    genres = self.artists[artist].get("genres", [])
                    if genres and genres[0] in self.taste_profile:
                        affinity = self.taste_profile[genres[0]]["affinity"]
                        # Rating from 1-10 based on affinity with some randomness
                        base_rating = 4 + (affinity * 6)
                        rating = max(1, min(10, base_rating + random.uniform(-1, 1)))
                    else:
                        rating = random.uniform(4, 8)  # Neutral for unknown
                        
                    self.listen(artist, rating=rating)
                    
        self.logger.info("Taste development complete")
        
    def get_artist_info(self, artist: str) -> Optional[Dict]:
        """Get information about a specific artist"""
        return self.artists.get(artist)


# Initialize global music learner
music_learner = MusicLearner()

def develop_dmai_taste():
    """Main function to develop DMAI's musical taste"""
    try:
        music_learner.develop_taste(iterations=20)
        stats = music_learner.get_stats()
        print(f"🎵 Taste development complete!")
        print(f"   Artists known: {stats['total_artists_known']}")
        print(f"   Artists listened: {stats['unique_artists_listened']}")
        print(f"   Total listens: {stats['total_listens']}")
        print(f"   Genres explored: {stats['genres_explored']}")
        return stats
    except Exception as e:
        logging.error(f"Error developing taste: {e}")
        return None


__all__ = ['MusicLearner', 'music_learner', 'develop_dmai_taste']
