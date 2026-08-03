
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

class ViralContentAnalyser:
    """Analyse viral content from social media to inform content generation."""
    
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.trends_file = self.data_path / "viral_trends.json"
        self.style_profile_file = self.data_path / "content_style_profile.json"
        self._load_data()
    
    def _load_data(self):
        if self.trends_file.exists():
            with open(self.trends_file) as f:
                self.trends = json.load(f)
        else:
            self.trends = {"trends": [], "updated_at": None}
        
        if self.style_profile_file.exists():
            with open(self.style_profile_file) as f:
                self.style_profile = json.load(f)
        else:
            self.style_profile = {
                "preferred_length_seconds": 30,
                "preferred_style": "educational",
                "preferred_topics": [],
                "engagement_factors": []
            }
    
    def _save_data(self):
        with open(self.trends_file, "w") as f:
            json.dump(self.trends, f, indent=2)
        with open(self.style_profile_file, "w") as f:
            json.dump(self.style_profile, f, indent=2)
    
    def analyse_trends(self):
        """Scan external sources for viral trends."""
        # This will be expanded to scrape YouTube, TikTok, etc.
        # For now, we'll use a placeholder
        print("Analysing viral trends...")
        # Placeholder data
        trends = [
            {"topic": "AI Automation", "engagement": 0.9, "platform": "YouTube"},
            {"topic": "Self‑Improvement", "engagement": 0.8, "platform": "TikTok"},
            {"topic": "Future Technology", "engagement": 0.7, "platform": "Twitter"},
        ]
        self.trends["trends"] = trends
        self.trends["updated_at"] = datetime.now().isoformat()
        self._update_style_profile(trends)
        self._save_data()
        return trends
    
    def _update_style_profile(self, trends):
        """Update the style profile based on trending content."""
        if not trends:
            return
        # Extract common topics
        topics = [t["topic"] for t in trends[:3]]
        self.style_profile["preferred_topics"] = topics
        # Adjust style based on engagement
        if trends[0]["engagement"] > 0.8:
            self.style_profile["preferred_style"] = "educational"
        self._save_data()
    
    def get_style_recommendation(self):
        """Return the current style recommendation."""
        return self.style_profile

if __name__ == "__main__":
    analyser = ViralContentAnalyser()
    trends = analyser.analyse_trends()
    print("Trends analysed:", trends)
    print("Style profile:", analyser.get_style_recommendation())
