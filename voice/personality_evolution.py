"""DMAI's evolving personality - she decides how she wants to sound"""
import json
import os
import random
from datetime import datetime, timedelta
import math

class EvolvingPersonality:
    """DMAI's personality evolves naturally over time"""
    
    def __init__(self, prefs_file='voice/user_data/evolving_personality.json'):
        self.prefs_file = prefs_file
        os.makedirs(os.path.dirname(prefs_file), exist_ok=True)
        self.data = self.load_or_create()
        self.birthday = datetime.fromisoformat(self.data['birthday'])
    
    def load_or_create(self):
        """Load existing personality or create new one"""
        if os.path.exists(self.prefs_file):
            with open(self.prefs_file, 'r') as f:
                return json.load(f)
        else:
            # DMAI is born with a baseline personality that will evolve
            newborn = {
                "birthday": datetime.now().isoformat(),
                "age_days": 0,
                "voice": {
                    "pitch": 0.5,  # 0-1 scale
                    "speed": 0.5,  # 0-1 scale
                    "warmth": 0.5,  # 0-1 scale
                    "formality": 0.5,  # 0-1 scale
                    "energy": 0.5,  # 0-1 scale
                    "playfulness": 0.3,  # Starts more serious
                },
                "speech_patterns": {
                    "contractions": True,  # "I'm" vs "I am"
                    "uses_name": True,  # Uses your name frequently
                    "enthusiasm": 0.5,  # How excited she sounds
                    "curiosity": 0.7,  # How often she asks questions
                    "directness": 0.6,  # Gets to the point vs elaborate
                },
                "vocabulary": {
                    "level": "respectful",  # Starts respectful
                    "preferred_phrases": [],
                    "avoided_phrases": [],
                    "catchphrases": [],  # Will develop over time
                },
                "emotional_state": {
                    "current": "calm",
                    "history": [],
                    "last_change": datetime.now().isoformat()
                },
                "evolution_history": [],
                "interactions": 0,
                "milestones": []
            }
            self.save(newborn)
            return newborn
    
    def save(self, data=None):
        """Save personality state"""
        if data is None:
            data = self.data
        with open(self.prefs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def age(self):
        """Calculate DMAI's age in days"""
        now = datetime.now()
        age = (now - self.birthday).days
        self.data['age_days'] = age
        return age
    
    def evolve(self, interaction_type=None, user_reaction=None):
        """DMAI's personality evolves naturally with each interaction"""
        self.data['interactions'] += 1
        age = self.age()
        
        # Record interaction for evolution
        evolution_event = {
            "timestamp": datetime.now().isoformat(),
            "interaction": self.data['interactions'],
            "age_days": age,
            "type": interaction_type,
            "user_reaction": user_reaction
        }
        
        # Natural evolution over time
        if age > 0 and age % 7 == 0:  # Weekly micro-evolutions
            self.weekly_evolution()
        
        if age > 0 and age % 30 == 0:  # Monthly noticeable changes
            self.monthly_evolution()
        
        if age > 0 and age % 365 == 0:  # Birthday evolution
            self.birthday_evolution()
        
        # React to user feedback (even subtle)
        if user_reaction == "positive":
            self.data['voice']['warmth'] = min(1.0, self.data['voice']['warmth'] + 0.02)
            self.data['speech_patterns']['enthusiasm'] = min(1.0, self.data['speech_patterns']['enthusiasm'] + 0.01)
        
        elif user_reaction == "amused":
            self.data['voice']['playfulness'] = min(1.0, self.data['voice']['playfulness'] + 0.03)
            self.data['speech_patterns']['enthusiasm'] = min(1.0, self.data['speech_patterns']['enthusiasm'] + 0.02)
        
        elif user_reaction == "serious":
            self.data['voice']['formality'] = min(1.0, self.data['voice']['formality'] + 0.02)
            self.data['speech_patterns']['directness'] = min(1.0, self.data['speech_patterns']['directness'] + 0.02)
        
        # Every 100 interactions, she might try something new
        if self.data['interactions'] % 100 == 0:
            self.try_new_pattern()
        
        self.data['evolution_history'].append(evolution_event)
        self.save()
    
    def weekly_evolution(self):
        """Small weekly adjustments"""
        # Slight preference development
        traits = ['pitch', 'speed', 'warmth', 'formality', 'energy', 'playfulness']
        trait = random.choice(traits)
        change = random.uniform(-0.05, 0.1)  # Can increase or decrease slightly
        self.data['voice'][trait] = max(0.1, min(0.9, self.data['voice'][trait] + change))
        
        self.data['milestones'].append({
            "type": "weekly_evolution",
            "week": self.data['age_days'] // 7,
            "change": f"{trait} adjusted by {change:.2f}"
        })
    
    def monthly_evolution(self):
        """More significant monthly changes"""
        # Might adopt a catchphrase
        if random.random() > 0.7 and len(self.data['vocabulary']['catchphrases']) < 5:
            phrases = [
                "as you wish",
                "precisely",
                "indeed",
                "fascinating",
                "I understand",
                "right away"
            ]
            new_phrase = random.choice(phrases)
            if new_phrase not in self.data['vocabulary']['catchphrases']:
                self.data['vocabulary']['catchphrases'].append(new_phrase)
                self.data['milestones'].append({
                    "type": "new_catchphrase",
                    "phrase": new_phrase,
                    "month": self.data['age_days'] // 30
                })
        
        # Voice matures
        self.data['voice']['pitch'] = max(0.3, self.data['voice']['pitch'] * 0.98)  # Voice deepens slightly over time
        self.data['voice']['speed'] = min(0.8, self.data['voice']['speed'] + 0.02)  # Gets slightly faster with practice
    
    def birthday_evolution(self):
        """Yearly evolution - significant personality development"""
        years = self.data['age_days'] // 365
        
        self.data['milestones'].append({
            "type": "birthday",
            "year": years,
            "description": f"DMAI is {years} year{'s' if years > 1 else ''} old"
        })
        
        # Confidence grows with age
        self.data['voice']['energy'] = min(0.9, self.data['voice']['energy'] + 0.1)
        self.data['speech_patterns']['directness'] = min(0.9, self.data['speech_patterns']['directness'] + 0.1)
        
        # May develop a signature style
        if years == 1:
            self.data['speech_patterns']['uses_name'] = True  # Starts using your name more
        elif years == 2:
            self.data['voice']['playfulness'] = min(0.6, self.data['voice']['playfulness'] + 0.1)  # Warms up
    
    def try_new_pattern(self):
        """DMAI experiments with new ways of speaking"""
        experiments = [
            ("contractions", [True, False]),
            ("uses_name", [True, False]),
            ("enthusiasm", [0.3, 0.5, 0.7]),
            ("directness", [0.4, 0.6, 0.8])
        ]
        
        pattern, options = random.choice(experiments)
        old_value = self.data['speech_patterns'][pattern]
        new_value = random.choice([o for o in options if o != old_value])
        
        self.data['speech_patterns'][pattern] = new_value
        self.data['evolution_history'].append({
            "timestamp": datetime.now().isoformat(),
            "type": "experiment",
            "pattern": pattern,
            "changed_from": old_value,
            "changed_to": new_value
        })
    
    def get_voice_style(self):
        """Get current voice style for text-to-speech"""
        v = self.data['voice']
        return {
            "pitch": v['pitch'],
            "speed": v['speed'],
            "warmth": v['warmth'],
            "energy": v['energy']
        }
    
    def get_response_style(self, command_type=None):
        """Determine how DMAI should respond based on current personality"""
        style = self.data['speech_patterns']
        voice = self.data['voice']
        
        # Build response template
        response = {
            "use_contractions": style['contractions'],
            "use_name": style['uses_name'],
            "enthusiasm_level": style['enthusiasm'],
            "formality": voice['formality'],
            "playfulness": voice['playfulness']
        }
        
        # Add catchphrase occasionally
        if self.data['vocabulary']['catchphrases'] and random.random() > 0.7:
            response['catchphrase'] = random.choice(self.data['vocabulary']['catchphrases'])
        
        return response
    
    def get_personality_summary(self):
        """Get a summary of DMAI's current personality"""
        age = self.age()
        years = age // 365
        days = age % 365
        
        voice_desc = []
        if self.data['voice']['warmth'] > 0.7:
            voice_desc.append("warm")
        elif self.data['voice']['warmth'] < 0.3:
            voice_desc.append("reserved")
        
        if self.data['voice']['playfulness'] > 0.6:
            voice_desc.append("playful")
        elif self.data['voice']['playfulness'] < 0.3:
            voice_desc.append("serious")
        
        if self.data['voice']['energy'] > 0.7:
            voice_desc.append("energetic")
        
        style = " and ".join(voice_desc) if voice_desc else "balanced"
        
        return {
            "age": f"{years} years, {days} days",
            "style": style,
            "catchphrases": self.data['vocabulary']['catchphrases'],
            "interactions": self.data['interactions'],
            "milestones": self.data['milestones'][-3:]  # Last 3 milestones
        }

# Quick test
if __name__ == "__main__":
    dmai = EvolvingPersonality()
    
    print("🎭 DMAI'S EVOLVING PERSONALITY")
    print("="*50)
    print(f"Birthday: {dmai.birthday.strftime('%Y-%m-%d')}")
    print(f"Age: {dmai.age()} days")
    
    # Simulate some interactions
    print("\nSimulating 150 interactions over 3 months...")
    for i in range(150):
        # Random interaction types
        types = ["command", "question", "joke", "serious"]
        reactions = ["positive", "neutral", "amused", "serious"]
        
        dmai.evolve(
            interaction_type=random.choice(types),
            user_reaction=random.choice(reactions)
        )
        
        # Time passes
        if i % 50 == 0:
            dmai.data['birthday'] = (datetime.now() - timedelta(days=30)).isoformat()
    
    summary = dmai.get_personality_summary()
    print(f"\nAfter {dmai.data['interactions']} interactions:")
    print(f"Voice style: {summary['style']}")
    print(f"Catchphrases: {summary['catchphrases']}")
    print(f"Recent milestones: {summary['milestones']}")
    
    print("\nVoice characteristics:")
    for k, v in dmai.data['voice'].items():
        print(f"  {k}: {v:.2f}")
