import sys
import os
import json
import re
import random
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))))))

class HumanSpeechLearner:
    def __init__(self):
        self.speech_file = "language_learning/data/human_speech.json"
        self.load_speech_patterns()
    
    def load_speech_patterns(self):
        if os.path.exists(self.speech_file):
            with open(self.speech_file, 'r') as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {
                "contractions": {
                    "i am": "I'm",
                    "you are": "you're",
                    "we are": "we're", 
                    "they are": "they're",
                    "is not": "isn't",
                    "are not": "aren't",
                    "was not": "wasn't",
                    "were not": "weren't",
                    "will not": "won't",
                    "would not": "wouldn't",
                    "cannot": "can't",
                    "could not": "couldn't",
                    "should not": "shouldn't",
                    "must not": "mustn't",
                    "do not": "don't",
                    "does not": "doesn't",
                    "did not": "didn't",
                    "have not": "haven't",
                    "has not": "hasn't",
                    "had not": "hadn't",
                    "let us": "let's"
                },
                "fillers": ["um", "uh", "like", "you know", "actually", "basically", "honestly"],
                "casual_greetings": ["hey", "hi", "hello", "what's up", "howdy", "yo"],
                "responses": {
                    "positive": ["cool", "awesome", "great", "nice", "sweet", "fantastic"],
                    "negative": ["ugh", "darn", "oh no", "bummer", "that sucks"],
                    "thinking": ["let me see", "hmm", "well", "so", "okay, so"]
                },
                "conversation_starters": [
                    "so, here's the thing",
                    "you know what I mean?",
                    "the way I see it",
                    "if you ask me",
                    "to be honest",
                    "honestly speaking",
                    "at the end of the day"
                ],
                "learned_patterns": []
            }
            self.save_patterns()
    
    def save_patterns(self):
        with open(self.speech_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def naturalize_text(self, text):
        if not text:
            return text
        
        for formal, contraction in self.patterns["contractions"].items():
            text = re.sub(rf'\b{formal}\b', contraction, text, flags=re.IGNORECASE)
        
        if random.random() < 0.3:
            filler = random.choice(self.patterns["fillers"])
            if ', ' in text:
                parts = text.split(', ', 1)
                text = f"{parts[0]}, {filler}, {parts[1]}"
        
        if text.startswith(('I ', 'It ', 'That ')) and random.random() < 0.2:
            starter = random.choice(self.patterns["conversation_starters"])
            text = f"{starter}, {text[0].lower()}{text[1:]}"
        
        return text
    
    def learn_from_transcript(self, transcript, source="conversation"):
        sentences = re.split(r'[.!?]+', transcript)
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 10:
                continue
            
            words = sentence.split()
            for i in range(len(words)-1):
                combo = f"{words[i]} {words[i+1]}"
                if combo in ["gonna", "wanna", "gotta", "dunno", "kinda", "sorta"]:
                    if combo not in self.patterns["contractions"]:
                        self.patterns["contractions"][combo] = combo
            
            potential_fillers = ["like", "actually", "basically", "literally"]
            for filler in potential_fillers:
                if filler in sentence and filler not in self.patterns["fillers"]:
                    self.patterns["fillers"].append(filler)
            
            if len(sentence.split()) > 3 and sentence not in [p.get('text') for p in self.patterns["learned_patterns"][-100:]]:
                self.patterns["learned_patterns"].append({
                    "text": sentence,
                    "source": source,
                    "learned": datetime.now().isoformat()
                })
        
        self.save_patterns()
    
    def get_natural_response(self, intent, context=None):
        responses = {
            "create": ["I'll get that sorted", "On it", "Working on that now", "Sure thing"],
            "research": ["Let me dig into that", "I'll find out", "Looking that up now"],
            "status": ["All good here", "Everything's running smoothly", "No issues to report"],
            "device": ["Checking your devices", "Let me find that for you", "One sec"]
        }
        
        base_response = random.choice(responses.get(intent, ["Okay"]))
        return self.naturalize_text(base_response)

if __name__ == "__main__":
    speech = HumanSpeechLearner()
    
    test_texts = [
        "I am going to create a video for you",
        "I cannot find your phone at this moment",
        "It is not working as expected",
        "We are going to learn a lot today"
    ]
    
    print("🎭 Natural Speech Learning Test")
    print("="*50)
    for text in test_texts:
        natural = speech.naturalize_text(text)
        print(f"Formal: {text}")
        print(f"Natural: {natural}")
        print("-"*50)
