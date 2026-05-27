"""
Lightweight screenshot extractor - optimized for free tier
"""

import base64
import json
import requests
import os
from pathlib import Path
from datetime import datetime

class LightweightExtractor:
    """Memory-efficient screenshot extraction"""
    
    def __init__(self):
        self.extracted_items = []
        # Limit stored items to prevent memory bloat
        self.max_stored_items = 50
    
    def analyze_screenshot(self, image_path: str, category: str = "auto") -> dict:
        """Extract info with minimal memory footprint"""
        
        # Read and process image in chunks
        with open(image_path, 'rb') as f:
            image_data = f.read()
            if len(image_data) > 5 * 1024 * 1024:  # Skip images > 5MB
                return {"error": "Image too large (>5MB)", "skipped": True}
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Free image data from memory
        del image_data
        
        prompt = f"Extract key info from this {category} screenshot. Return JSON with: title, summary, actionable_items, urls, code."
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {"error": "No API key", "content_type": "error"}
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4-vision-preview",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}],
                    "max_tokens": 1000,  # Reduced for memory
                    "temperature": 0.3
                },
                timeout=30
            )
            
            # Free base64 from memory
            del image_base64
            
            if response.status_code == 200:
                result = response.json()
                extracted = json.loads(result['choices'][0]['message']['content'])
                extracted['source'] = Path(image_path).name
                extracted['timestamp'] = datetime.now().isoformat()
                
                # Limit storage
                self.extracted_items.append(extracted)
                if len(self.extracted_items) > self.max_stored_items:
                    self.extracted_items.pop(0)
                
                return extracted
                
        except Exception as e:
            return {"error": str(e), "content_type": "error"}
        
        return {"error": "Unknown error"}
    
    def extract_batch(self, image_paths: list, category: str = "auto") -> dict:
        """Process batch with memory limits"""
        results = []
        for i, path in enumerate(image_paths):
            if i > 20:  # Limit batch size
                results.append({"skipped": True, "reason": "Batch limit reached"})
                break
            results.append(self.analyze_screenshot(path, category))
        return {"total": len(results), "results": results}

def initialize_lightweight_extractor():
    return LightweightExtractor()
