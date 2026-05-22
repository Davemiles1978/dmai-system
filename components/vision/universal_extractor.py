import base64
import re
import json
import requests
import os
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExtractedContent:
    content_type: str
    title: str
    content: str
    source_file: str
    timestamp: float
    tags: List[str]
    confidence: float
    actionable: bool
    code: str = ""
    url: str = ""
    references: List[str] = None

class UniversalScreenshotExtractor:
    def __init__(self):
        self.extracted_items = []
        self.learned_knowledge = []
        self.implementation_queue = []
    
    def analyze_screenshot(self, image_path: str, category: str = "auto") -> Dict:
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = f"Analyze this screenshot. Category hint: {category}. Extract: algorithms, AI systems, research, repos, prompts, storage designs, code, URLs. Return JSON with fields: content_type, title, summary, detailed_content, code_snippets, urls, repos, algorithms, ai_systems, research_papers, prompts, storage_schemas, actionable, implementation_priority, tags."
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "gpt-4-vision-preview", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}], "max_tokens": 2000},
                    timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    extracted = json.loads(result['choices'][0]['message']['content'])
                    extracted['source_file'] = Path(image_path).name
                    extracted['timestamp'] = datetime.now().isoformat()
                    self.extracted_items.append(extracted)
                    if extracted.get('actionable'):
                        self.implementation_queue.append(extracted)
                    return extracted
            except Exception as e:
                return {"error": str(e), "content_type": "error"}
        return {"error": "No API key", "content_type": "error"}
    
    def extract_batch(self, image_paths: List[str], category: str = "auto") -> Dict:
        results = []
        for path in image_paths:
            results.append(self.analyze_screenshot(path, category))
        return {"total": len(results), "results": results}
    
    def get_implementation_queue(self) -> List[Dict]:
        return self.implementation_queue

class KnowledgeIntegrator:
    def __init__(self, si_core=None):
        self.si_core = si_core
        self.integrated_items = []
    
    def integrate_algorithm(self, algorithm: Dict) -> bool:
        print(f"Integrating algorithm: {algorithm.get('name', 'Unknown')}")
        return True
    
    def integrate_prompt(self, prompt: Dict) -> bool:
        prompt_library = Path("data/prompts.json")
        prompts = []
        if prompt_library.exists():
            with open(prompt_library, 'r') as f:
                prompts = json.load(f)
        prompts.append({"text": prompt.get('text'), "purpose": prompt.get('purpose'), "added_at": datetime.now().isoformat()})
        with open(prompt_library, 'w') as f:
            json.dump(prompts, f, indent=2)
        return True
    
    def integrate_research(self, paper: Dict) -> bool:
        research_library = Path("data/research_papers.json")
        papers = []
        if research_library.exists():
            with open(research_library, 'r') as f:
                papers = json.load(f)
        papers.append(paper)
        with open(research_library, 'w') as f:
            json.dump(papers, f, indent=2)
        return True
    
    def integrate_repo(self, repo: Dict) -> bool:
        repo_queue = Path("data/repos_to_study.json")
        repos = []
        if repo_queue.exists():
            with open(repo_queue, 'r') as f:
                repos = json.load(f)
        repos.append(repo)
        with open(repo_queue, 'w') as f:
            json.dump(repos, f, indent=2)
        return True

def initialize_universal_extractor():
    return {"extractor": UniversalScreenshotExtractor(), "integrator": KnowledgeIntegrator()}
