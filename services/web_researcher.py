#!/usr/bin/env python3
"""Web Researcher - DMAI - Surface web learning for AI innovations"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import time
import json
import logging
import requests
from datetime import datetime
from core.paths import ROOT

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("WEB_RESEARCHER")

class WebResearcher:
    def __init__(self):
        self.root = ROOT
        self.research_dir = self.root / "data" / "research" / "web"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.findings_file = self.research_dir / "findings.json"
        self.load_findings()
        logger.info(f"🌐 Web Researcher initialized")
        logger.info(f"📂 Research dir: {self.research_dir}")
        
    def load_findings(self):
        """Load existing research findings"""
        try:
            if self.findings_file.exists():
                with open(self.findings_file, 'r') as f:
                    self.findings = json.load(f)
                logger.info(f"📚 Loaded {len(self.findings)} previous findings")
            else:
                self.findings = {}
        except Exception as e:
            logger.error(f"Error loading findings: {e}")
            self.findings = {}
    
    def save_findings(self):
        """Save research findings"""
        try:
            with open(self.findings_file, 'w') as f:
                json.dump(self.findings, f, indent=2)
            logger.info(f"💾 Saved {len(self.findings)} findings")
        except Exception as e:
            logger.error(f"Error saving findings: {e}")
    
    def search_ai_innovations(self):
        """Search for AI innovations and news"""
        sources = [
            {
                "name": "HuggingFace Top Models",
                "url": "https://huggingface.co/api/models?sort=downloads&limit=20",
                "type": "api"
            },
            {
                "name": "arXiv AI Papers",
                "url": "https://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=10",
                "type": "arxiv"
            },
            {
                "name": "GitHub AI Trending",
                "url": "https://api.github.com/search/repositories?q=artificial-intelligence&sort=stars&order=desc",
                "type": "github"
            },
            {
                "name": "Reddit Machine Learning",
                "url": "https://www.reddit.com/r/MachineLearning/hot/.json?limit=10",
                "type": "reddit"
            }
        ]
        
        new_findings = []
        for source in sources:
            try:
                logger.info(f"🔍 Researching: {source['name']}")
                headers = {'User-Agent': 'DMAI/1.0 (Research Bot; +http://dmai.local)'}
                response = requests.get(source['url'], timeout=30, headers=headers)
                
                if response.status_code == 200:
                    finding = {
                        "source": source['name'],
                        "url": source['url'],
                        "type": source['type'],
                        "timestamp": datetime.now().isoformat(),
                        "data": response.text[:2000],  # Store preview
                        "status": "success"
                    }
                    new_findings.append(finding)
                    logger.info(f"✅ Found data from {source['name']}")
                else:
                    logger.warning(f"⚠️ {source['name']} returned {response.status_code}")
                
                time.sleep(2)  # Be polite to APIs
                
            except Exception as e:
                logger.error(f"❌ Error researching {source['name']}: {e}")
                finding = {
                    "source": source['name'],
                    "url": source['url'],
                    "type": source['type'],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "failed"
                }
                new_findings.append(finding)
        
        return new_findings
    
    def extract_innovations(self, findings):
        """Extract and process innovations from findings"""
        innovations = []
        
        for finding in findings:
            if finding['status'] != 'success':
                continue
                
            # Simple keyword extraction
            keywords = ['ai', 'artificial intelligence', 'machine learning', 'neural', 
                       'deep learning', 'transformer', 'gpt', 'llm', 'model']
            
            data_lower = finding.get('data', '').lower()
            found_keywords = [k for k in keywords if k in data_lower]
            
            if found_keywords:
                innovation = {
                    "source": finding['source'],
                    "timestamp": finding['timestamp'],
                    "keywords": found_keywords,
                    "relevance": len(found_keywords) / len(keywords)
                }
                innovations.append(innovation)
        
        return innovations
    
    def run_once(self):
        """Run one research cycle"""
        logger.info("🔬 Starting web research cycle")
        
        # Search for innovations
        findings = self.search_ai_innovations()
        
        # Extract insights
        innovations = self.extract_innovations(findings)
        
        # Store findings
        cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.findings[cycle_id] = {
            "timestamp": datetime.now().isoformat(),
            "findings_count": len(findings),
            "innovations_count": len(innovations),
            "findings": findings,
            "innovations": innovations
        }
        
        # Keep only last 100 cycles
        if len(self.findings) > 100:
            oldest = sorted(self.findings.keys())[0]
            del self.findings[oldest]
        
        self.save_findings()
        
        logger.info(f"📊 Research complete: {len(findings)} sources, {len(innovations)} innovations")
        return len(innovations) > 0
    
    def run_continuous(self):
        """Run continuously"""
        logger.info("🌐 Web Researcher started in continuous mode")
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"🔄 Research cycle {cycle}")
            self.run_once()
            logger.info("⏰ Next research in 30 minutes")
            time.sleep(1800)  # 30 minutes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Web Researcher")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    researcher = WebResearcher()
    
    if args.test:
        logger.info("🧪 Running in TEST mode")
        researcher.run_once()
    elif args.continuous:
        researcher.run_continuous()
    else:
        parser.print_help()
