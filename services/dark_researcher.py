#!/usr/bin/env python3
"""Dark Web Researcher - DMAI - Deep web learning for AI innovations"""

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
logger = logging.getLogger("DARK_RESEARCHER")

class DarkResearcher:
    def __init__(self):
        self.root = ROOT
        self.research_dir = self.root / "data" / "research" / "dark"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.findings_file = self.research_dir / "findings.json"
        self.tor_available = self.check_tor()
        self.load_findings()
        logger.info(f"🌑 Dark Web Researcher initialized")
        logger.info(f"📂 Research dir: {self.research_dir}")
        logger.info(f"🔌 Tor available: {self.tor_available}")
        
    def check_tor(self):
        """Check if Tor is available"""
        try:
            # Test Tor proxy
            proxies = {
                'http': 'socks5h://127.0.0.1:9050',
                'https': 'socks5h://127.0.0.1:9050'
            }
            test_url = 'http://check.torproject.org'
            response = requests.get(test_url, proxies=proxies, timeout=10)
            return response.status_code == 200
        except:
            return False
    
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
    
    def search_dark_web(self):
        """Search dark web for AI innovations (if Tor available)"""
        if not self.tor_available:
            logger.warning("⚠️ Tor not available - skipping dark web research")
            return []
        
        proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050'
        }
        
        # Dark web sources (onion sites)
        sources = [
            {
                "name": "Darknet AI Forum",
                "url": "http://ai4dark.onion/forum",
                "type": "forum"
            },
            {
                "name": "Hidden AI Research",
                "url": "http://research7.onion/latest",
                "type": "research"
            },
            {
                "name": "Dark ML Repository",
                "url": "http://mlhub.onion/models",
                "type": "models"
            }
        ]
        
        findings = []
        for source in sources:
            try:
                logger.info(f"🔍 Researching dark source: {source['name']}")
                response = requests.get(source['url'], proxies=proxies, timeout=60)
                
                if response.status_code == 200:
                    finding = {
                        "source": source['name'],
                        "url": source['url'],
                        "type": source['type'],
                        "timestamp": datetime.now().isoformat(),
                        "data": response.text[:2000],
                        "status": "success"
                    }
                    findings.append(finding)
                    logger.info(f"✅ Found data from {source['name']}")
                else:
                    logger.warning(f"⚠️ {source['name']} returned {response.status_code}")
                
                time.sleep(5)  # Be extra polite to onion sites
                
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
                findings.append(finding)
        
        return findings
    
    def search_clearnet_alternatives(self):
        """Search clearnet for AI innovations (fallback when Tor unavailable)"""
        sources = [
            {
                "name": "AI Security Papers",
                "url": "https://arxiv.org/list/cs.CR/recent",
                "type": "arxiv"
            },
            {
                "name": "Privacy Preserving ML",
                "url": "https://github.com/topics/federated-learning",
                "type": "github"
            },
            {
                "name": "Encrypted AI Models",
                "url": "https://huggingface.co/models?search=encrypted",
                "type": "huggingface"
            }
        ]
        
        findings = []
        for source in sources:
            try:
                logger.info(f"🔍 Researching clearnet: {source['name']}")
                headers = {'User-Agent': 'DMAI/1.0 (Research Bot; +http://dmai.local)'}
                response = requests.get(source['url'], timeout=30, headers=headers)
                
                if response.status_code == 200:
                    finding = {
                        "source": source['name'],
                        "url": source['url'],
                        "type": source['type'],
                        "timestamp": datetime.now().isoformat(),
                        "data": response.text[:2000],
                        "status": "success",
                        "note": "Clearnet alternative (Tor unavailable)"
                    }
                    findings.append(finding)
                    logger.info(f"✅ Found data from {source['name']}")
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error researching {source['name']}: {e}")
        
        return findings
    
    def run_once(self):
        """Run one research cycle"""
        logger.info("🔬 Starting dark web research cycle")
        
        # Try dark web first if Tor available
        if self.tor_available:
            findings = self.search_dark_web()
        else:
            logger.info("🌐 Using clearnet alternatives (Tor not available)")
            findings = self.search_clearnet_alternatives()
        
        # Store findings
        cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.findings[cycle_id] = {
            "timestamp": datetime.now().isoformat(),
            "tor_available": self.tor_available,
            "findings_count": len(findings),
            "findings": findings
        }
        
        # Keep only last 50 cycles
        if len(self.findings) > 50:
            oldest = sorted(self.findings.keys())[0]
            del self.findings[oldest]
        
        self.save_findings()
        
        logger.info(f"📊 Research complete: {len(findings)} sources found")
        return len(findings) > 0
    
    def run_continuous(self):
        """Run continuously"""
        logger.info("🌑 Dark Web Researcher started in continuous mode")
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"🔄 Research cycle {cycle}")
            self.run_once()
            logger.info("⏰ Next research in 1 hour")
            time.sleep(3600)  # 1 hour

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMAI Dark Web Researcher")
    parser.add_argument("--test", action="store_true", help="Run one cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    researcher = DarkResearcher()
    
    if args.test:
        logger.info("🧪 Running in TEST mode")
        researcher.run_once()
    elif args.continuous:
        researcher.run_continuous()
    else:
        parser.print_help()
