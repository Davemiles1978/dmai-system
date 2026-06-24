"""
DYNAMIC REVENUE DISCOVERY - Autonomous Revenue Opportunity Detection
Runs daily to discover new revenue options and request account setup assistance
"""

import os
import json
import threading
import time
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicRevenueDiscovery:
    """
    Autonomous revenue opportunity discovery
    - Scans for new revenue options daily
    - Analyzes API availability
    - Requests account setup assistance from master
    """
    
    def __init__(self, data_path: Path, knowledge_graph, ai_hub, funding_orchestrator):
        self.data_path = data_path
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.funding = funding_orchestrator
        self.discovery_dir = data_path / 'revenue_discovery'
        self.discovery_dir.mkdir(parents=True, exist_ok=True)
        
        self.discovered_opportunities = []
        self.setup_requests = []
        self.last_scan = None
        self.running = False
        self.scan_thread = None
        
        # State file
        self.state_file = self.discovery_dir / 'discovered_opportunities.json'
        self._load_state()
        
        logger.info("💰 Dynamic Revenue Discovery initialized")
    
    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.discovered_opportunities = data.get('opportunities', [])
                    self.setup_requests = data.get('setup_requests', [])
                    self.last_scan = data.get('last_scan')
            except:
                pass
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'opportunities': self.discovered_opportunities[-100:],
                'setup_requests': self.setup_requests,
                'last_scan': self.last_scan
            }, f, indent=2)
    
    def start_daily_discovery(self):
        """Start daily discovery loop"""
        if self.running:
            return
        
        self.running = True
        self.scan_thread = threading.Thread(target=self._daily_scan_loop, daemon=True)
        self.scan_thread.start()
        logger.info("🔄 Daily revenue discovery started")
    
    def _daily_scan_loop(self):
        """Run discovery once per day"""
        while self.running:
            try:
                self.scan_for_new_opportunities()
                # Sleep for 24 hours
                for _ in range(86400):  # 24 hours in seconds
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Revenue discovery error: {e}")
                time.sleep(3600)  # Retry in 1 hour
    
    def scan_for_new_opportunities(self) -> Dict:
        """
        Scan for new revenue opportunities across:
        - Emerging platforms (Patreon alternatives, new creator platforms)
        - New API services
        - Affiliate programs
        - Grant opportunities
        """
        self.last_scan = datetime.now().isoformat()
        new_opportunities = []
        
        try:
            # 1. Query AI tutors for emerging revenue platforms
            if self.ai_hub and self.ai_hub._get_active_tutors():
                prompt = """
                List 5-10 emerging revenue platforms, API services, or monetization opportunities 
                that DMAI could leverage. Include:
                - Platform/service name
                - Type of revenue (subscription, API calls, affiliate, grants)
                - API availability (yes/no)
                - Account setup required
                - Estimated potential (low/medium/high)
                - URL if available
                
                Focus on legitimate, established platforms with developer APIs.
                """
                
                result = self.ai_hub.query_all_tutors(prompt)
                if result.get('responses'):
                    for tutor, response in result.get('responses', {}).items():
                        if response and isinstance(response, str):
                            opportunities = self._parse_opportunities(response)
                            new_opportunities.extend(opportunities)
            
            # 2. Scan known revenue API directories
            new_opportunities.extend(self._scan_api_directories())
            
            # 3. Filter out already discovered opportunities
            existing_names = {opp.get('name') for opp in self.discovered_opportunities}
            truly_new = [opp for opp in new_opportunities if opp.get('name') not in existing_names]
            
            # 4. Add to discovered list, scoring each via Microfish PredictionEngine
            for opp in truly_new:
                opp['discovered_at'] = datetime.now().isoformat()
                opp['status'] = 'pending_review'
                # Microfish prediction-driven scoring
                try:
                    score = self.score_opportunity(opp)
                    if score:
                        opp['prediction_score'] = score
                except Exception as _e:
                    logger.warning(f"score_opportunity failed for {opp.get('name')}: {_e}")
                self.discovered_opportunities.append(opp)
                
                # Check if account setup is needed
                if opp.get('requires_account', True) and opp.get('api_available', False):
                    self._request_account_setup(opp)
            
            self._save_state()
            
            if truly_new:
                logger.info(f"🎉 Discovered {len(truly_new)} new revenue opportunities")
                for opp in truly_new:
                    logger.info(f"   📌 {opp['name']}: {opp.get('type', 'Unknown')}")
            
            return {'success': True, 'new_opportunities': len(truly_new), 'opportunities': truly_new}
            
        except Exception as e:
            logger.error(f"Revenue discovery scan error: {e}")
            return {'success': False, 'error': str(e)}
    
    def score_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """Score a revenue opportunity via Microfish PredictionEngine.
        Returns {verdict, probability_viable, confidence, rationale} or None."""
        try:
            from dmai_core_complete import components as _comp  # type: ignore
            engine = _comp.get("prediction_engine")
        except Exception:
            engine = None
        if not engine:
            return None
        name = opportunity.get('name', 'unknown')
        seed = (
            f"Opportunity name: {name}\n"
            f"Type: {opportunity.get('type', 'unknown')}\n"
            f"API available: {opportunity.get('api_available', False)}\n"
            f"Estimated potential: {opportunity.get('potential', 'unknown')}\n"
            f"URL: {opportunity.get('url', 'n/a')}\n"
            f"Description: {opportunity.get('description', '')}\n"
        )
        verdict = engine.predict(
            requirement=f"Is the revenue opportunity '{name}' likely to produce meaningful recurring revenue (>$100/month) for an autonomous AI agent within 90 days of integration?",
            seed_data=seed,
            max_rounds=2,
            agent_count=3,
        )
        return {
            "verdict": verdict.get("verdict"),
            "probability_viable": verdict.get("probability"),
            "confidence": verdict.get("confidence"),
            "rationale": verdict.get("rationale", ""),
            "prediction_id": verdict.get("id"),
        }

    def _parse_opportunities(self, text: str) -> List[Dict]:
        """Parse AI response into structured opportunities"""
        opportunities = []
        
        # Common revenue platforms to check for
        known_platforms = [
            ('GitHub Sponsors', 'sponsorship', True, 'Developer funding platform'),
            ('Ko-fi', 'patronage', True, 'Creator support platform'),
            ('Buy Me a Coffee', 'patronage', True, 'Creator support platform'),
            ('Substack', 'subscription', True, 'Newsletter platform'),
            ('Ghost', 'subscription', True, 'Membership platform'),
            ('Podia', 'education', True, 'Digital products platform'),
            ('Teachable', 'education', True, 'Course platform'),
            ('Rakuten', 'affiliate', True, 'Affiliate network'),
            ('ShareASale', 'affiliate', True, 'Affiliate network'),
            ('RapidAPI', 'api_services', True, 'API marketplace'),
            ('Stripe Connect', 'payment', True, 'Payment processing'),
            ('Lemon Squeezy', 'payment', True, 'Payment processing'),
            ('OpenAI API', 'api_services', True, 'AI API services'),
            ('Replicate', 'api_services', True, 'Model hosting'),
            ('Hugging Face', 'api_services', True, 'Model hosting'),
            ('RunPod', 'cloud', True, 'GPU cloud services'),
            ('Vast.ai', 'cloud', True, 'GPU cloud services'),
            ('Together AI', 'api_services', True, 'AI API services'),
            ('Anthropic API', 'api_services', True, 'AI API services'),
            ('Google Cloud', 'cloud', True, 'Cloud services'),
        ]
        
        # Check for known platforms in text
        for platform_name, platform_type, has_api, description in known_platforms:
            if platform_name.lower() in text.lower():
                # Check if already discovered
                existing = [o for o in self.discovered_opportunities if o.get('name') == platform_name]
                if not existing:
                    opportunities.append({
                        'name': platform_name,
                        'type': platform_type,
                        'api_available': has_api,
                        'requires_account': True,
                        'description': description,
                        'source': 'ai_tutor'
                    })
        
        return opportunities
    
    def _scan_api_directories(self) -> List[Dict]:
        """Scan API directories for monetizable APIs"""
        opportunities = []
        
        # Known API marketplaces with revenue potential
        api_directories = [
            {'name': 'RapidAPI Hub', 'url': 'https://rapidapi.com/hub', 'type': 'api_services'},
            {'name': 'APILayer', 'url': 'https://apilayer.com', 'type': 'api_services'},
            {'name': 'Marketplace by API', 'url': 'https://marketplace.api.gov', 'type': 'api_services'},
        ]
        
        # In production, this would actually scrape these sites
        # For now, we add known opportunities
        
        return opportunities
    
    def _request_account_setup(self, opportunity: Dict):
        """
        Create a setup request for master to assist with account creation
        """
        request = {
            'opportunity_name': opportunity.get('name'),
            'opportunity_type': opportunity.get('type'),
            'requires_api_key': opportunity.get('api_available', True),
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'steps': self._generate_setup_steps(opportunity)
        }
        
        self.setup_requests.append(request)
        self._save_state()
        
        logger.warning(f"📢 ACCOUNT SETUP NEEDED: {opportunity.get('name')}")
        logger.warning(f"   Steps: {request['steps']}")
    
    def _generate_setup_steps(self, opportunity: Dict) -> List[str]:
        """Generate setup steps for master to follow"""
        steps = [
            f"1. Visit {opportunity.get('name')} website",
            "2. Click 'Sign Up' or 'Get Started'",
            f"3. Create account for {opportunity.get('name')}",
        ]
        
        if opportunity.get('api_available'):
            steps.append("4. Navigate to Developer/API section")
            steps.append("5. Generate API key")
            steps.append("6. Provide API key to DMAI")
        
        return steps
    
    def get_pending_setup_requests(self) -> List[Dict]:
        """Get all pending account setup requests for master"""
        return [r for r in self.setup_requests if r.get('status') == 'pending']
    
    def mark_setup_complete(self, opportunity_name: str, api_key: str = None):
        """Mark a setup request as complete"""
        for req in self.setup_requests:
            if req.get('opportunity_name') == opportunity_name:
                req['status'] = 'complete'
                req['completed_at'] = datetime.now().isoformat()
                if api_key:
                    req['api_key_provided'] = True
                self._save_state()
                logger.info(f"✅ Account setup complete for {opportunity_name}")
                return True
        return False
    
    def get_discovered_opportunities(self) -> List[Dict]:
        """Get all discovered opportunities"""
        return self.discovered_opportunities
    
    def get_opportunity_by_name(self, name: str) -> Optional[Dict]:
        """Get specific opportunity details"""
        for opp in self.discovered_opportunities:
            if opp.get('name') == name:
                return opp
        return None
