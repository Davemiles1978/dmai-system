"""
SELF-FUNDING TRAINING - PHASE 1: COMPREHENSIVE KNOWLEDGE ACQUISITION
Teaches DMAI about ALL potential revenue streams through AI tutor learning.
NO simulation. NO execution. Pure knowledge acquisition.

Phase 1 covers 10 revenue avenues:
1. Quantitative Trading - Market analysis, strategy development
2. Content Creation - Blog, video, social media strategies
3. AI Services - API services, model hosting, consulting
4. Software Products - SaaS, tools, libraries
5. Affiliate/Referral - Strategic partnerships, commissions
6. Data Services - Data APIs, analytics, insights
7. Education/Training - Courses, tutorials, mentorship
8. Consulting/Analysis - Expert services, research reports
9. Ad Revenue - Content monetization, sponsorships
10. Crowdfunding/Patronage - Patreon, sponsorships, grants

FUTURE PHASES:
- Phase 2: Paper Execution (requires master approval)
- Phase 3: Real Execution (requires master capital + approval)
"""

import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SelfFundingTraining:
    """
    Phase 1: Comprehensive knowledge acquisition about ALL self-funding avenues.
    No execution. No simulation. Pure learning.
    """
    
    def __init__(self, data_path: Path, financial_manager, knowledge_graph, ai_hub):
        self.data_path = data_path
        self.financial_manager = financial_manager
        self.knowledge_graph = knowledge_graph
        self.ai_hub = ai_hub
        self.training_dir = data_path / 'training' / 'funding'
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # ====================================================================
        # COMPREHENSIVE REVENUE AVENUES - What DMAI will learn about
        # ====================================================================
        self.revenue_avenues = {
            # 1. QUANTITATIVE TRADING
            'quant_trading': {
                'name': 'Quantitative Trading',
                'description': 'Algorithmic trading, market making, arbitrage',
                'topics': [
                    'market_microstructure', 'technical_analysis', 'fundamental_analysis',
                    'risk_management', 'trading_psychology', 'strategy_development',
                    'execution_mechanics', 'regulatory_compliance', 'backtesting_methodology',
                    'portfolio_optimization', 'alpha_generation', 'high_frequency_concepts'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 2. CONTENT CREATION
            'content_creation': {
                'name': 'Content Creation',
                'description': 'Blog posts, videos, social media, podcasts',
                'topics': [
                    'audience_growth', 'content_strategy', 'seo_optimization',
                    'video_production', 'writing_techniques', 'social_media_algorithms',
                    'engagement_metrics', 'content_calendars', 'platform_specific_strategies',
                    'viral_mechanics', 'storytelling', 'brand_building'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 3. AI SERVICES
            'ai_services': {
                'name': 'AI Services',
                'description': 'API services, model hosting, AI consulting',
                'topics': [
                    'api_design', 'model_deployment', 'pricing_strategies',
                    'service_level_agreements', 'customer_acquisition', 'infrastructure_costs',
                    'scaling_considerations', 'competitive_landscape', 'use_case_identification',
                    'value_proposition', 'technical_support', 'documentation_best_practices'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 4. SOFTWARE PRODUCTS
            'software_products': {
                'name': 'Software Products',
                'description': 'SaaS, tools, libraries, applications',
                'topics': [
                    'product_market_fit', 'saas_pricing_models', 'customer_lifetime_value',
                    'churn_reduction', 'feature_prioritization', 'open_source_strategies',
                    'freemium_conversion', 'distribution_channels', 'product_led_growth',
                    'technical_debt_management', 'user_onboarding', 'feedback_loops'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 5. AFFILIATE & REFERRAL
            'affiliate_referral': {
                'name': 'Affiliate & Referral',
                'description': 'Strategic partnerships, commissions, referral programs',
                'topics': [
                    'affiliate_networks', 'commission_structures', 'partnership_development',
                    'referral_program_design', 'tracking_attribution', 'incentive_alignment',
                    'cross_promotion', 'influencer_marketing', 'b2b_partnerships',
                    'revenue_sharing_models', 'contract_negotiation', 'performance_metrics'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 6. DATA SERVICES
            'data_services': {
                'name': 'Data Services',
                'description': 'Data APIs, analytics, insights, research',
                'topics': [
                    'data_collection', 'data_processing', 'insight_generation',
                    'api_monetization', 'data_quality_assurance', 'privacy_compliance',
                    'real_time_analytics', 'custom_research', 'benchmarking_services',
                    'data_visualization', 'predictive_insights', 'market_intelligence'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 7. EDUCATION & TRAINING
            'education_training': {
                'name': 'Education & Training',
                'description': 'Courses, tutorials, mentorship, certifications',
                'topics': [
                    'course_creation', 'curriculum_design', 'learning_objectives',
                    'pricing_strategies', 'student_acquisition', 'engagement_retention',
                    'certification_programs', 'cohort_based_courses', 'mentorship_models',
                    'corporate_training', 'workshop_facilitation', 'educational_partnerships'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 8. CONSULTING & ANALYSIS
            'consulting_analysis': {
                'name': 'Consulting & Analysis',
                'description': 'Expert services, research reports, strategic advice',
                'topics': [
                    'consulting_engagements', 'statement_of_work', 'hourly_vs_project_pricing',
                    'expert_networks', 'research_reports', 'white_papers',
                    'strategic_recommendations', 'client_management', 'deliverable_standards',
                    'intellectual_property', 'non_disclosure_agreements', 'reputation_building'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 9. AD REVENUE
            'ad_revenue': {
                'name': 'Ad Revenue',
                'description': 'Display ads, sponsorships, programmatic advertising',
                'topics': [
                    'ad_networks', 'cpm_vs_cpc', 'programmatic_advertising',
                    'sponsorship_models', 'audience_demographics', 'ad_placement_optimization',
                    'yield_management', 'direct_sales', 'ad_fraud_prevention',
                    'privacy_compliance', 'viewability_metrics', 'native_advertising'
                ],
                'progress': 0.0,
                'completed': False
            },
            
            # 10. CROWDFUNDING & PATRONAGE
            'crowdfunding_patronage': {
                'name': 'Crowdfunding & Patronage',
                'description': 'Patreon, GitHub Sponsors, grants, crowdfunding',
                'topics': [
                    'patreon_models', 'membership_tiers', 'creator_economy',
                    'crowdfunding_campaigns', 'grant_proposals', 'sponsorship_packages',
                    'community_building', 'exclusive_content', 'donor_retention',
                    'fundraising_strategies', 'impact_reporting', 'open_source_funding'
                ],
                'progress': 0.0,
                'completed': False
            }
        }
        
        # Track learned concepts
        self.learned_concepts = set()
        
        # Strategy candidates per avenue
        self.strategy_candidates = {
            'quant_trading': [],
            'content_creation': [],
            'ai_services': [],
            'software_products': [],
            'affiliate_referral': [],
            'data_services': [],
            'education_training': [],
            'consulting_analysis': [],
            'ad_revenue': [],
            'crowdfunding_patronage': []
        }
        
        # Learning active flag
        self.learning_active = False
        self.learning_thread = None
        self._training_complete = False
        
        # Current learning focus
        self.current_avenue = None
        self.current_topic = None
        
        # State file
        self.state_file = self.training_dir / 'knowledge_state.json'
        self._load_state()
        
        total_concepts = sum(len(d['topics']) for d in self.revenue_avenues.values())
        logger.info(f"💰 Self-Funding Training initialized (PHASE 1: Comprehensive Knowledge Acquisition)")
        logger.info(f"   Revenue avenues: {len(self.revenue_avenues)}")
        logger.info(f"   Total concepts to learn: {total_concepts}")
        for avenue_name, avenue in self.revenue_avenues.items():
            logger.info(f"      📚 {avenue['name']}: {len(avenue['topics'])} topics")
    
    # ====================================================================
    # NEO4J PERSISTENCE METHODS
    # ====================================================================
    
    def _save_to_neo4j(self):
        """Save funding state to Neo4j for persistence across restarts"""
        try:
            from neo4j import GraphDatabase
            
            uri = os.environ.get('NEO4J_URI')
            user = os.environ.get('NEO4J_USER')
            password = os.environ.get('NEO4J_PASSWORD')
            
            if not uri or not user or not password:
                logger.debug("Neo4j credentials not available, skipping funding state save")
                return
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Save main funding state
                session.run("""
                    MERGE (f:FundingState {id: 'core'})
                    SET f.completed_avenues = $completed_avenues,
                        f.concepts_learned = $concepts_learned,
                        f.concepts_total = $concepts_total,
                        f.learning_active = $learning_active,
                        f.training_complete = $training_complete,
                        f.progress = $progress,
                        f.updated_at = datetime()
                """, {
                    'completed_avenues': [name for name, data in self.revenue_avenues.items() if data.get('completed', False)],
                    'concepts_learned': len(self.learned_concepts),
                    'concepts_total': sum(len(d['topics']) for d in self.revenue_avenues.values()),
                    'learning_active': self.learning_active,
                    'training_complete': self._training_complete,
                    'progress': self.get_progress()
                })
                
                # Save each avenue's progress
                for avenue_name, avenue in self.revenue_avenues.items():
                    session.run("""
                        MERGE (a:FundingAvenue {id: $avenue_name})
                        SET a.name = $name,
                            a.progress = $progress,
                            a.completed = $completed,
                            a.updated_at = datetime()
                    """, {
                        'avenue_name': avenue_name,
                        'name': avenue['name'],
                        'progress': avenue.get('progress', 0),
                        'completed': avenue.get('completed', False)
                    })
                
                # Save learned concepts as nodes
                for concept in self.learned_concepts:
                    session.run("""
                        MERGE (c:FundingConcept {id: $concept})
                        SET c.learned_at = datetime()
                    """, {'concept': concept})
                
            driver.close()
            logger.debug(f"✅ Funding state saved to Neo4j: {len(self.learned_concepts)} concepts, {self.get_progress():.1f}% complete")
        except Exception as e:
            logger.error(f"Failed to save funding state to Neo4j: {e}")
    
    def _load_from_neo4j(self) -> bool:
        """Load funding state from Neo4j"""
        try:
            from neo4j import GraphDatabase
            
            uri = os.environ.get('NEO4J_URI')
            user = os.environ.get('NEO4J_USER')
            password = os.environ.get('NEO4J_PASSWORD')
            
            if not uri or not user or not password:
                return False
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Load main funding state
                result = session.run("MATCH (f:FundingState {id: 'core'}) RETURN f")
                record = result.single()
                if record:
                    data = record['f']
                    # Restore learned concepts count and completion status
                    self._training_complete = data.get('training_complete', False)
                    
                    # Load avenue progress
                    avenue_result = session.run("MATCH (a:FundingAvenue) RETURN a.id as id, a.progress as progress, a.completed as completed")
                    for avenue in avenue_result:
                        if avenue['id'] in self.revenue_avenues:
                            self.revenue_avenues[avenue['id']]['progress'] = avenue['progress'] or 0
                            self.revenue_avenues[avenue['id']]['completed'] = avenue['completed'] or False
                    
                    # Load learned concepts
                    concept_result = session.run("MATCH (c:FundingConcept) RETURN c.id as concept")
                    for concept in concept_result:
                        self.learned_concepts.add(concept['concept'])
                    
                    logger.info(f"✅ Funding state loaded from Neo4j: {len(self.learned_concepts)} concepts, {self.get_progress():.1f}% complete")
                    driver.close()
                    return True
            driver.close()
        except Exception as e:
            logger.warning(f"Neo4j unavailable — funding state will use in-memory defaults ({e})")
        return False
    
    def _load_state(self):
        """Load learning state - tries Neo4j first, then local file"""
        # Try Neo4j first for persistence across deployments
        if self._load_from_neo4j():
            self._save_state()  # Sync local file with Neo4j data
            return
        
        # Fallback to local file
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    for avenue_name, avenue_data in state.get('revenue_avenues', {}).items():
                        if avenue_name in self.revenue_avenues:
                            self.revenue_avenues[avenue_name]['progress'] = avenue_data.get('progress', 0)
                            self.revenue_avenues[avenue_name]['completed'] = avenue_data.get('completed', False)
                    self.learned_concepts = set(state.get('learned_concepts', []))
                    self.strategy_candidates = state.get('strategy_candidates', self.strategy_candidates)
                    self.learning_active = False  # NEVER auto-start on load
                    self._training_complete = state.get('training_complete', False)
                    
                    # Check if training is complete
                    all_completed = all(data['completed'] for data in self.revenue_avenues.values())
                    if all_completed or len(self.learned_concepts) >= sum(len(d['topics']) for d in self.revenue_avenues.values()):
                        self._training_complete = True
                    
                    logger.info(f"📂 Loaded funding knowledge state from local file: {len(self.learned_concepts)} concepts learned")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save learning state - saves to both local file AND Neo4j for persistence"""
        try:
            # Save to local file
            state = {
                'revenue_avenues': self.revenue_avenues,
                'learned_concepts': list(self.learned_concepts),
                'strategy_candidates': self.strategy_candidates,
                'learning_active': False,  # Always save as False - state restored only on crash recovery
                'training_complete': self._training_complete,
                'phase': '2 - Paper Execution (Strategy Testing)' if self._ready_for_phase_2() else '1 - Comprehensive Knowledge Acquisition',
                'ready_for_phase_2': self._ready_for_phase_2(),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Also save to Neo4j for cross-deployment persistence
            self._save_to_neo4j()
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    # ====================================================================
    # STANDARDIZED METHODS - Called by main system
    # ====================================================================
    
    def start(self):
        """Standardized start method - starts or resumes learning"""
        return self.start_learning()
    
    def update(self):
        """Standardized update method - called during evolution cycles"""
        if not self.learning_active:
            return
        
        # Update progress calculation
        total_concepts = sum(len(d['topics']) for d in self.revenue_avenues.values())
        if total_concepts > 0:
            progress = (len(self.learned_concepts) / total_concepts) * 100
            progress = min(100, progress)
            
            # Update each avenue's progress
            for avenue_name, avenue in self.revenue_avenues.items():
                learned_in_avenue = sum(1 for t in avenue['topics'] if t in self.learned_concepts)
                avenue['progress'] = learned_in_avenue / len(avenue['topics']) * 100
                if learned_in_avenue >= len(avenue['topics']):
                    avenue['completed'] = True
            
            self._save_state()
            
            if progress >= 99.9:
                self._training_complete = True
                self.learning_active = False
                logger.info("💰 Self-Funding Training COMPLETE!")
    
    def start_learning(self, avenue: str = None) -> Dict:
        """
        Start knowledge acquisition about revenue avenues.
        
        Args:
            avenue: Optional specific avenue to focus on. If None, learns all sequentially.
        
        Returns:
            Dict with status information
        """
        # Block if training already complete
        if self._training_complete:
            return {
                'success': False, 
                'error': 'Training already completed',
                'message': 'Self-Funding Training is already 100% complete'
            }
        
        # Block if learning already active
        if self.learning_active:
            return {
                'success': False, 
                'error': 'Learning already active',
                'message': 'Self-Funding learning is already running'
            }
        
        if avenue and avenue not in self.revenue_avenues:
            return {
                'success': False,
                'error': f'Unknown avenue: {avenue}',
                'available_avenues': list(self.revenue_avenues.keys())
            }
        
        self.learning_active = True
        self.current_avenue = avenue
        self._save_state()
        self.learning_thread = threading.Thread(target=self._run_knowledge_acquisition, daemon=True)
        self.learning_thread.start()
        
        if avenue:
            logger.info(f"📚 Starting focused learning: {self.revenue_avenues[avenue]['name']}")
        else:
            logger.info("📚 Starting comprehensive revenue knowledge acquisition (ALL avenues)")
        
        return {
            'success': True,
            'mode': 'knowledge_acquisition',
            'phase': '1 - Comprehensive Knowledge',
            'message': f'Learning about {avenue if avenue else "all revenue avenues"} from AI tutors. No execution will occur.',
            'avenues': [avenue] if avenue else list(self.revenue_avenues.keys()),
            'total_concepts': sum(len(d['topics']) for d in self.revenue_avenues.values())
        }
    
    def stop_learning(self) -> Dict:
        """Stop/knowledge acquisition"""
        if not self.learning_active:
            return {
                'success': False,
                'error': 'Learning not active',
                'message': 'Self-Funding learning is not currently running'
            }
        
        self.learning_active = False
        self._save_state()
        
        logger.info("⏸️ Self-Funding Learning PAUSED")
        return {
            'success': True,
            'message': 'Self-Funding learning paused',
            'progress': self.get_progress()
        }
    
    def crash_recovery(self) -> Dict:
        """
        Called when system restarts after crash/power cut
        Auto-resumes learning if it was active before crash
        ONLY for crash recovery - does NOT auto-start on normal boot
        """
        if self._training_complete:
            logger.info("✅ Self-Funding Training already complete - no recovery needed")
            return {'recovered': False, 'reason': 'already_complete'}
        
        # Check if we have incomplete learning (progress > 0 but not complete)
        current_progress = self.get_progress()
        if current_progress > 0 and current_progress < 100 and not self._training_complete:
            logger.info(f"🔄 CRASH RECOVERY: Resuming Self-Funding Learning from {current_progress:.1f}%")
            return self.start_learning()
        
        logger.info("📋 Self-Funding Learning has no incomplete state to recover")
        return {'recovered': False, 'reason': 'no_incomplete_learning'}
    
    def get_status(self) -> Dict:
        """Get learning status - reads directly from state file for accuracy"""
        try:
            # Always read from the actual state file
            if self.state_file and self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                learned_concepts = state.get('learned_concepts', [])
                revenue_avenues = state.get('revenue_avenues', {})
                completed_avenues = [k for k, v in revenue_avenues.items() if v.get('completed', False)]
                total_concepts = sum(len(v.get('topics', [])) for v in revenue_avenues.values())
                learned_count = len(learned_concepts)
                overall_progress = (learned_count / total_concepts * 100) if total_concepts > 0 else 0
                
                # Get strategy counts from state
                strategy_candidates = state.get('strategy_candidates', {})
                
                return {
                    'active': self.learning_active,
                    'phase': state.get('phase', '1 - Comprehensive Knowledge Acquisition'),
                    'overall_progress_percent': overall_progress,
                    'progress': overall_progress,
                    'complete': state.get('training_complete', False) or overall_progress >= 99.9,
                    'concepts_learned': learned_count,
                    'concepts_total': total_concepts,
                    'completed_avenues': completed_avenues,
                    'completed_avenues_count': len(completed_avenues),
                    'total_avenues': len(revenue_avenues),
                    'revenue_avenues': revenue_avenues,
                    'strategy_candidates': {
                        avenue: len(candidates) if isinstance(candidates, list) else 0
                        for avenue, candidates in strategy_candidates.items()
                    },
                    'ready_for_phase_2': state.get('ready_for_phase_2', False),
                    'message': 'Knowledge acquisition mode - no revenue generation occurs',
                    'can_start': not self.learning_active and not (state.get('training_complete', False) or overall_progress >= 99.9),
                    'can_stop': self.learning_active,
                    'status': 'training' if self.learning_active else 'paused'
                }
            else:
                # Fallback to in-memory if file doesn't exist
                total_concepts = sum(len(d['topics']) for d in self.revenue_avenues.values())
                learned_count = len(self.learned_concepts)
                overall_progress = (learned_count / total_concepts * 100) if total_concepts > 0 else 0
                
                return {
                    'active': self.learning_active,
                    'phase': '1 - Comprehensive Knowledge Acquisition',
                    'overall_progress_percent': overall_progress,
                    'progress': overall_progress,
                    'complete': self._training_complete or overall_progress >= 99.9,
                    'concepts_learned': learned_count,
                    'concepts_total': total_concepts,
                    'completed_avenues': [name for name, data in self.revenue_avenues.items() if data['completed']],
                    'completed_avenues_count': len([name for name, data in self.revenue_avenues.items() if data['completed']]),
                    'total_avenues': len(self.revenue_avenues),
                    'revenue_avenues': self.revenue_avenues,
                    'strategy_candidates': {avenue: len(candidates) for avenue, candidates in self.strategy_candidates.items()},
                    'ready_for_phase_2': self._ready_for_phase_2(),
                    'message': 'Knowledge acquisition mode - no revenue generation occurs',
                    'can_start': not self.learning_active and not (self._training_complete or overall_progress >= 99.9),
                    'can_stop': self.learning_active,
                    'status': 'training' if self.learning_active else 'paused'
                }
        except Exception as e:
            logger.error(f"Status read error: {e}")
            # Return cached status as fallback
            total_concepts = sum(len(d['topics']) for d in self.revenue_avenues.values())
            learned_count = len(self.learned_concepts)
            overall_progress = (learned_count / total_concepts * 100) if total_concepts > 0 else 0
            return {
                'active': self.learning_active,
                'phase': '1 - Comprehensive Knowledge Acquisition',
                'overall_progress_percent': overall_progress,
                'progress': overall_progress,
                'complete': self._training_complete or overall_progress >= 99.9,
                'concepts_learned': learned_count,
                'concepts_total': total_concepts,
                'message': 'Knowledge acquisition mode - no revenue generation occurs (fallback)',
                'status': 'training' if self.learning_active else 'paused'
            }
    
    def get_progress(self) -> float:
        """Return current progress percentage for dashboard display"""
        total_concepts = sum(len(d['topics']) for d in self.revenue_avenues.values())
        learned_count = len(self.learned_concepts)
        return (learned_count / total_concepts * 100) if total_concepts > 0 else 0
    
    def _ready_for_phase_2(self) -> bool:
        """Check if DMAI has learned enough to proceed to Phase 2 (paper execution)"""
        # Require all revenue avenues to be completed
        all_completed = all(data['completed'] for data in self.revenue_avenues.values())
        
        # Require at least one strategy candidate per avenue
        has_strategies = all(
            len(self.strategy_candidates[avenue]) >= 1 
            for avenue in self.revenue_avenues.keys()
        )
        
        return all_completed and has_strategies
    
    def _run_knowledge_acquisition(self):
        """Main learning loop - NO EXECUTION, just knowledge"""
        logger.info("📚 Comprehensive Revenue Knowledge Acquisition thread started")
        
        try:
            # Determine learning order
            if self.current_avenue:
                avenues_to_learn = [(self.current_avenue, self.revenue_avenues[self.current_avenue])]
            else:
                avenues_to_learn = list(self.revenue_avenues.items())
            
            for avenue_name, avenue in avenues_to_learn:
                if not self.learning_active:
                    break
                
                # Skip if avenue already completed
                if avenue['completed']:
                    logger.info(f"📚 Avenue already completed: {avenue['name']}")
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Learning Avenue: {avenue['name']}")
                logger.info(f"   {avenue['description']}")
                logger.info(f"{'='*60}")
                
                for topic in avenue['topics']:
                    if not self.learning_active:
                        break
                    
                    if topic in self.learned_concepts:
                        continue
                    
                    self.current_topic = topic
                    logger.info(f"   📖 Learning: {topic}")
                    
                    # Learn from AI tutors
                    knowledge = self._learn_concept(topic, avenue_name, avenue['name'])
                    
                    # Store in knowledge graph with full context
                    concept_key = f"funding_{avenue_name}_{topic}"
                    self.knowledge_graph.add_knowledge(
                        subject=concept_key,
                        predicate="is_about",
                        object=topic,
                        metadata={
                            'avenue': avenue_name,
                            'avenue_name': avenue['name'],
                            'topic': topic,
                            'learned_at': datetime.now().isoformat(),
                            'knowledge_length': len(knowledge)
                        }
                    )
                    
                    # Also add the knowledge content
                    self.knowledge_graph.add_concept(concept_key, knowledge[:500])
                    
                    # Notify UnifiedLearningOrchestrator
                    
                    if hasattr(self, 'unified_learning'):
                    
                        self.unified_learning.on_concept_mastered("funding", topic, {'avenue': avenue_name})
                    
                    self.learned_concepts.add(topic)
                    
                    # Update avenue progress
                    learned_in_avenue = sum(1 for t in avenue['topics'] if t in self.learned_concepts)
                    avenue['progress'] = learned_in_avenue / len(avenue['topics']) * 100
                    
                    # Check if avenue is complete
                    if learned_in_avenue >= len(avenue['topics']):
                        avenue['completed'] = True
                        logger.info(f"   ✅ COMPLETED: {avenue['name']} - All topics mastered!")
                        
                        # Generate strategy candidates for this avenue
                        self._generate_strategy_candidates(avenue_name)
                    
                    self._save_state()
                    
                    logger.info(f"      ✅ Learned: {topic} ({avenue['progress']:.1f}% complete)")
                    time.sleep(0.5)  # Brief pause to prevent rate limiting
            
            # Check if all avenues complete
            all_completed = all(data['completed'] for data in self.revenue_avenues.values())
            if all_completed:
                self._training_complete = True
                self._save_state()
            
        except Exception as e:
            logger.error(f"Knowledge acquisition thread error: {e}")
            self._save_state()
        finally:
            self.learning_active = False
            self._save_state()
        
        # Final completion summary
        completed_count = sum(1 for a in self.revenue_avenues.values() if a['completed'])
        
        logger.info(f"\n{'='*60}")
        if self._training_complete:
            logger.info("🎉 COMPREHENSIVE REVENUE KNOWLEDGE ACQUISITION COMPLETE!")
        else:
            logger.info("⏸️ COMPREHENSIVE REVENUE KNOWLEDGE ACQUISITION PAUSED!")
        logger.info(f"   Concepts learned: {len(self.learned_concepts)}")
        logger.info(f"   Avenues completed: {completed_count}/{len(self.revenue_avenues)}")
        logger.info(f"   Strategy candidates: {sum(len(c) for c in self.strategy_candidates.values())}")
        logger.info(f"\n📋 Ready for Phase 2 (Paper Execution): {self._ready_for_phase_2()}")
        logger.info(f"{'='*60}")
    
    def _learn_concept(self, concept: str, avenue: str, avenue_name: str) -> str:
        """Learn a concept from AI tutors about a revenue avenue"""
        try:
            if self.ai_hub and hasattr(self.ai_hub, '_get_active_tutors') and self.ai_hub._get_active_tutors():
                prompt = f"""Teach me about {concept} for {avenue_name} ({avenue}) as a revenue generation strategy.

Provide comprehensive knowledge including:
1. Core definition and importance
2. Key principles and mechanics
3. Practical implementation strategies
4. Success metrics and KPIs
5. Common mistakes and how to avoid them
6. Real-world examples and case studies

Focus on actionable, practical knowledge that can be applied."""
                
                result = self.ai_hub.query_all_tutors(prompt)
                if result and result.get('responses'):
                    for tutor, response in result.get('responses', {}).items():
                        if response and isinstance(response, str) and len(response) > 100:
                            return response[:2000]
        except Exception as e:
            logger.debug(f"AI tutor learning failed for {concept}: {e}")
        
        return f"Revenue knowledge: {concept} in {avenue_name}. [Will be populated by AI tutors when available]"
    
    def _generate_strategy_candidates(self, avenue_name: str):
        """Generate strategy candidates for a completed revenue avenue"""
        avenue = self.revenue_avenues[avenue_name]
        logger.info(f"   🧠 Generating strategy candidates for {avenue['name']}...")
        
        try:
            if self.ai_hub and hasattr(self.ai_hub, '_get_active_tutors'):
                prompt = f"""Based on your knowledge of {avenue['name']} ({avenue['description']}), 
generate 2-3 specific, actionable strategies that DMAI could implement to generate revenue.

For each strategy provide:
1. Strategy name
2. Description and approach
3. Required resources and skills
4. Estimated timeline to revenue
5. Potential revenue range
6. Risk factors
7. Success metrics

Make these concrete and executable strategies."""
                
                result = self.ai_hub.query_all_tutors(prompt)
                if result and result.get('responses'):
                    for tutor, response in result.get('responses', {}).items():
                        if response and isinstance(response, str) and len(response) > 200:
                            # Parse multiple strategies from response
                            strategies = self._parse_strategies(response, avenue_name)
                            self.strategy_candidates[avenue_name].extend(strategies)
                            logger.info(f"      ✅ Generated {len(strategies)} strategy candidates")
                            return
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
        
        # Fallback: create template strategies
        for i in range(2):
            strategy = {
                'id': f"{avenue_name}_strategy_{i+1}_{datetime.now().strftime('%Y%m%d')}",
                'avenue': avenue_name,
                'name': f"{avenue['name']} Strategy {i+1}",
                'description': f"""Strategy for {avenue['name']} based on learned concepts:
{', '.join(avenue['topics'][:5])}

This is a PROPOSED strategy for review only. NOT ACTIVE.""",
                'status': 'proposed',
                'requires_master_review': True,
                'created_at': datetime.now().isoformat(),
                'concepts_used': avenue['topics'][:5]
            }
            self.strategy_candidates[avenue_name].append(strategy)
        
        logger.info(f"      ✅ Generated {len(self.strategy_candidates[avenue_name])} strategy templates")

    def get_unique_topics(self) -> set:
        """Return set of unique topics across all avenues"""
        unique = set()
        for avenue, data in self.revenue_avenues.items():
            for topic in data['topics']:
                unique.add(topic)
        return unique
    
    def get_missing_concepts(self) -> list:
        """Return list of concepts that haven't been learned yet"""
        unique_topics = self.get_unique_topics()
        learned = self.learned_concepts
        missing = list(unique_topics - learned)
        return missing

    def fix_duplicate_topic_counting(self) -> Dict:
        """
        Properly fix the duplicate topic counting issue.
        Uses unique topics for total count instead of counting duplicates.
        """
        unique_topics = self.get_unique_topics()
        total_unique = len(unique_topics)
        learned_unique = len(self.learned_concepts)
        
        logger.info(f"🔧 Fixing duplicate topic counting...")
        logger.info(f"   Unique topics total: {total_unique}")
        logger.info(f"   Learned unique: {learned_unique}")
        
        missing = self.get_missing_concepts()
        if missing:
            logger.info(f"   Missing concepts: {missing}")
            return {
                'success': False,
                'message': f'Still missing {len(missing)} concepts: {missing}',
                'missing_concepts': missing,
                'total_unique_topics': total_unique,
                'learned_unique': learned_unique
            }
        
        # All unique topics are learned - mark Phase 1 complete
        self._training_complete = True
        self.learning_active = False
        
        # Mark all avenues as completed
        for avenue in self.revenue_avenues.values():
            avenue['completed'] = True
        
        # Generate strategies for any avenue that doesn't have them
        strategies_generated = 0
        for avenue_name in self.revenue_avenues.keys():
            if len(self.strategy_candidates.get(avenue_name, [])) == 0:
                self._generate_strategy_candidates(avenue_name)
                strategies_generated += len(self.strategy_candidates.get(avenue_name, []))
                logger.info(f"   Generated {len(self.strategy_candidates[avenue_name])} strategies for {avenue_name}")
        
        self.save_state()
        
        return {
            'success': True,
            'message': 'Phase 1 completed using unique topic counting',
            'total_unique_topics': total_unique,
            'learned_unique': learned_unique,
            'strategies_generated': strategies_generated,
            'strategy_counts': {k: len(v) for k, v in self.strategy_candidates.items()},
            'ready_for_phase_2': self._ready_for_phase_2()
        }
    
    def fix_duplicate_topic_counting(self) -> Dict:
        """
        Properly fix the duplicate topic counting issue.
        Uses unique topics for total count instead of counting duplicates.
        """
        unique_topics = self.get_unique_topics()
        total_unique = len(unique_topics)
        learned_unique = len(self.learned_concepts)
        
        logger.info(f"🔧 Fixing duplicate topic counting...")
        logger.info(f"   Unique topics total: {total_unique}")
        logger.info(f"   Learned unique: {learned_unique}")
        
        missing = self.get_missing_concepts()
        if missing:
            logger.info(f"   Missing concepts: {missing}")
            return {
                'success': False,
                'message': f'Still missing {len(missing)} concepts: {missing}',
                'missing_concepts': missing,
                'total_unique_topics': total_unique,
                'learned_unique': learned_unique
            }
        
        # All unique topics are learned - mark Phase 1 complete
        self._training_complete = True
        self.learning_active = False
        
        # Mark all avenues as completed
        for avenue in self.revenue_avenues.values():
            avenue['completed'] = True
        
        # Generate strategies for any avenue that doesn't have them
        strategies_generated = 0
        for avenue_name in self.revenue_avenues.keys():
            if len(self.strategy_candidates.get(avenue_name, [])) == 0:
                self._generate_strategy_candidates(avenue_name)
                strategies_generated += len(self.strategy_candidates.get(avenue_name, []))
                logger.info(f"   Generated {len(self.strategy_candidates[avenue_name])} strategies for {avenue_name}")
        
        self.save_state()
        
        return {
            'success': True,
            'message': 'Phase 1 completed using unique topic counting',
            'total_unique_topics': total_unique,
            'learned_unique': learned_unique,
            'strategies_generated': strategies_generated,
            'strategy_counts': {k: len(v) for k, v in self.strategy_candidates.items()},
            'ready_for_phase_2': self._ready_for_phase_2()
        }

    def fix_concept_counting(self) -> Dict:
        """
        Fix the concept counting by using unique topics instead of counting duplicates
        """
        # Calculate unique topics
        unique_topics = set()
        for avenue, data in self.revenue_avenues.items():
            for topic in data['topics']:
                unique_topics.add(topic)
        
        total_unique = len(unique_topics)
        learned_unique = len(self.learned_concepts)
        
        logger.info(f"🔧 Fixing concept counting: {learned_unique}/{total_unique} unique concepts")
        
        # If learned_unique >= total_unique, mark as ready
        if learned_unique >= total_unique:
            self._training_complete = True
            logger.info("✅ Phase 1 marked complete with unique concept counting")
            
            # Generate strategies if needed
            for avenue_name in self.revenue_avenues.keys():
                if len(self.strategy_candidates.get(avenue_name, [])) == 0:
                    self._generate_strategy_candidates(avenue_name)
            
            self.save_state()
            return {
                'success': True,
                'message': 'Phase 1 completed using unique concept counting',
                'unique_concepts_total': total_unique,
                'unique_concepts_learned': learned_unique,
                'strategies_generated': sum(len(c) for c in self.strategy_candidates.values())
            }
        else:
            return {
                'success': False,
                'message': f'Still missing {total_unique - learned_unique} concepts',
                'unique_concepts_total': total_unique,
                'unique_concepts_learned': learned_unique,
                'missing_concepts': list(unique_topics - self.learned_concepts)
            }
    
    def _parse_strategies(self, response: str, avenue_name: str) -> List[Dict]:
        """Parse AI response into structured strategy objects"""
        # Simple parsing - in production this would be more sophisticated
        strategies = []
        
        # Create one strategy from the response
        strategy = {
            'id': f"{avenue_name}_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'avenue': avenue_name,
            'name': f"AI-Generated {self.revenue_avenues[avenue_name]['name']} Strategy",
            'description': response[:1500],
            'status': 'proposed',
            'requires_master_review': True,
            'created_at': datetime.now().isoformat(),
            'source': 'ai_tutor_synthesis'
        }
        strategies.append(strategy)
        
        return strategies
    
    def get_strategy_candidates(self, avenue: str = None) -> Dict:
        """Get strategy candidates for review"""
        if avenue:
            return {avenue: self.strategy_candidates.get(avenue, [])}
        return self.strategy_candidates
    
    def request_phase_2_approval(self) -> Dict:
        """
        Request master approval to move to Phase 2 (Paper Execution)
        """
        if not self._ready_for_phase_2():
            incomplete_avenues = [
                name for name, data in self.revenue_avenues.items()
                if not data['completed']
            ]
            avenues_without_strategies = [
                name for name, candidates in self.strategy_candidates.items()
                if len(candidates) == 0
            ]
            
            return {
                'success': False,
                'error': 'Not ready for Phase 2',
                'requirements_remaining': {
                    'incomplete_avenues': incomplete_avenues,
                    'avenues_without_strategies': avenues_without_strategies,
                    'total_concepts_remaining': sum(
                        len(d['topics']) - sum(1 for t in d['topics'] if t in self.learned_concepts)
                        for d in self.revenue_avenues.values()
                    )
                }
            }
        
        return {
            'success': True,
            'message': 'Ready for Phase 2: Paper Execution',
            'requires_master_approval': True,
            'phase_2_description': """
            Phase 2 will execute strategies with PAPER accounts only.
            - No real money involved
            - Real market data, simulated executions
            - Performance tracking
            - Master can review all activity
            """,
            'strategy_candidates': self.strategy_candidates,
            'knowledge_summary': {
                'avenues_completed': len([a for a in self.revenue_avenues.values() if a['completed']]),
                'total_avenues': len(self.revenue_avenues),
                'concepts_learned': len(self.learned_concepts),
                'strategies_developed': sum(len(c) for c in self.strategy_candidates.values())
            }
        }
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning summary"""
        return {
            'phase': '1 - Knowledge Acquisition',
            'status': 'active' if self.learning_active else 'paused',
            'revenue_avenues': {
                name: {
                    'name': data['name'],
                    'progress': data['progress'],
                    'completed': data['completed'],
                    'topics_learned': sum(1 for t in data['topics'] if t in self.learned_concepts),
                    'topics_total': len(data['topics'])
                }
                for name, data in self.revenue_avenues.items()
            },
            'overall_progress': self.get_progress(),
            'strategy_candidates': {
                avenue: len(candidates)
                for avenue, candidates in self.strategy_candidates.items()
            },
            'ready_for_phase_2': self._ready_for_phase_2()
        }
    
    def get_avenue_requirements(self) -> Dict:
        """Get requirements for each revenue avenue"""
        return {
            'quant_trading': {
                'requires': ['exchange_api_keys', 'market_data', 'risk_management'],
                'phase_2_requirements': ['paper_trading_api', 'strategy_validation']
            },
            'content_creation': {
                'requires': ['platform_api_keys', 'content_generation'],
                'phase_2_requirements': ['social_media_api', 'analytics_tracking']
            },
            'ai_services': {
                'requires': ['payment_processor', 'api_infrastructure'],
                'phase_2_requirements': ['stripe_test_keys', 'api_gateway']
            },
            'software_products': {
                'requires': ['development_resources', 'hosting_infrastructure'],
                'phase_2_requirements': ['deployment_pipeline', 'monitoring']
            },
            'affiliate_referral': {
                'requires': ['affiliate_networks', 'tracking_system'],
                'phase_2_requirements': ['affiliate_api_keys', 'commission_tracking']
            },
            'data_services': {
                'requires': ['data_sources', 'processing_infrastructure'],
                'phase_2_requirements': ['data_api_keys', 'analytics_pipeline']
            },
            'education_training': {
                'requires': ['content_platform', 'delivery_system'],
                'phase_2_requirements': ['lms_integration', 'payment_processing']
            },
            'consulting_analysis': {
                'requires': ['expertise', 'communication_tools'],
                'phase_2_requirements': ['client_management', 'deliverable_templates']
            },
            'ad_revenue': {
                'requires': ['ad_network_accounts', 'traffic'],
                'phase_2_requirements': ['ad_network_api_keys', 'analytics']
            },
            'crowdfunding_patronage': {
                'requires': ['platform_accounts', 'community'],
                'phase_2_requirements': ['platform_api_keys', 'community_tools']
            }
        }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class FundingOrchestrator:
    """
    Orchestrates self-funding knowledge acquisition.
    PHASE 1 ONLY - No execution of any revenue generation.
    """
    
    def __init__(self, data_path: Path, financial_manager, knowledge_graph, ai_hub):
        self.data_path = data_path
        self.training = SelfFundingTraining(data_path, financial_manager, knowledge_graph, ai_hub)
    
    def start(self) -> Dict:
        """Standardized start method - starts or resumes learning"""
        return self.training.start_learning()
    
    def update(self) -> None:
        """Standardized update method - called during evolution cycles"""
        self.training.update()
    
    def start_learning(self, avenue: str = None) -> Dict:
        """Start knowledge acquisition (Phase 1)"""
        return self.training.start_learning(avenue)
    
    def stop_learning(self) -> Dict:
        """Stop knowledge acquisition"""
        return self.training.stop_learning()
    
    def status(self) -> Dict:
        """Get learning status"""
        return self.training.get_status()
    
    def get_progress(self) -> float:
        """Get progress percentage"""
        return self.training.get_progress()
    
    def crash_recovery(self) -> Dict:
        """Called after system crash to recover learning"""
        return self.training.crash_recovery()
    
    def get_strategy_candidates(self, avenue: str = None) -> Dict:
        """Get strategy candidates for master review"""
        return self.training.get_strategy_candidates(avenue)
    
    def request_phase_2_approval(self) -> Dict:
        """Request approval to move to Phase 2 (Paper Execution)"""
        return self.training.request_phase_2_approval()
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive learning summary"""
        return self.training.get_learning_summary()
    
    def get_avenue_requirements(self) -> Dict:
        """Get requirements for each revenue avenue"""
        return self.training.get_avenue_requirements()
    
    # ========================================================================
    # CONTENT CREATION - Maintained as it's separate from revenue generation
    # ========================================================================
    def create_content(self, platform: str, content_type: str, topic: str) -> Dict:
        """
        Create content using AI (requires platform API keys)
        This is separate from revenue generation - content can be created without monetization
        """
        available_keys = {
            'twitter_api_key': os.getenv('TWITTER_API_KEY'),
            'twitter_api_secret': os.getenv('TWITTER_API_SECRET'),
            'youtube_api_key': os.getenv('YOUTUBE_API_KEY'),
            'linkedin_client_id': os.getenv('LINKEDIN_CLIENT_ID'),
            'linkedin_client_secret': os.getenv('LINKEDIN_CLIENT_SECRET')
        }
        
        required_keys = {
            'twitter': ['twitter_api_key', 'twitter_api_secret'],
            'youtube': ['youtube_api_key'],
            'linkedin': ['linkedin_client_id', 'linkedin_client_secret']
        }
        
        required = required_keys.get(platform, [])
        missing = [k for k in required if not available_keys.get(k)]
        
        if missing:
            return {
                'success': False,
                'error': f'Missing API keys for {platform}: {missing}',
                'note': 'Content creation requires API keys for posting'
            }
        
        # Generate content using AI
        content = self.training._learn_concept(topic, f"{platform}_content", platform)
        
        return {
            'platform': platform,
            'content_type': content_type,
            'topic': topic,
            'content': content[:500],
            'ready_to_post': True,
            'requires_api_keys': True,
            'note': 'Content ready. DMAI does not auto-post without approval.'
        }
