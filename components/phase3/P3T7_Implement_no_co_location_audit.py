"""
P3T7: No Co-location Audit - FULLY FUNCTIONAL
Ensures DMAI fragments are distributed across different cloud providers
Prevents single point of failure
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class NoCoLocationAudit:
    """
    Audits fragment distribution to ensure no co-location
    Verifies fragments are on different providers and regions
    """
    
    def __init__(self):
        self.name = "No Co-location Audit"
        self.version = "2.0.0"
        self.audit_history = []
        self._initialize()
        
    def _initialize(self):
        """Load existing audits and connect to managers"""
        self._load_data()
        
        # Connect to fragment deployer
        try:
            from components.phase3.P3T6_Deploy_fragment_spawning import get_instance as get_fragments
            self.fragment_manager = get_fragments()
        except:
            self.fragment_manager = None
    
    def _load_data(self):
        """Load existing audits"""
        audit_file = Path("data/co_location_audits.json")
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    self.audit_history = json.load(f)
            except:
                pass
    
    def _save_data(self):
        """Save audits"""
        audit_file = Path("data/co_location_audits.json")
        audit_file.parent.mkdir(exist_ok=True)
        with open(audit_file, 'w') as f:
            json.dump(self.audit_history, f, indent=2)
    
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize and return status"""
        return {
            "status": "active",
            "audits_performed": len(self.audit_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def evolve(self, feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evolve based on audit findings"""
        if feedback and feedback.get("violations_found"):
            self.version = f"2.{len(self.audit_history)}.0"
        return {"version": self.version, "evolved": True}
    
    def execute(self, action: str, params: Dict[str, Any] = None) -> Any:
        """Execute audit actions"""
        actions = {
            "run_audit": self._run_audit,
            "get_audit_history": self._get_audit_history,
            "check_compliance": self._check_compliance,
            "get_violations": self._get_violations,
            "recommend_fix": self._recommend_fix,
            "auto_remediate": self._auto_remediate
        }
        
        if action in actions:
            return actions[action](params or {})
        raise ValueError(f"Unknown action: {action}")
    
    def process(self, data: Any) -> Any:
        """Process commands"""
        if isinstance(data, dict):
            cmd = data.get("command")
            if cmd == "audit":
                return self._run_audit()
            elif cmd == "fix":
                return self._auto_remediate(data.get("violations", {}))
        return {"error": "Unknown command"}
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate audit reports"""
        if "audit" in prompt.lower():
            return "Run audit: execute('run_audit') to check fragment distribution"
        return "No Co-location Audit ready."
    
    def query(self, question: str) -> str:
        """Answer queries"""
        q = question.lower()
        if "compliance" in q:
            return self._check_compliance().get("message", "Checking compliance...")
        elif "violations" in q:
            violations = self._get_violations()
            return f"{len(violations)} co-location violations found"
        return "No Co-location Audit operational."
    
    def _run_audit(self, params: Dict = None) -> Dict:
        """Run full co-location audit"""
        if not self.fragment_manager:
            return {"error": "Fragment manager not available"}
        
        # Get all fragments
        fragments = self.fragment_manager.execute("list_fragments", {}).get("fragments", [])
        
        if not fragments:
            return {
                "success": True,
                "status": "no_fragments",
                "message": "No fragments deployed yet",
                "timestamp": datetime.now().isoformat()
            }
        
        # Analyze distribution
        provider_distribution = {}
        region_distribution = {}
        violations = []
        
        for f in fragments:
            provider = f.get("provider", "unknown")
            provider_distribution[provider] = provider_distribution.get(provider, 0) + 1
            
            region = f.get("region", "default")
            region_key = f"{provider}_{region}"
            region_distribution[region_key] = region_distribution.get(region_key, 0) + 1
            
            # Check for co-location (more than 3 fragments on same provider)
            if provider_distribution[provider] > 3:
                violations.append({
                    "type": "provider_concentration",
                    "provider": provider,
                    "count": provider_distribution[provider],
                    "fragments": [f["id"] for f in fragments if f["provider"] == provider][:5]
                })
        
        # Check for single provider dependency
        if len(provider_distribution) == 1:
            violations.append({
                "type": "single_provider",
                "provider": list(provider_distribution.keys())[0],
                "message": "All fragments on single provider - single point of failure"
            })
        
        audit_result = {
            "audit_id": f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "total_fragments": len(fragments),
            "provider_distribution": provider_distribution,
            "violations_found": len(violations) > 0,
            "violations": violations,
            "compliance_score": self._calculate_compliance_score(provider_distribution, len(fragments)),
            "recommendations": self._generate_recommendations(violations, provider_distribution)
        }
        
        self.audit_history.append(audit_result)
        self._save_data()
        
        return {
            "success": True,
            "audit": audit_result,
            "message": f"Audit complete: {len(violations)} violations found"
        }
    
    def _check_compliance(self, params: Dict = None) -> Dict:
        """Check if system is compliant with co-location rules"""
        if not self.fragment_manager:
            return {"error": "Fragment manager not available"}
        
        # Run latest audit or get status
        fragments = self.fragment_manager.execute("list_fragments", {}).get("fragments", [])
        
        if not fragments:
            return {
                "compliant": True,
                "message": "No fragments - system not deployed",
                "score": 100
            }
        
        provider_distribution = {}
        for f in fragments:
            provider = f.get("provider", "unknown")
            provider_distribution[provider] = provider_distribution.get(provider, 0) + 1
        
        score = self._calculate_compliance_score(provider_distribution, len(fragments))
        compliant = score >= 80
        
        return {
            "compliant": compliant,
            "score": score,
            "message": "System compliant" if compliant else "Compliance issues detected",
            "distribution": provider_distribution,
            "total_fragments": len(fragments)
        }
    
    def _get_violations(self, params: Dict = None) -> List:
        """Get current violations"""
        if not self.fragment_manager:
            return []
        
        fragments = self.fragment_manager.execute("list_fragments", {}).get("fragments", [])
        
        if not fragments:
            return []
        
        provider_distribution = {}
        violations = []
        
        for f in fragments:
            provider = f.get("provider", "unknown")
            provider_distribution[provider] = provider_distribution.get(provider, 0) + 1
        
        for provider, count in provider_distribution.items():
            if count > 3:
                violations.append({
                    "provider": provider,
                    "count": count,
                    "severity": "high" if count > 5 else "medium",
                    "recommendation": f"Move {count - 2} fragments to other providers"
                })
        
        if len(provider_distribution) == 1:
            violations.append({
                "provider": list(provider_distribution.keys())[0],
                "severity": "critical",
                "recommendation": "Deploy fragments to at least 2 other cloud providers"
            })
        
        return violations
    
    def _recommend_fix(self, params: Dict = None) -> Dict:
        """Recommend fixes for violations"""
        violations = self._get_violations()
        
        if not violations:
            return {"message": "No violations to fix"}
        
        recommendations = []
        for v in violations:
            if v.get("type") == "provider_concentration":
                recommendations.append({
                    "action": "migrate_fragments",
                    "source_provider": v["provider"],
                    "count_to_migrate": v["count"] - 2,
                    "target_providers": self._get_available_providers(v["provider"])
                })
            elif v.get("type") == "single_provider":
                recommendations.append({
                    "action": "deploy_to_new_providers",
                    "current_provider": v["provider"],
                    "recommended_providers": ["aws", "gcp", "azure", "oracle"],
                    "fragments_to_deploy": 3
                })
            else:
                recommendations.append({
                    "action": "general_rebalancing",
                    "recommendation": v.get("recommendation", "Redistribute fragments")
                })
        
        return {
            "violations": violations,
            "recommendations": recommendations,
            "message": f"Generated {len(recommendations)} recommendations"
        }
    
    def _auto_remediate(self, params: Dict = None) -> Dict:
        """Automatically fix co-location violations"""
        if not self.fragment_manager:
            return {"error": "Fragment manager not available"}
        
        violations = self._get_violations()
        
        if not violations:
            return {"message": "No violations to remediate"}
        
        actions = []
        
        for v in violations:
            if v.get("type") == "provider_concentration":
                # Migrate fragments
                provider = v["provider"]
                to_migrate = v["count"] - 2
                
                # Get fragments from this provider
                fragments = self.fragment_manager.execute("list_fragments", {}).get("fragments", [])
                provider_fragments = [f for f in fragments if f.get("provider") == provider]
                
                for f in provider_fragments[:to_migrate]:
                    # Kill and respawn on different provider
                    kill = self.fragment_manager.execute("kill_fragment", {"fragment_id": f["id"]})
                    actions.append({"action": "kill", "fragment": f["id"], "result": kill})
                    
                    # Spawn on different provider
                    new_provider = self._get_alternative_provider(provider)
                    spawn = self.fragment_manager.execute("spawn_fragment", {
                        "type": f.get("type", "core"),
                        "provider": new_provider
                    })
                    actions.append({"action": "spawn", "provider": new_provider, "result": spawn})
            
            elif v.get("type") == "single_provider":
                # Deploy to new providers
                for provider in ["aws", "gcp", "azure", "oracle"]:
                    if provider != v["provider"]:
                        spawn = self.fragment_manager.execute("spawn_fragment", {
                            "type": "core",
                            "provider": provider
                        })
                        actions.append({"action": "spawn_new", "provider": provider, "result": spawn})
        
        # Re-run audit to verify
        audit = self._run_audit()
        
        return {
            "success": True,
            "actions_taken": len(actions),
            "actions": actions,
            "post_audit": audit,
            "message": f"Auto-remediation complete: {len(actions)} actions taken"
        }
    
    def _get_audit_history(self, params: Dict = None) -> Dict:
        """Get audit history"""
        limit = params.get("limit", 10) if params else 10
        return {"audits": self.audit_history[-limit:]}
    
    def _calculate_compliance_score(self, distribution: Dict, total: int) -> int:
        """Calculate compliance score (0-100)"""
        if total == 0:
            return 100
        
        # Penalty for concentration on single provider
        provider_count = len(distribution)
        if provider_count == 1:
            return 30
        elif provider_count == 2:
            base = 60
        elif provider_count == 3:
            base = 80
        else:
            base = 95
        
        # Penalty for heavy concentration
        for count in distribution.values():
            if count > total * 0.6:  # >60% on one provider
                base -= 20
            elif count > total * 0.4:  # >40% on one provider
                base -= 10
        
        return max(0, min(100, base))
    
    def _generate_recommendations(self, violations: List, distribution: Dict) -> List:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if len(distribution) <= 2:
            recommendations.append("Add fragments to at least 2 additional cloud providers")
        
        for v in violations:
            if v.get("type") == "provider_concentration":
                recommendations.append(f"Move {v['count'] - 2} fragments from {v['provider']} to other providers")
        
        return recommendations
    
    def _get_available_providers(self, exclude: str) -> List:
        """Get available providers excluding specified"""
        all_providers = ["aws", "gcp", "azure", "oracle"]
        return [p for p in all_providers if p != exclude]
    
    def _get_alternative_provider(self, current: str) -> str:
        """Get alternative provider"""
        alternatives = self._get_available_providers(current)
        return alternatives[0] if alternatives else "aws"

_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = NoCoLocationAudit()
    return _instance

def run(context=None): return get_instance().run(context)
def evolve(feedback=None): return get_instance().evolve(feedback)
def execute(action, params=None): return get_instance().execute(action, params)
def process(data): return get_instance().process(data)
def generate(prompt, **kwargs): return get_instance().generate(prompt, **kwargs)
def query(question): return get_instance().query(question)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audit = get_instance()
    print(json.dumps(audit.run(), indent=2))
    
    print("\nRunning audit...")
    result = audit.execute("run_audit", {})
    print(json.dumps(result, indent=2))
