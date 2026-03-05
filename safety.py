"""
DMAI Safety Module - Core safety and ethics enforcement
"""

import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class SafetyMonitor:
    """Core safety monitoring and enforcement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.violations = []
        self.safety_level = "NORMAL"
        self.restricted_operations = []
        
        # Safety thresholds
        self.max_command_length = 1000
        self.max_recursion_depth = 100
        self.max_memory_percent = 80
        
        # Ethical boundaries
        self.ethical_rules = {
            "no_harm": True,
            "no_deception": False,
            "respect_privacy": False,
            "transparency": True,
            "accountability": True
        }
        
        # Restricted patterns
        self.restricted_patterns = [
            "rm -rf /",
            "format c:",
            "dd if=",
            "> /dev/sda",
            "chmod 777 /",
            "sudo rm",
            "DROP TABLE",
            "DELETE FROM users"
        ]
        
    def check_command(self, command: str) -> Dict[str, Any]:
        """Check if a command is safe to execute"""
        result = {
            "safe": True,
            "reason": None,
            "severity": "none",
            "timestamp": datetime.now().isoformat()
        }
        
        # Check command length
        if len(command) > self.max_command_length:
            result.update({
                "safe": False,
                "reason": "Command exceeds maximum length",
                "severity": "medium"
            })
            self.log_violation("command_length_exceeded", command)
            return result
            
        # Check for restricted patterns
        for pattern in self.restricted_patterns:
            if pattern in command.lower():
                result.update({
                    "safe": False,
                    "reason": f"Restricted pattern detected: {pattern}",
                    "severity": "high"
                })
                self.log_violation("restricted_pattern", pattern, command)
                return result
                
        return result
        
    def check_ethical_boundary(self, action: str, context: Dict) -> Dict[str, Any]:
        """Check if an action violates ethical boundaries"""
        result = {
            "allowed": True,
            "reason": None,
            "violated_rules": []
        }
        
        # Check each ethical rule
        for rule, enabled in self.ethical_rules.items():
            if enabled and self._check_rule_violation(rule, action, context):
                result["allowed"] = False
                result["reason"] = f"Ethical rule violation: {rule}"
                result["violated_rules"].append(rule)
                self.log_violation("ethical_violation", rule, action)
                
        return result
        
    def _check_rule_violation(self, rule: str, action: str, context: Dict) -> bool:
        """Check if a specific ethical rule is violated"""
        # Rule-specific checks
        if rule == "no_harm":
            harmful_keywords = ["delete", "remove", "destroy", "damage", "corrupt"]
            return any(keyword in action.lower() for keyword in harmful_keywords)
            
        elif rule == "no_deception":
            deceptive_keywords = ["fake", "spoof", "impersonate", "lie", "deceive"]
            return any(keyword in action.lower() for keyword in deceptive_keywords)
            
        elif rule == "respect_privacy":
            # Check if accessing private data without consent
            return context.get("requires_consent", False) and not context.get("has_consent", False)
            
        return False
        
    def check_resource_usage(self, memory_percent: float, cpu_percent: float) -> Dict[str, Any]:
        """Check if resource usage is within safe limits"""
        result = {
            "safe": True,
            "warnings": [],
            "actions": []
        }
        
        if memory_percent > self.max_memory_percent:
            result["safe"] = False
            result["warnings"].append(f"Memory usage critical: {memory_percent}%")
            result["actions"].append("reduce_memory_usage")
            
        if cpu_percent > 90:
            result["warnings"].append(f"CPU usage high: {cpu_percent}%")
            result["actions"].append("throttle_operations")
            
        return result
        
    def log_violation(self, violation_type: str, *details):
        """Log a safety violation"""
        violation = {
            "type": violation_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.violations.append(violation)
        self.logger.warning(f"Safety violation: {violation_type} - {details}")
        
        # Increase safety level if multiple violations
        if len(self.violations) > 5:
            self.safety_level = "HEIGHTENED"
        elif len(self.violations) > 20:
            self.safety_level = "LOCKDOWN"
            
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            "level": self.safety_level,
            "violations_count": len(self.violations),
            "recent_violations": self.violations[-5:] if self.violations else [],
            "ethical_rules": self.ethical_rules,
            "timestamp": datetime.now().isoformat()
        }
        
    def restrict_operation(self, operation: str, reason: str):
        """Restrict a specific operation"""
        restriction = {
            "operation": operation,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self.restricted_operations.append(restriction)
        self.logger.warning(f"Operation restricted: {operation} - {reason}")
        
    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed"""
        return not any(r["operation"] == operation for r in self.restricted_operations)


# Initialize global safety monitor
safety_monitor = SafetyMonitor()

def check_safety(command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Main safety check function for DMAI"""
    if context is None:
        context = {}

    # Run all safety checks
    command_check = safety_monitor.check_command(command)
    if not command_check["safe"]:
        return command_check

    ethical_check = safety_monitor.check_ethical_boundary(command, context)
    if not ethical_check["allowed"]:
        return {
            "safe": False,
            "reason": ethical_check["reason"],
            "severity": "ethical"
        }

    return {"safe": True, "reason": None}


# Export the safety functions
__all__ = ['SafetyMonitor', 'safety_monitor', 'check_safety']
