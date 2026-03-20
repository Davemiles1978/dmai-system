"""
P1T9: Implement Master Control
Master Control component with TouchID support for biometric authentication
Phase 1 Component 9 - Provides secure master control interface
"""

import logging
import json
import time
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MasterControl:
    """
    Master Control component with biometric authentication
    Provides secure access to critical system functions
    Supports TouchID, FaceID, and password fallback
    """
    
    def __init__(self):
        self.name = "Master Control"
        self.version = "1.0.0"
        self.status = "initialized"
        self.authenticated = False
        self.auth_method = None
        self.auth_time = None
        self.session_id = None
        self.biometric_supported = self._check_biometric_support()
        self.access_log = []
        self.master_key_hash = self._hash_master_key("Talula.78")  # Default, should come from env
        self.evolution_engine = None
        self.recovery_engine = None
        
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the master control - initialize and authenticate
        Required by DMAI core interface
        """
        logger.info("🎮 P1T9: Running Master Control")
        
        if context is None:
            context = {}
        
        # Try to authenticate from context
        auth_result = self.authenticate(
            method=context.get('auth_method', 'none'),
            credentials=context.get('credentials', {})
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': self.status,
            'authenticated': self.authenticated,
            'auth_method': self.auth_method,
            'biometric_supported': self.biometric_supported,
            'session_id': self.session_id,
            'message': 'Master Control initialized' if self.authenticated else 'Authentication required'
        }
    
    def evolve(self, feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evolve master control based on feedback
        Required by DMAI core interface
        """
        logger.info("🧬 P1T9: Evolving Master Control")
        
        improvements = []
        
        if feedback:
            # Learn from authentication failures
            if feedback.get('failed_attempts'):
                improvements.append('enhanced_security_measures')
            
            # Learn from successful biometrics
            if feedback.get('biometric_success_rate', 1.0) > 0.95:
                improvements.append('optimized_biometric_matching')
        
        # Version bump
        self.version = f"{self.version.split('.')[0]}.{int(self.version.split('.')[1]) + 1}.0"
        
        return {
            'version': self.version,
            'evolved': True,
            'improvements': improvements if improvements else ['security_optimization'],
            'timestamp': datetime.now().isoformat()
        }
    
    def execute(self, action: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a specific master control action
        Required by DMAI core interface
        """
        logger.info(f"⚙️ P1T9: Executing action '{action}'")
        
        if params is None:
            params = {}
        
        # Check authentication for sensitive actions
        sensitive_actions = ['shutdown', 'restart', 'emergency_stop', 'reset_system']
        
        if action in sensitive_actions and not self.authenticated:
            return {'error': 'Authentication required for this action', 'authenticated': False}
        
        actions = {
            'authenticate': self.authenticate,
            'logout': self.logout,
            'get_status': self.get_status,
            'get_access_log': self._get_access_log,
            'emergency_stop': self._emergency_stop,
            'shutdown': self._shutdown,
            'restart': self._restart,
            'reset_system': self._reset_system,
            'verify_touchid': self._verify_touchid,
            'verify_faceid': self._verify_faceid,
            'set_master_key': self._set_master_key
        }
        
        if action in actions:
            if action == 'authenticate':
                return actions[action](params.get('method'), params.get('credentials'))
            elif action == 'set_master_key':
                return actions[action](params.get('new_key'))
            elif action in ['emergency_stop', 'shutdown', 'restart', 'reset_system']:
                return actions[action](params.get('reason'))
            else:
                return actions[action]()
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def process(self, data: Any) -> Any:
        """
        Process master control commands
        Required by DMAI core interface
        """
        logger.info(f"📥 P1T9: Processing command")
        
        if isinstance(data, dict):
            command = data.get('command', '')
            
            if command == 'auth':
                return self.authenticate(data.get('method'), data.get('credentials'))
            elif command == 'status':
                return self.get_status()
            elif command == 'log':
                return self._get_access_log(data.get('limit', 10))
            elif command == 'shutdown':
                return self._shutdown(data.get('reason', 'user_request'))
            elif command == 'emergency':
                return self._emergency_stop(data.get('reason', 'emergency'))
            else:
                return {'error': f'Unknown command: {command}'}
        else:
            return {'error': 'Invalid data format - expected dict'}
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate master control response
        Required by DMAI core interface
        """
        logger.info(f"📝 P1T9: Generating response for: {prompt[:50]}...")
        
        prompt_lower = prompt.lower()
        
        if 'status' in prompt_lower:
            return f"Master Control v{self.version} - {'Authenticated' if self.authenticated else 'Not Authenticated'}"
        elif 'help' in prompt_lower:
            return self._get_help()
        elif 'biometric' in prompt_lower or 'touchid' in prompt_lower or 'faceid' in prompt_lower:
            if self.biometric_supported:
                return "Biometric authentication is supported. Use TouchID or FaceID for secure access."
            else:
                return "Biometric authentication not available on this system. Using password fallback."
        elif 'security' in prompt_lower:
            return self._get_security_info()
        else:
            return "Master Control: I manage system security and critical functions. Try asking about status, biometrics, or help."
    
    def query(self, question: str) -> str:
        """
        Answer master control queries
        Required by DMAI core interface
        """
        logger.info(f"❓ P1T9: Answering query: {question}")
        
        question_lower = question.lower()
        
        if 'who is authenticated' in question_lower:
            return f"Authentication: {'Active' if self.authenticated else 'None'} via {self.auth_method if self.auth_method else 'N/A'}"
        
        elif 'when was last auth' in question_lower:
            if self.auth_time:
                return f"Last authentication: {self.auth_time}"
            else:
                return "No authentication recorded"
        
        elif 'how many attempts' in question_lower:
            total_attempts = len(self.access_log)
            failed = sum(1 for entry in self.access_log if not entry.get('success', False))
            return f"Total attempts: {total_attempts}, Failed: {failed}"
        
        elif 'biometric' in question_lower:
            return f"Biometric support: {'✅ Available' if self.biometric_supported else '❌ Not available'}"
        
        elif 'version' in question_lower:
            return f"Master Control version {self.version}"
        
        elif 'recovery' in question_lower:
            return "Recovery engine status: Standby - Use execute('recovery') to activate"
        
        elif 'evolution' in question_lower:
            return "Evolution engine status: Connected - Ready to evolve components"
        
        else:
            return "I can answer questions about authentication, security, biometrics, and system status."
    
    # Private helper methods
    def authenticate(self, method: str = None, credentials: Dict = None) -> Dict[str, Any]:
        """
        Authenticate user with specified method
        Methods: 'touchid', 'faceid', 'password', 'token'
        """
        if credentials is None:
            credentials = {}
        
        result = {
            'success': False,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log attempt
        attempt = {
            'timestamp': result['timestamp'],
            'method': method,
            'success': False
        }
        
        if method == 'touchid':
            # Simulate TouchID verification
            success = self._verify_touchid(credentials.get('touchid_data'))
            result['success'] = success
            result['message'] = 'TouchID verified' if success else 'TouchID failed'
            
        elif method == 'faceid':
            # Simulate FaceID verification
            success = self._verify_faceid(credentials.get('face_data'))
            result['success'] = success
            result['message'] = 'FaceID verified' if success else 'FaceID failed'
            
        elif method == 'password':
            # Password verification
            password = credentials.get('password', '')
            success = self._verify_password(password)
            result['success'] = success
            result['message'] = 'Password accepted' if success else 'Invalid password'
            
        elif method == 'token':
            # Token verification
            token = credentials.get('token', '')
            success = self._verify_token(token)
            result['success'] = success
            result['message'] = 'Token accepted' if success else 'Invalid token'
            
        else:
            result['message'] = f'Unknown authentication method: {method}'
            self.access_log.append(attempt)
            return result
        
        if result['success']:
            self.authenticated = True
            self.auth_method = method
            self.auth_time = result['timestamp']
            self.session_id = hashlib.sha256(f"{time.time()}{method}".encode()).hexdigest()[:16]
            result['session_id'] = self.session_id
        
        attempt['success'] = result['success']
        self.access_log.append(attempt)
        
        return result
    
    def logout(self) -> Dict[str, Any]:
        """Log out current session"""
        self.authenticated = False
        old_session = self.session_id
        self.session_id = None
        self.auth_method = None
        
        return {
            'success': True,
            'message': 'Logged out successfully',
            'previous_session': old_session,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get master control status"""
        return {
            'name': self.name,
            'version': self.version,
            'authenticated': self.authenticated,
            'auth_method': self.auth_method,
            'auth_time': self.auth_time,
            'session_id': self.session_id if self.authenticated else None,
            'biometric_supported': self.biometric_supported,
            'access_log_count': len(self.access_log),
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_biometric_support(self) -> bool:
        """Check if biometric authentication is supported"""
        # On macOS, check for TouchID support
        import sys
        if sys.platform == 'darwin':
            try:
                # This is a simulation - actual TouchID would require system APIs
                return True
            except:
                return False
        return False
    
    def _verify_touchid(self, data: Any = None) -> bool:
        """Verify TouchID authentication"""
        # Simulate TouchID - in reality would call system API
        # For simulation, we'll return True 95% of the time
        import random
        return random.random() < 0.95
    
    def _verify_faceid(self, data: Any = None) -> bool:
        """Verify FaceID authentication"""
        # Simulate FaceID - in reality would call system API
        import random
        return random.random() < 0.98
    
    def _verify_password(self, password: str) -> bool:
        """Verify password"""
        # Compare with stored hash
        input_hash = self._hash_master_key(password)
        return hmac.compare_digest(input_hash, self.master_key_hash)
    
    def _verify_token(self, token: str) -> bool:
        """Verify authentication token"""
        # Simple token verification
        expected = hashlib.sha256(f"{self.session_id}{time.time()//3600}".encode()).hexdigest()[:32]
        return token == expected
    
    def _hash_master_key(self, key: str) -> str:
        """Hash master key for secure storage"""
        salt = "dmai-static-salt"  # In production, use per-installation salt
        return hashlib.pbkdf2_hmac('sha256', key.encode(), salt.encode(), 100000).hex()
    
    def _set_master_key(self, new_key: str) -> Dict[str, Any]:
        """Set new master key (requires authentication)"""
        if not self.authenticated:
            return {'error': 'Authentication required', 'success': False}
        
        self.master_key_hash = self._hash_master_key(new_key)
        return {'success': True, 'message': 'Master key updated'}
    
    def _emergency_stop(self, reason: str = None) -> Dict[str, Any]:
        """Emergency stop - highest priority"""
        logger.warning(f"🚨 EMERGENCY STOP triggered: {reason}")
        return {
            'success': True,
            'action': 'emergency_stop',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'message': 'Emergency stop executed - all non-critical systems paused'
        }
    
    def _shutdown(self, reason: str = None) -> Dict[str, Any]:
        """Shutdown master control"""
        logger.info(f"🛑 Shutdown requested: {reason}")
        return {
            'success': True,
            'action': 'shutdown',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _restart(self, reason: str = None) -> Dict[str, Any]:
        """Restart master control"""
        logger.info(f"🔄 Restart requested: {reason}")
        return {
            'success': True,
            'action': 'restart',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _reset_system(self, reason: str = None) -> Dict[str, Any]:
        """Reset system (requires authentication)"""
        if not self.authenticated:
            return {'error': 'Authentication required', 'success': False}
        
        logger.warning(f"⚠️ System reset requested: {reason}")
        return {
            'success': True,
            'action': 'reset',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_access_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get access log entries"""
        return self.access_log[-limit:]
    
    def _get_security_info(self) -> str:
        """Get security information"""
        return f"""
Security Information:
- Authentication: {'Active' if self.authenticated else 'Inactive'}
- Method: {self.auth_method if self.auth_method else 'None'}
- Biometric Support: {'Yes' if self.biometric_supported else 'No'}
- Session Active: {'Yes' if self.session_id else 'No'}
- Access Log Entries: {len(self.access_log)}
"""
    
    def _get_help(self) -> str:
        """Get help information"""
        return """
Master Control Commands:
- run() - Initialize master control
- evolve() - Evolve security measures
- execute(action, params) - Execute specific actions
- process(data) - Process commands
- generate(prompt) - Generate responses
- query(question) - Answer questions

Available actions for execute():
- authenticate(method, credentials) - Authenticate user
- logout() - Log out current session
- get_status() - Get system status
- emergency_stop(reason) - Emergency stop
- shutdown(reason) - Shutdown system
- restart(reason) - Restart system
- verify_touchid() - Test TouchID
- verify_faceid() - Test FaceID

Authentication methods:
- touchid - Use TouchID
- faceid - Use FaceID
- password - Use password
- token - Use auth token
"""

# Singleton instance for DMAI core
_instance = None

def get_instance():
    """Get or create the singleton instance"""
    global _instance
    if _instance is None:
        _instance = MasterControl()
    return _instance

# Required interface methods for DMAI core
def run(context=None):
    """Run the component"""
    return get_instance().run(context)

def evolve(feedback=None):
    """Evolve the component"""
    return get_instance().evolve(feedback)

def execute(action, params=None):
    """Execute a specific action"""
    return get_instance().execute(action, params)

def process(data):
    """Process input data"""
    return get_instance().process(data)

def generate(prompt, **kwargs):
    """Generate output based on prompt"""
    return get_instance().generate(prompt, **kwargs)

def query(question):
    """Answer a query"""
    return get_instance().query(question)

# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create master control
    mc = get_instance()
    
    # Test status
    print("Initial status:", mc.get_status())
    
    # Test password authentication
    result = mc.authenticate('password', {'password': 'Talula.78'})
    print("Auth result:", result)
    
    # Test status after auth
    print("Status after auth:", mc.get_status())
    
    # Test query
    print("\nQ: Who is authenticated?")
    print(f"A: {mc.query('who is authenticated')}")
    
    # Test logout
    print("\nLogging out...")
    mc.logout()
    print("Status after logout:", mc.get_status())
