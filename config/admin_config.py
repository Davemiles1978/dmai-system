import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta

class AdminAuth:
    def __init__(self):
        self.admin_id = "davidmiles1978"  # Your specific ID
        # In production, this should be loaded from environment variables
        self.master_hash = hashlib.sha256("YourSecurePassword123!".encode()).hexdigest()
        self.active_sessions = {}
        
    def verify_admin(self, username, password):
        """Verify admin credentials"""
        if username != self.admin_id:
            return False
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(password_hash, self.master_hash)
    
    def create_session(self):
        """Create a secure session token"""
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        self.active_sessions[token] = datetime.now() + timedelta(hours=1)
        return token
    
    def verify_session(self, token):
        """Verify session token"""
        if token in self.active_sessions:
            if datetime.now() < self.active_sessions[token]:
                return True
            else:
                del self.active_sessions[token]
        return False
    
    def revoke_session(self, token):
        """Logout - revoke session"""
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False
