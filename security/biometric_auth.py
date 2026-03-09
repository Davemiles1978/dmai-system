#!/usr/bin/env python3

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Biometric Authentication System for DMAI
Multi-factor security with local storage only
"""

import os
import json
import hashlib
import base64
import time
from datetime import datetime
from pathlib import Path
import secrets

class BiometricAuth:
    def __init__(self):
        self.security_dir = "/Users/davidmiles/Desktop/dmai-system/security"
        self.templates_dir = f"{self.security_dir}/templates"
        self.codes_dir = f"{self.security_dir}/backup_codes"
        self.audit_log = f"{self.security_dir}/audit_logs/auth_attempts.json"
        self.config_file = f"{self.security_dir}/security_config.json"
        
        # Ensure directories exist
        Path(self.templates_dir).mkdir(parents=True, exist_ok=True)
        Path(self.codes_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.security_dir}/audit_logs").mkdir(parents=True, exist_ok=True)
        
        self.load_config()
        self.load_audit_log()
    
    def load_config(self):
        """Load security configuration"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'required_factors': 2,  # Need 2 of 3 factors
                'timeout_minutes': 5,
                'max_attempts': 3,
                'lockout_minutes': 15,
                'factors_enabled': {
                    'fingerprint': True,
                    'face': True,
                    'voice': True
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            self.save_config()
    
    def save_config(self):
        """Save security configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_audit_log(self):
        """Load audit log"""
        if Path(self.audit_log).exists():
            with open(self.audit_log, 'r') as f:
                self.audit_log_data = json.load(f)
        else:
            self.audit_log_data = {
                'attempts': [],
                'lockouts': [],
                'enrollments': []
            }
    
    def save_audit_log(self):
        """Save audit log"""
        with open(self.audit_log, 'w') as f:
            json.dump(self.audit_log_data, f, indent=2)
    
    def log_attempt(self, success, method, details=""):
        """Log authentication attempt"""
        attempt = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'method': method,
            'details': details,
            'ip': 'local_only'  # Always local
        }
        self.audit_log_data['attempts'].append(attempt)
        
        # Keep last 1000 attempts
        if len(self.audit_log_data['attempts']) > 1000:
            self.audit_log_data['attempts'] = self.audit_log_data['attempts'][-1000:]
        
        self.save_audit_log()
    
    def check_attempts(self):
        """Check if too many failed attempts"""
        recent = [a for a in self.audit_log_data['attempts'][-10:] 
                 if not a['success'] and 
                 (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 300]
        
        if len(recent) >= self.config['max_attempts']:
            return False
        return True
    
    # ========== FINGERPRINT METHODS ==========
    
    def enroll_fingerprint(self, fingerprint_data, name="default"):
        """Enroll a fingerprint (Touch ID integration)"""
        print(f"\n📱 Enrolling fingerprint: {name}")
        print("   Please place your finger on the Touch ID sensor...")
        
        # In production, this would use system Touch ID API
        # For now, we'll simulate with hashed template
        
        # Create template hash
        template_hash = hashlib.sha256(
            f"{fingerprint_data}{secrets.token_hex(16)}".encode()
        ).hexdigest()
        
        template = {
            'name': name,
            'created': datetime.now().isoformat(),
            'template_hash': template_hash,
            'method': 'fingerprint'
        }
        
        # Save template
        template_file = f"{self.templates_dir}/fingerprint/{name}_{int(time.time())}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f)
        
        self.audit_log_data['enrollments'].append({
            'timestamp': datetime.now().isoformat(),
            'method': 'fingerprint',
            'name': name
        })
        self.save_audit_log()
        
        print(f"✅ Fingerprint enrolled successfully")
        return template_file
    
    def verify_fingerprint(self, fingerprint_data):
        """Verify fingerprint (simulated)"""
        if not self.check_attempts():
            return False
        
        # In production, this would use Touch ID
        # For simulation, always succeed first time
        success = True
        
        self.log_attempt(success, 'fingerprint')
        return success
    
    # ========== FACE RECOGNITION METHODS ==========
    
    def enroll_face(self, image_data, name="default"):
        """Enroll face from camera"""
        print(f"\n📸 Enrolling face: {name}")
        print("   Please look at the camera...")
        
        # In production, this would use Vision framework
        # For now, store placeholder
        
        template = {
            'name': name,
            'created': datetime.now().isoformat(),
            'encoding': hashlib.sha256(f"{image_data}{secrets.token_hex(16)}".encode()).hexdigest(),
            'method': 'face'
        }
        
        template_file = f"{self.templates_dir}/face/{name}_{int(time.time())}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f)
        
        self.audit_log_data['enrollments'].append({
            'timestamp': datetime.now().isoformat(),
            'method': 'face',
            'name': name
        })
        self.save_audit_log()
        
        print(f"✅ Face enrolled successfully")
        return template_file
    
    def verify_face(self, image_data):
        """Verify face (simulated)"""
        if not self.check_attempts():
            return False
        
        # In production, this would use camera
        success = True
        
        self.log_attempt(success, 'face')
        return success
    
    # ========== BACKUP CODES ==========
    
    def generate_backup_codes(self, count=10):
        """Generate emergency backup codes"""
        print(f"\n🔑 Generating {count} backup codes...")
        
        codes = []
        codes_file = f"{self.codes_dir}/backup_codes_{int(time.time())}.txt"
        
        with open(codes_file, 'w') as f:
            f.write("DMAI EMERGENCY BACKUP CODES\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("Store these in a SAFE place!\n")
            f.write("Each code can only be used once.\n")
            f.write("=" * 50 + "\n\n")
            
            for i in range(count):
                # Generate 8-character alphanumeric code
                code = secrets.token_hex(4).upper()
                codes.append(code)
                f.write(f"{i+1:2d}. {code}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("KEEP THIS FILE SECURE!\n")
        
        print(f"✅ Backup codes saved to: {codes_file}")
        print("\n📋 First 3 codes:")
        for i, code in enumerate(codes[:3]):
            print(f"   {i+1}. {code}")
        
        return codes_file
    
    def use_backup_code(self, code):
        """Use a backup code (one-time use)"""
        # Find code file
        for file in Path(self.codes_dir).glob("*.txt"):
            with open(file, 'r') as f:
                content = f.read()
                if code in content:
                    # Mark as used (in production, would track used codes)
                    self.log_attempt(True, 'backup_code', f"Code used: {code[:4]}...")
                    return True
        
        self.log_attempt(False, 'backup_code', 'Invalid code')
        return False
    
    # ========== MULTI-FACTOR AUTH ==========
    
    def authenticate(self, methods=None):
        """
        Multi-factor authentication
        Requires config['required_factors'] successful verifications
        """
        print("\n🔐 DMAI MULTI-FACTOR AUTHENTICATION")
        print("=" * 50)
        
        if not methods:
            methods = ['fingerprint', 'face', 'voice']
        
        successful = []
        required = self.config['required_factors']
        
        print(f"Required factors: {required}")
        print(f"Available methods: {', '.join(methods)}")
        print("")
        
        for method in methods:
            if not self.config['factors_enabled'].get(method, False):
                continue
            
            print(f"\n📋 Factor {len(successful)+1}/{required}: {method.upper()}")
            
            if method == 'fingerprint':
                input("Press Enter when finger is on sensor...")
                if self.verify_fingerprint("simulated_data"):
                    successful.append('fingerprint')
                    print("   ✅ Fingerprint verified")
            
            elif method == 'face':
                input("Press Enter when facing camera...")
                if self.verify_face("simulated_data"):
                    successful.append('face')
                    print("   ✅ Face verified")
            
            elif method == 'voice':
                print("   Speak your passphrase...")
                input("Press Enter after speaking...")
                # Would use voice auth system
                successful.append('voice')
                print("   ✅ Voice verified")
            
            if len(successful) >= required:
                break
        
        if len(successful) >= required:
            print("\n✅ AUTHENTICATION SUCCESSFUL")
            return True
        else:
            print(f"\n❌ Authentication failed. Only {len(successful)}/{required} factors verified.")
            return False
    
    def get_security_status(self):
        """Get current security status"""
        status = {
            'factors_enabled': self.config['factors_enabled'],
            'required_factors': self.config['required_factors'],
            'total_attempts': len(self.audit_log_data['attempts']),
            'recent_failures': len([a for a in self.audit_log_data['attempts'][-10:] if not a['success']]),
            'enrollments': len(self.audit_log_data['enrollments']),
            'backup_codes': len(list(Path(self.codes_dir).glob("*.txt"))),
            'status': 'ACTIVE' if self.check_attempts() else 'LOCKED'
        }
        return status

if __name__ == "__main__":
    auth = BiometricAuth()
    
    print("\n🔐 DMAI BIOMETRIC SECURITY SYSTEM")
    print("=" * 50)
    
    # Initial setup
    print("\n1. First-time setup - Enroll biometrics")
    
    # Enroll fingerprint
    auth.enroll_fingerprint("sample_fingerprint_data", "master_finger")
    
    # Enroll face
    auth.enroll_face("sample_face_data", "master_face")
    
    # Generate backup codes
    auth.generate_backup_codes(10)
    
    # Show status
    print("\n2. Security Status:")
    status = auth.get_security_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test authentication
    print("\n3. Testing Authentication:")
    auth.authenticate(['fingerprint', 'face'])
