#!/bin/bash
# Security management script

cd /Users/davidmiles/Desktop/dmai-system

case "$1" in
    setup)
        echo "🔐 Setting up biometric security..."
        python3 security/biometric_auth.py
        ;;
    status)
        echo "📊 Security Status:"
        python3 -c "
from security.biometric_auth import BiometricAuth
auth = BiometricAuth()
status = auth.get_security_status()
for k, v in status.items():
    print(f'  {k}: {v}')
"
        ;;
    backup-codes)
        echo "🔑 Generating new backup codes..."
        python3 -c "
from security.biometric_auth import BiometricAuth
auth = BiometricAuth()
auth.generate_backup_codes(10)
"
        ;;
    auth)
        echo "🔐 Testing authentication..."
        python3 -c "
from security.biometric_auth import BiometricAuth
auth = BiometricAuth()
auth.authenticate()
"
        ;;
    logs)
        echo "📋 Recent authentication attempts:"
        tail -20 security/audit_logs/auth_attempts.json | python3 -m json.tool
        ;;
    *)
        echo "Usage: $0 {setup|status|backup-codes|auth|logs}"
        echo ""
        echo "  setup       - Initial security setup"
        echo "  status      - Show security status"
        echo "  backup-codes- Generate new backup codes"
        echo "  auth        - Test authentication"
        echo "  logs        - View auth logs"
        ;;
esac
