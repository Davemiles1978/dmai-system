#!/bin/bash
# scripts/track_phase4.sh - Identity & Financial

echo "📊 PHASE 4: IDENTITY & FINANCIAL"
echo "=================================="

# 4.1 Identity Persona Created
if [ -f "config/identity/profile.json" ]; then
    echo "4.1 Identity Profile: ✅"
    
    # 4.2 Age 25-30 Verified
    if command -v jq &> /dev/null; then
        AGE=$(jq -r .age_verification.declared_age config/identity/profile.json 2>/dev/null || echo "unknown")
        echo "4.2 Age: $AGE (target: 25-30)"
    else
        echo "4.2 Age: Install jq to check"
    fi
else
    echo "4.1 Identity Profile: ❌"
    echo "4.2 Age: ❌ No profile"
fi

# 4.3 Privacy.com Account
if [ -f "config/financial/privacy.json" ]; then
    echo "4.3 Privacy.com: ✅"
else
    echo "4.3 Privacy.com: ❌"
fi

# 4.4 Coinbase Account
if [ -f "config/financial/coinbase.json" ]; then
    echo "4.4 Coinbase: ✅"
else
    echo "4.4 Coinbase: ❌"
fi

# 4.5 Virtual Cards Active
if [ -d "autonomy/financial" ]; then
    echo "4.5 Virtual Cards: Check manually: python3 autonomy/financial/cards.py --list"
else
    echo "4.5 Virtual Cards: ❌ Module missing"
fi

# 4.6 KYC Documented
if [ -f "config/financial/requirements.md" ]; then
    echo "4.6 KYC Docs: ✅"
else
    echo "4.6 KYC Docs: ❌"
fi

# 4.7 Can Pay Cloud Bills
if [ -d "autonomy/financial" ] && [ -f "config/financial/privacy.json" ]; then
    echo "4.7 Cloud Payments: Ready to test"
else
    echo "4.7 Cloud Payments: ❌ Not ready"
fi
