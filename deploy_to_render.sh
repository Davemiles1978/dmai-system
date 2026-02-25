#!/bin/bash
# DEPLOY TO RENDER.COM - FREE, EASY, WORKS FROM ANY DEVICE

echo "🚀 Deploying DMAI to Render.com (FREE forever)"
echo "=============================================="

# Create render.yaml configuration
cat > render.yaml << 'EOF'
services:
  - type: web
    name: dmai
    env: python
    buildCommand: |
      pip install -r requirements.txt
      python mark_all_for_evolution.py
    startCommand: |
      python api_server.py &
      python -m http.server 8080 --bind 0.0.0.0 --directory ui
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    autoDeploy: true
    plan: free
EOF

# Create requirements.txt if it doesn't exist
cat > requirements.txt << 'EOF'
flask
requests
numpy
EOF

echo ""
echo "✅ Render.com config created!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Go to https://render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub or upload this folder"
echo "4. Choose 'Free' plan"
echo "5. Click 'Create Web Service'"
echo ""
echo "🌐 AFTER DEPLOYMENT, YOU'LL GET A URL LIKE:"
echo "   https://dmai.onrender.com"
echo ""
echo "📱 ACCESS FROM ANY DEVICE USING THAT URL!"
