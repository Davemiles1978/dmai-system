#!/bin/bash
# Launch AI Evolution System with virtual environment

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting AI Evolution System..."
echo "📡 Server starting at http://localhost:8080"
open http://localhost:8080
python -m http.server 8080 --directory ui#!/bin/bash
echo "🚀 Starting AI Evolution System..."
echo "📡 Server starting at http://localhost:8080"
open http://localhost:8080
python3 -m http.server 8080 --directory ui
