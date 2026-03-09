#!/bin/bash
# DMAI Unified Launcher - Sets correct environment for all scripts

export DMAI_ROOT="/Users/davidmiles/Desktop/dmai-system"
export PYTHONPATH="$DMAI_ROOT:$PYTHONPATH"

cd "$DMAI_ROOT"
source venv/bin/activate

echo "🚀 DMAI Environment Launcher"
echo "============================="
echo "📂 Root: $DMAI_ROOT"
echo "🐍 Python: $(which python3)"
echo "📚 PYTHONPATH: $PYTHONPATH"
echo ""

# Function to run a script with proper environment
run_script() {
    echo "▶️ Running: $1"
    echo "----------------------------------------"
    python3 "$@"
    echo "----------------------------------------"
}

# If arguments provided, run that script
if [ $# -gt 0 ]; then
    run_script "$@"
else
    echo "Usage: ./dmai_launcher.sh <script.py> [args...]"
    echo ""
    echo "Examples:"
    echo "  ./dmai_launcher.sh language_learning/internet_learner.py --test"
    echo "  ./dmai_launcher.sh services/book_reader.py --continuous"
    echo "  ./dmai_launcher.sh voice/dmai_voice_with_learning.py"
fi
