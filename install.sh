#!/usr/bin/env bash
# =============================================================================
#  DMAI — One-Click Installer
#  Run:  bash install.sh
#  Installs all dependencies, configures .env, and starts the system.
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

ok()   { echo -e "${GRN}  [OK]${RST} $*"; }
info() { echo -e "${CYN}  -->  ${RST}$*"; }
warn() { echo -e "${YLW}  [!]  ${RST}$*"; }
fail() { echo -e "${RED}  [ERR]${RST} $*"; exit 1; }
step() { echo -e "\n${BLD}${CYN}━━━  $*  ━━━${RST}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
clear
echo -e "${BLD}${CYN}"
cat << 'BANNER'
  ██████╗ ███╗   ███╗ █████╗ ██╗
  ██╔══██╗████╗ ████║██╔══██╗██║
  ██║  ██║██╔████╔██║███████║██║
  ██║  ██║██║╚██╔╝██║██╔══██║██║
  ██████╔╝██║ ╚═╝ ██║██║  ██║██║
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
  Autonomous Intelligence System  v6.0.0
BANNER
echo -e "${RST}"
echo -e "  ${BLD}One-Click Installer${RST} — sets up everything from scratch."
echo -e "  Repo: https://github.com/Davemiles1978/dmai-system\n"

# ── Step 1: OS / Prerequisites ────────────────────────────────────────────────
step "Step 1 / 7 — Checking Prerequisites"

# Python 3.11+
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJ=$(echo "$PY_VER" | cut -d. -f1)
    PY_MIN=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJ" -ge 3 ] && [ "$PY_MIN" -ge 11 ]; then
        ok "Python $PY_VER found"
    else
        warn "Python $PY_VER found — DMAI requires 3.11+."
        info "Install Python 3.11: https://www.python.org/downloads/"
        # Try pyenv auto-install on macOS
        if command -v brew &>/dev/null; then
            info "Attempting: brew install python@3.11"
            brew install python@3.11 || fail "Could not install Python 3.11 via Homebrew."
            # Re-point python3
            export PATH="$(brew --prefix python@3.11)/bin:$PATH"
        else
            fail "Please install Python 3.11+ manually, then re-run this script."
        fi
    fi
else
    fail "Python 3 not found. Install Python 3.11+: https://www.python.org/downloads/"
fi

PYTHON=$(command -v python3.11 2>/dev/null || command -v python3)
info "Using: $PYTHON"

# pip
$PYTHON -m pip --version &>/dev/null || fail "pip not found. Run: $PYTHON -m ensurepip --upgrade"
ok "pip found"

# git
command -v git &>/dev/null || fail "git not found. Install via: brew install git  OR  apt install git"
ok "git found"

# ── Step 2: Clone / Update Repo ───────────────────────────────────────────────
step "Step 2 / 7 — Repository"

REPO_URL="https://github.com/Davemiles1978/dmai-system.git"
INSTALL_DIR="${DMAI_DIR:-$HOME/dmai-system}"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Repo already exists at $INSTALL_DIR — pulling latest..."
    git -C "$INSTALL_DIR" pull origin main --ff-only || warn "Pull failed (local changes?). Continuing with existing code."
else
    info "Cloning into $INSTALL_DIR ..."
    git clone "$REPO_URL" "$INSTALL_DIR" || fail "Clone failed. Check your internet connection."
fi
ok "Repository ready at $INSTALL_DIR"
cd "$INSTALL_DIR"

# ── Step 3: Virtual Environment ───────────────────────────────────────────────
step "Step 3 / 7 — Python Virtual Environment"

VENV_DIR="$INSTALL_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR" || fail "Failed to create venv."
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# Activate
source "$VENV_DIR/bin/activate"
info "Active Python: $(python --version)"

# Upgrade pip silently
python -m pip install --upgrade pip --quiet
ok "pip upgraded"

# ── Step 4: Install Dependencies ──────────────────────────────────────────────
step "Step 4 / 7 — Installing Dependencies"

info "Installing core requirements (this may take 2-3 minutes)..."
pip install -r requirements.txt --quiet \
    --no-warn-script-location \
    2>&1 | grep -E "^(ERROR|WARNING|Successfully installed)" || true

info "Installing training requirements..."
pip install -r requirements_training.txt --quiet \
    --no-warn-script-location \
    2>&1 | grep -E "^(ERROR|WARNING|Successfully installed)" || true

ok "All dependencies installed"

# ── Step 5: Environment Configuration ────────────────────────────────────────
step "Step 5 / 7 — Environment Configuration"

ENV_FILE="$INSTALL_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo -e "\n  ${YLW}An existing .env file was found.${RST}"
    read -rp "  Overwrite it? [y/N]: " OVERWRITE_ENV
    if [[ ! "$OVERWRITE_ENV" =~ ^[Yy]$ ]]; then
        ok ".env kept as-is — skipping configuration wizard"
        SKIP_ENV=true
    else
        SKIP_ENV=false
    fi
else
    SKIP_ENV=false
fi

if [ "$SKIP_ENV" = false ]; then
    info "Copying .env.template -> .env"
    cp .env.template "$ENV_FILE"

    echo ""
    echo -e "  ${BLD}Enter your configuration values.${RST}"
    echo -e "  Press ENTER to skip optional keys (you can fill them later in .env).\n"

    prompt_env() {
        local KEY="$1"
        local LABEL="$2"
        local REQUIRED="${3:-optional}"
        local DEFAULT="${4:-}"

        if [ "$REQUIRED" = "required" ]; then
            while true; do
                read -rp "  ${BLD}$LABEL${RST} [required]: " VAL
                if [ -n "$VAL" ]; then break; fi
                echo -e "  ${RED}This field is required.${RST}"
            done
        else
            read -rp "  $LABEL [optional, Enter to skip]: " VAL
        fi

        VAL="${VAL:-$DEFAULT}"
        if [ -n "$VAL" ]; then
            # Replace placeholder in .env
            sed -i.bak "s|${KEY}=your_value_here|${KEY}=${VAL}|g" "$ENV_FILE"
            sed -i.bak "s|${KEY}=.*#|${KEY}=${VAL}  #|g" "$ENV_FILE"
            # Clean up backup files
            rm -f "${ENV_FILE}.bak"
        fi
    }

    # ── Required ──────────────────────────────────────────────────────────────
    echo -e "  ${BLD}--- REQUIRED ---${RST}"
    prompt_env "MASTER_PASSWORD"    "Master admin password"             required
    prompt_env "DATABASE_URL"       "PostgreSQL DATABASE_URL"           required

    # ── Core AI (at least one recommended) ───────────────────────────────────
    echo -e "\n  ${BLD}--- AI PROVIDERS (add at least one) ---${RST}"
    prompt_env "OPENAI_API_KEY"     "OpenAI API key"
    prompt_env "ANTHROPIC_API_KEY"  "Anthropic (Claude) API key"
    prompt_env "GEMINI_API_KEY"     "Google Gemini API key"
    prompt_env "DEEPSEEK_API_KEY"   "DeepSeek API key"
    prompt_env "PERPLEXITY_API_KEY" "Perplexity API key"
    prompt_env "XAI_API_KEY"        "xAI / Grok API key"

    # ── Media / Content ───────────────────────────────────────────────────────
    echo -e "\n  ${BLD}--- MEDIA & CONTENT ---${RST}"
    prompt_env "ELEVENLABS_API_KEY" "ElevenLabs TTS key"
    prompt_env "RUNWAY_API_KEY"     "Runway ML video key"
    prompt_env "STABILITY_API_KEY"  "Stability AI key"
    prompt_env "REPLICATE_API_KEY"  "Replicate key"

    # ── Notifications ─────────────────────────────────────────────────────────
    echo -e "\n  ${BLD}--- NOTIFICATIONS ---${RST}"
    prompt_env "TELEGRAM_BOT_TOKEN" "Telegram bot token"
    prompt_env "TELEGRAM_CHAT_ID"   "Telegram chat ID"

    # ── Optional advanced ─────────────────────────────────────────────────────
    echo -e "\n  ${BLD}--- ADVANCED (optional, can add later) ---${RST}"
    prompt_env "HUGGINGFACE_API_KEY"  "Hugging Face API key"
    prompt_env "PINECONE_API_KEY"     "Pinecone vector DB key"
    prompt_env "PINECONE_INDEX"       "Pinecone index name"
    prompt_env "ALPACA_API_KEY"       "Alpaca trading key"
    prompt_env "ALPACA_SECRET_KEY"    "Alpaca trading secret"
    prompt_env "GITHUB_TOKEN_MAIN"    "GitHub personal access token"

    # Set local defaults
    sed -i.bak "s|RENDER=false|RENDER=false|g" "$ENV_FILE"
    sed -i.bak "s|PORT=10000|PORT=5000|g"      "$ENV_FILE"
    rm -f "${ENV_FILE}.bak"

    ok ".env written to $ENV_FILE"
fi

# ── Step 6: Data Directory + Health Check ─────────────────────────────────────
step "Step 6 / 7 — Pre-Flight Checks"

mkdir -p data/
ok "data/ directory ready"

# Run the pre-flight checker if it exists
if [ -f "scripts/check_system.py" ]; then
    info "Running system health check..."
    python scripts/check_system.py 2>&1 | sed 's/^/    /' || warn "Some checks flagged — review above. The system will still start."
fi

# ── Step 7: Launch ────────────────────────────────────────────────────────────
step "Step 7 / 7 — Launch"

echo ""
echo -e "  ${BLD}How would you like to run DMAI?${RST}"
echo -e "  ${CYN}1)${RST} Start now (foreground — Ctrl+C to stop)"
echo -e "  ${CYN}2)${RST} Start in background (nohup — logs to dmai.log)"
echo -e "  ${CYN}3)${RST} Install as a launchd service (macOS auto-start on login)"
echo -e "  ${CYN}4)${RST} Deploy to Render (opens browser)"
echo -e "  ${CYN}5)${RST} Exit — I'll start it manually"
echo ""
read -rp "  Choice [1-5]: " LAUNCH_CHOICE

case "$LAUNCH_CHOICE" in
    1)
        info "Starting DMAI on http://localhost:5000 ..."
        info "Dashboard: http://localhost:5000/dashboard"
        info "API:       http://localhost:5000/api/status"
        echo ""
        python dmai_core_complete.py
        ;;
    2)
        info "Starting DMAI in background..."
        nohup python dmai_core_complete.py > dmai.log 2>&1 &
        DMAI_PID=$!
        echo "$DMAI_PID" > dmai.pid
        sleep 2
        if kill -0 "$DMAI_PID" 2>/dev/null; then
            ok "DMAI running (PID $DMAI_PID)"
            ok "Dashboard: http://localhost:5000/dashboard"
            ok "Logs:      tail -f $INSTALL_DIR/dmai.log"
            ok "Stop:      kill \$(cat $INSTALL_DIR/dmai.pid)"
        else
            fail "Process exited immediately. Check: cat $INSTALL_DIR/dmai.log"
        fi
        ;;
    3)
        # macOS launchd plist
        PLIST_DIR="$HOME/Library/LaunchAgents"
        PLIST_FILE="$PLIST_DIR/com.dmai.system.plist"
        mkdir -p "$PLIST_DIR"
        cat > "$PLIST_FILE" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dmai.system</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_DIR/bin/python</string>
        <string>$INSTALL_DIR/dmai_core_complete.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/dmai.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/dmai_error.log</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
PLIST
        launchctl load "$PLIST_FILE"
        ok "DMAI registered as launchd service"
        ok "It will now start automatically on login"
        ok "Start:   launchctl start com.dmai.system"
        ok "Stop:    launchctl stop  com.dmai.system"
        ok "Remove:  launchctl unload $PLIST_FILE"
        ok "Logs:    tail -f $INSTALL_DIR/dmai.log"
        ;;
    4)
        info "Opening Render dashboard..."
        open "https://dashboard.render.com" 2>/dev/null || \
            echo "  Visit: https://dashboard.render.com"
        echo ""
        echo -e "  ${BLD}Render deployment steps:${RST}"
        echo -e "  1. New > Web Service > Connect GitHub > Davemiles1978/dmai-system"
        echo -e "  2. Runtime: Python 3  |  Build: pip install -r requirements.txt"
        echo -e "  3. Start: gunicorn dmai_core_complete:app --bind 0.0.0.0:\$PORT --timeout 120 --workers 1 --threads 2"
        echo -e "  4. Add all env vars from your .env file in the Render Environment tab"
        echo -e "  5. Click Deploy"
        ;;
    *)
        echo ""
        ok "Installation complete. Start manually with:"
        echo -e "    cd $INSTALL_DIR"
        echo -e "    source .venv/bin/activate"
        echo -e "    python dmai_core_complete.py"
        ;;
esac

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLD}${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}${GRN}  DMAI Installation Complete${RST}"
echo -e "${BLD}${GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo ""
echo -e "  Dashboard  : ${CYN}http://localhost:5000/dashboard${RST}"
echo -e "  API status : ${CYN}http://localhost:5000/api/status${RST}"
echo -e "  Admin      : ${CYN}http://localhost:5000/admin${RST}"
echo -e "  Docs       : ${CYN}$INSTALL_DIR/docs/COMPLETE_SETUP_GUIDE.md${RST}"
echo -e "  Edit keys  : ${CYN}$ENV_FILE${RST}"
echo ""
