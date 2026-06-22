#!/usr/bin/env python3
"""
DMAI – Local Development Startup Script
Loads .env, validates required variables, runs DB migrations, and starts
the Flask development server on port 5000.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"

# ── Required environment variables ──────────────────────────────────────────
REQUIRED_VARS = [
    "MASTER_PASSWORD",
    "DATABASE_URL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "OPENAI_API_KEY",
]

# ── Optional vars whose presence determines which components are active ──────
COMPONENT_VARS = {
    "Telegram":          ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
    "OpenAI":            ["OPENAI_API_KEY"],
    "DeepSeek":          ["DEEPSEEK_API_KEY"],
    "Gemini":            ["GEMINI_API_KEY"],
    "Anthropic Claude":  ["ANTHROPIC_API_KEY"],
    "Perplexity":        ["PERPLEXITY_API_KEY"],
    "xAI (Grok)":        ["XAI_API_KEY"],
    "Hugging Face":      ["HUGGINGFACE_API_KEY"],
    "Google AI Studio":  ["GOOGLE_AI_STUDIO_KEY"],
    "Mistral":           ["MISTRAL_API_KEY"],
    "Stability AI":      ["STABILITY_API_KEY"],
    "ElevenLabs":        ["ELEVENLABS_API_KEY"],
    "Runway ML":         ["RUNWAY_API_KEY"],
    "Replicate":         ["REPLICATE_API_KEY"],
    "Together AI":       ["TOGETHER_API_KEY"],
    "Cohere":            ["COHERE_API_KEY"],
    "Pinecone":          ["PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_ENVIRONMENT"],
    "GitHub (main)":     ["GITHUB_TOKEN_MAIN"],
    "GitHub (secondary)":["GITHUB_TOKEN_SECONDARY"],
    "Alpaca Trading":    ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"],
    "AWS":               ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "KDP / Amazon":      ["KDP_EMAIL", "KDP_PASSWORD"],
    "PostgreSQL DB":     ["DATABASE_URL"],
}

DEV_PORT = 5000


def banner():
    print(f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════╗
║              DMAI  –  Local Development Mode             ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")


def load_env():
    """Load .env from the project root (next to this script)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(f"{YELLOW}[WARN] .env not found at {env_path}{RESET}")
        print(f"       Copy .env.template → .env and fill in your values.\n")
        return False

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"{GREEN}[OK]   Loaded environment from {env_path}{RESET}")
        return True
    except ImportError:
        # Fallback: manual parse
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.split("#")[0].strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), value)
        print(f"{GREEN}[OK]   Loaded environment from {env_path} (manual parse){RESET}")
        return True


def check_required_vars():
    """Abort if any required variable is missing or still at placeholder."""
    missing = []
    for var in REQUIRED_VARS:
        val = os.environ.get(var, "")
        if not val or val in ("your_value_here", ""):
            missing.append(var)

    if missing:
        print(f"\n{RED}[ERROR] Missing required environment variable(s):{RESET}")
        for v in missing:
            print(f"        • {v}")
        print(f"\n  Set these in your .env file before starting DMAI.\n")
        sys.exit(1)

    print(f"{GREEN}[OK]   All required variables are set{RESET}")


def run_db_migrations():
    """Run database migrations if Flask-Migrate / Alembic is available."""
    # Check if alembic is installed
    if importlib.util.find_spec("alembic") is None:
        print(f"{YELLOW}[SKIP] Alembic not installed – skipping DB migrations{RESET}")
        return

    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        print(f"{YELLOW}[SKIP] No migrations/ directory found – skipping DB migrations{RESET}")
        return

    print(f"{CYAN}[INFO] Running DB migrations (flask db upgrade)…{RESET}")
    result = subprocess.run(
        [sys.executable, "-m", "flask", "db", "upgrade"],
        env={**os.environ, "FLASK_APP": "dmai_core_complete"},
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"{GREEN}[OK]   DB migrations applied successfully{RESET}")
    else:
        print(f"{RED}[WARN] DB migration failed (non-fatal in dev):{RESET}")
        print(f"       {result.stderr.strip()[:300]}")


def active_components():
    """Return list of (component_name, active) tuples."""
    results = []
    for name, keys in COMPONENT_VARS.items():
        active = all(
            os.environ.get(k, "") not in ("", "your_value_here")
            for k in keys
        )
        results.append((name, active))
    return results


def print_component_status(components):
    active   = [n for n, ok in components if ok]
    inactive = [n for n, ok in components if not ok]

    col_width = 26
    print(f"\n{BOLD}  Active Components ({len(active)}/{len(components)}){RESET}")
    print("  " + "─" * 56)

    # Print in two columns
    all_items = [(n, ok) for n, ok in components]
    for i in range(0, len(all_items), 2):
        left  = all_items[i]
        right = all_items[i + 1] if i + 1 < len(all_items) else None

        def fmt(item):
            name, ok = item
            tick = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
            return f"  {tick} {name:<{col_width}}"

        line = fmt(left)
        if right:
            line += fmt(right)
        print(line)

    print()


def start_flask():
    """Start the Flask development server."""
    app_module = "dmai_core_complete"
    app_file   = Path(__file__).parent / f"{app_module}.py"

    if not app_file.exists():
        print(f"{RED}[ERROR] {app_file} not found.{RESET}")
        print("        Make sure dmai_core_complete.py is in the same directory.\n")
        sys.exit(1)

    os.environ["FLASK_APP"] = app_module
    os.environ["FLASK_ENV"] = "development"
    os.environ["FLASK_DEBUG"] = "1"

    print(f"{BOLD}{GREEN}  Starting Flask dev server on http://0.0.0.0:{DEV_PORT}{RESET}\n")

    try:
        from dmai_core_complete import app
        app.run(host="0.0.0.0", port=DEV_PORT, debug=True, use_reloader=True)
    except ImportError as exc:
        print(f"{RED}[ERROR] Could not import dmai_core_complete: {exc}{RESET}")
        sys.exit(1)


def main():
    banner()

    # 1. Load environment
    load_env()

    # 2. Validate required vars
    check_required_vars()

    # 3. DB migrations
    run_db_migrations()

    # 4. Show component status
    components = active_components()
    print_component_status(components)

    # 5. Start server
    start_flask()


if __name__ == "__main__":
    main()
