#!/usr/bin/env python3
"""
DMAI Pre-Flight Checker
=======================
Run before launch to verify all system requirements are met.
Usage:  python scripts/check_ready.py
        python scripts/check_ready.py --json     (machine-readable output)
"""

import sys
import os
import json
import socket
import importlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ── Colour helpers ────────────────────────────────────────────────────────────
USE_COLOUR = sys.stdout.isatty()
def c(colour, text):
    codes = {"red": "\033[0;31m", "grn": "\033[0;32m", "ylw": "\033[1;33m",
             "cyn": "\033[0;36m", "bld": "\033[1m", "rst": "\033[0m"}
    if not USE_COLOUR:
        return text
    return f"{codes.get(colour,'')}{text}{codes['rst']}"

def ok(label, detail=""):
    sym = c("grn", "  [PASS]")
    print(f"{sym}  {c('bld', label)}" + (f"  — {detail}" if detail else ""))
def warn(label, detail=""):
    sym = c("ylw", "  [WARN]")
    print(f"{sym}  {c('bld', label)}" + (f"  — {detail}" if detail else ""))
def fail(label, detail=""):
    sym = c("red", "  [FAIL]")
    print(f"{sym}  {c('bld', label)}" + (f"  — {detail}" if detail else ""))
def section(title):
    print(f"\n{c('cyn', c('bld', f'--- {title} ---'))}")


# ── Checks ────────────────────────────────────────────────────────────────────
results = []

def record(status, label, detail="", fix=""):
    results.append({"status": status, "label": label, "detail": detail, "fix": fix})
    if status == "pass":   ok(label, detail)
    elif status == "warn": warn(label, detail)
    else:                  fail(label, detail)


# ─── 1. Python version ────────────────────────────────────────────────────────
section("Python")
vi = sys.version_info
py_str = f"{vi.major}.{vi.minor}.{vi.micro}"
if vi.major == 3 and vi.minor >= 11:
    record("pass", "Python version", py_str)
elif vi.major == 3 and vi.minor >= 9:
    record("warn", "Python version", f"{py_str} (3.11+ recommended)",
           fix="Install Python 3.11: https://www.python.org/downloads/")
else:
    record("fail", "Python version", f"{py_str} — 3.11+ required",
           fix="Install Python 3.11: https://www.python.org/downloads/")


# ─── 2. Required packages ────────────────────────────────────────────────────
section("Core Packages")
REQUIRED_PACKAGES = [
    ("flask",             "Flask web framework"),
    ("flask_cors",        "Flask CORS"),
    ("dotenv",            "python-dotenv"),
    ("requests",          "HTTP requests"),
    ("aiohttp",           "Async HTTP"),
    ("httpx",             "httpx"),
    ("psutil",            "System utils"),
    ("networkx",          "Knowledge graph"),
    ("numpy",             "NumPy"),
    ("bs4",               "BeautifulSoup4"),
    ("feedparser",        "Feed parser"),
]
OPTIONAL_PACKAGES = [
    ("openai",            "OpenAI SDK"),
    ("anthropic",         "Anthropic SDK"),
    ("google.generativeai", "Google Gemini SDK"),
    ("psycopg2",          "PostgreSQL driver"),
    # ("neo4j", "Neo4j driver"),  # Removed — using SQLite
    ("telegram",          "Telegram bot"),
    ("replicate",         "Replicate SDK"),
    ("pinecone",          "Pinecone vector DB"),
    ("alpaca_trade_api",  "Alpaca trading"),
]

for pkg, label in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
        record("pass", label)
    except ImportError:
        record("fail", label, "not installed",
               fix=f"pip install {pkg.split('.')[0]}")

section("Optional Packages")
for pkg, label in OPTIONAL_PACKAGES:
    try:
        importlib.import_module(pkg)
        record("pass", label)
    except ImportError:
        record("warn", label, "not installed — feature will be disabled")


# ─── 3. Environment variables ────────────────────────────────────────────────
section("Environment Variables")

# Load .env if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        record("pass", ".env file", str(env_path))
    except ImportError:
        # Manual parse
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        record("warn", ".env file", "loaded without python-dotenv")
else:
    record("fail", ".env file", f"not found at {env_path}",
           fix="cp .env.template .env  then fill in your values")

REQUIRED_ENV = ["MASTER_PASSWORD", "DATABASE_URL"]
RECOMMENDED_ENV = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY", "PERPLEXITY_API_KEY",
]
OPTIONAL_ENV = [
    "ELEVENLABS_API_KEY", "RUNWAY_API_KEY", "STABILITY_API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "REPLICATE_API_KEY", "PINECONE_API_KEY", "HUGGINGFACE_API_KEY",
    "ALPACA_API_KEY", "GITHUB_TOKEN_MAIN",
]

for key in REQUIRED_ENV:
    val = os.environ.get(key, "")
    if val and val != "your_value_here":
        record("pass", key, f"set ({len(val)} chars)")
    else:
        record("fail", key, "not set or still placeholder",
               fix=f"Set {key} in your .env file")

section("Recommended API Keys (at least one AI provider needed)")
at_least_one_ai = False
for key in RECOMMENDED_ENV:
    val = os.environ.get(key, "")
    if val and val != "your_value_here":
        record("pass", key, f"set ({len(val)} chars)")
        at_least_one_ai = True
    else:
        record("warn", key, "not set — this provider will be disabled")

if not at_least_one_ai:
    record("fail", "AI Provider", "No AI provider keys set — training will be skipped",
           fix="Add at least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")

section("Optional Integrations")
for key in OPTIONAL_ENV:
    val = os.environ.get(key, "")
    if val and val != "your_value_here":
        record("pass", key, "set")
    else:
        record("warn", key, "not set — feature disabled until configured")


# ─── 4. Database connectivity ────────────────────────────────────────────────
section("Database")
db_url = os.environ.get("DATABASE_URL", "")
if not db_url or db_url == "your_value_here":
    record("fail", "Database URL", "DATABASE_URL not set",
           fix="Set DATABASE_URL in .env (PostgreSQL connection string)")
elif db_url.startswith("postgresql") or db_url.startswith("postgres"):
    # Try a socket connection to the host
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(db_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=3):
            record("pass", "PostgreSQL reachable", f"{host}:{port}")
    except Exception as e:
        record("warn", "PostgreSQL", f"Cannot reach {host}:{port} — {e}",
               fix="Check DATABASE_URL host/port and that the DB server is running")
else:
    record("warn", "Database URL", f"Unexpected scheme: {db_url[:20]}...",
           fix="DATABASE_URL should start with postgresql://")


# ─── 5. Port availability ─────────────────────────────────────────────────────
section("Port")
port = int(os.environ.get("PORT", "5000"))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(1)
    result = s.connect_ex(("127.0.0.1", port))
if result == 0:
    record("warn", f"Port {port}", f"already in use — something is running on :{port}",
           fix=f"Kill the process: lsof -ti:{port} | xargs kill -9")
else:
    record("pass", f"Port {port}", "available")


# ─── 6. File structure ────────────────────────────────────────────────────────
section("File Structure")
ROOT = Path(__file__).parent.parent
REQUIRED_FILES = [
    "dmai_core_complete.py",
    "requirements.txt",
    ".env",
    "components/si_core",
    "static/dashboard.html",
]
for rel_path in REQUIRED_FILES:
    p = ROOT / rel_path
    if p.exists():
        record("pass", rel_path)
    else:
        record("warn", rel_path, "not found — some features may be unavailable")

# data/ dir
data_dir = ROOT / "data"
data_dir.mkdir(exist_ok=True)
record("pass", "data/ directory", str(data_dir))


# ─── 7. DMAI component imports ────────────────────────────────────────────────
section("DMAI Components")
sys.path.insert(0, str(ROOT))
COMPONENTS = [
    ("dmai_syllabus_data",                      "Syllabus data"),
    ("components.si_core",                       "SICore KPI engine"),
    ("components.training.ComprehensiveAITraining", "AI Training"),
    ("components.si_training.FullSITrainingProgram",  "SI Training"),
    ("components.update_engine.PeriodicUpdateEngine", "Update Engine"),
    ("components.phase11.AIIntegrationHub",      "AI Integration Hub"),
    ("components.orchestrator.DMAITrainingOrchestrator", "Training Orchestrator"),
]
for mod, label in COMPONENTS:
    try:
        importlib.import_module(mod)
        record("pass", label)
    except ImportError as e:
        record("warn", label, f"import failed: {e}")
    except Exception as e:
        record("warn", label, f"error: {e}")


# ─── Summary ─────────────────────────────────────────────────────────────────
section("Summary")
total   = len(results)
passed  = sum(1 for r in results if r["status"] == "pass")
warned  = sum(1 for r in results if r["status"] == "warn")
failed  = sum(1 for r in results if r["status"] == "fail")

print(f"\n  {c('bld','Checks:')}  {total} total  "
      f"{c('grn', f'{passed} passed')}  "
      f"{c('ylw', f'{warned} warnings')}  "
      f"{c('red', f'{failed} failed')}\n")

if failed > 0:
    print(c("red", "  SYSTEM NOT READY — fix failed checks before starting DMAI.\n"))
    print(c("bld", "  Required fixes:"))
    for r in results:
        if r["status"] == "fail" and r.get("fix"):
            print(f"    >> {r['label']}: {r['fix']}")
    print()
elif warned > 0:
    print(c("ylw", "  READY WITH WARNINGS — some features will be disabled."))
    print(c("ylw", "  The system will start, but add missing keys for full functionality.\n"))
else:
    print(c("grn", c("bld", "  ALL CHECKS PASSED — system is ready to run.\n")))

# Machine-readable output
if "--json" in sys.argv:
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": total, "passed": passed, "warnings": warned, "failed": failed},
        "ready": failed == 0,
        "checks": results,
    }
    print(json.dumps(out, indent=2))

sys.exit(0 if failed == 0 else 1)
