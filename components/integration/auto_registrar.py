"""
DMAI AutoRegistrar
==================
Automatically acquires API keys for free-tier providers that support
programmatic registration. Where an API doesn't allow programmatic signup,
it surfaces the direct signup URL so the operator (David) can action it
in under 60 seconds.

Strategy per provider
---------------------
- HuggingFace  → POST /api/users (public registration endpoint) + token creation
- GitHub Models → already covered by GITHUB_TOKEN — no extra registration
- Groq          → no public signup API — surface link
- OpenRouter    → no public signup API — surface link
- Cohere        → no public signup API — surface link
- Tavily        → no public signup API — surface link
- Cerebras      → no public signup API — surface link
- Mistral       → no public signup API — surface link
- Cloudflare    → requires account + zone, manual — surface link
- Google AI     → requires Google account, manual — surface link

Critical rules
--------------
- NEVER store or use credentials that don't belong to this DMAI deployment.
- NEVER scrape, harvest, or reuse keys from third parties.
- All registered accounts use the DMAI service email from DMAI_EMAIL env var.
- Only call register_all() from a background thread, never on the request path.
"""

import os
import re
import json
import time
import logging
import threading
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Registration state file ────────────────────────────────────────────────────
STATE_FILE = Path("data/auto_registrar_state.json")

# ── Provider registration specs ───────────────────────────────────────────────
# Each entry: (provider_id, env_var, can_auto_register, signup_url, notes)
PROVIDERS = [
    {
        "id":           "groq",
        "env_var":      "GROQ_API_KEY",
        "auto":         False,
        "signup_url":   "https://console.groq.com/keys",
        "free_tier":    "14,400 req/day — fastest inference available (Llama 3.3 70B)",
        "time_seconds": 45,
    },
    {
        "id":           "openrouter",
        "env_var":      "OPENROUTER_API_KEY",
        "auto":         False,
        "signup_url":   "https://openrouter.ai/keys",
        "free_tier":    "Many free models at $0 — Gemma, Llama, Mistral free tier",
        "time_seconds": 60,
    },
    {
        "id":           "huggingface",
        "env_var":      "HUGGINGFACE_API_KEY",
        "auto":         True,
        "signup_url":   "https://huggingface.co/settings/tokens",
        "free_tier":    "$0.10/month free credit, 1000+ models",
        "time_seconds": 5,
    },
    {
        "id":           "cohere",
        "env_var":      "COHERE_API_KEY",
        "auto":         False,
        "signup_url":   "https://dashboard.cohere.com/api-keys",
        "free_tier":    "1,000 req/month, 20 req/min — Command R+",
        "time_seconds": 60,
    },
    {
        "id":           "tavily",
        "env_var":      "TAVILY_API_KEY",
        "auto":         False,
        "signup_url":   "https://tavily.com/#api",
        "free_tier":    "1,000 searches/month free",
        "time_seconds": 45,
    },
    {
        "id":           "cerebras",
        "env_var":      "CEREBRAS_API_KEY",
        "auto":         False,
        "signup_url":   "https://cloud.cerebras.ai",
        "free_tier":    "Free tier — Llama 3.1 70B at 2,100 tokens/sec",
        "time_seconds": 60,
    },
    {
        "id":           "mistral",
        "env_var":      "MISTRAL_API_KEY",
        "auto":         False,
        "signup_url":   "https://console.mistral.ai",
        "free_tier":    "Free tier on Mistral 7B + Codestral Mamba",
        "time_seconds": 60,
    },
    {
        "id":           "cloudflare",
        "env_var":      "CLOUDFLARE_API_KEY",
        "auto":         False,
        "signup_url":   "https://dash.cloudflare.com/profile/api-tokens",
        "free_tier":    "10,000 neurons/day — Llama 3.1, DeepSeek R1",
        "time_seconds": 90,
    },
    {
        "id":           "google_ai_studio",
        "env_var":      "GOOGLE_AI_STUDIO_KEY",
        "auto":         False,
        "signup_url":   "https://aistudio.google.com/apikey",
        "free_tier":    "1,500 req/day, 250K tokens/min — Gemini Flash",
        "time_seconds": 45,
    },
    {
        "id":           "perplexity",
        "env_var":      "PERPLEXITY_API_KEY",
        "auto":         False,
        "signup_url":   "https://docs.perplexity.ai",
        "free_tier":    "$5 free credit on signup",
        "time_seconds": 45,
    },
]


class AutoRegistrar:
    """Manages automatic and guided registration for free-tier AI providers."""

    CHECK_INTERVAL = 3600  # Re-check every hour

    def __init__(self, dmai_app=None):
        self.dmai = dmai_app
        self._state = self._load_state()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background registration loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dmai-auto-registrar"
        )
        self._thread.start()
        logger.info("AutoRegistrar background loop started")

    def stop(self):
        self._stop.set()

    def register_all(self) -> Dict:
        """
        Attempt registration for all missing providers.
        Returns a summary: {auto_registered, pending_manual, already_active}
        """
        results = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "auto_registered":  [],
            "pending_manual":   [],
            "already_active":   [],
            "errors":           [],
        }

        for spec in PROVIDERS:
            pid   = spec["id"]
            evar  = spec["env_var"]
            key   = os.environ.get(evar, "").strip()

            if key and key.lower() not in ("", "none", "pending", "your_value_here"):
                results["already_active"].append(pid)
                continue

            # Try auto-registration first
            if spec.get("auto"):
                ok, new_key, err = self._auto_register(pid, spec)
                if ok and new_key:
                    os.environ[evar] = new_key
                    self._record_registered(pid, new_key)
                    results["auto_registered"].append(pid)
                    logger.info("AutoRegistrar: registered %s", pid)
                    continue
                else:
                    logger.warning("AutoRegistrar: auto-register %s failed: %s", pid, err)
                    results["errors"].append({"provider": pid, "error": err})

            # Surface signup link
            results["pending_manual"].append({
                "provider":     pid,
                "signup_url":   spec["signup_url"],
                "free_tier":    spec["free_tier"],
                "time_seconds": spec["time_seconds"],
                "env_var":      evar,
            })

        self._save_state()
        logger.info(
            "AutoRegistrar: %d auto-registered, %d pending manual, %d already active",
            len(results["auto_registered"]),
            len(results["pending_manual"]),
            len(results["already_active"]),
        )
        return results

    def get_pending_signups(self) -> List[Dict]:
        """Return providers needing manual signup, sorted by easiest first."""
        pending = []
        for spec in PROVIDERS:
            key = os.environ.get(spec["env_var"], "").strip()
            if not key or key.lower() in ("none", "pending", "your_value_here"):
                pending.append({
                    "provider":     spec["id"],
                    "signup_url":   spec["signup_url"],
                    "free_tier":    spec["free_tier"],
                    "time_seconds": spec["time_seconds"],
                    "env_var":      spec["env_var"],
                })
        pending.sort(key=lambda x: x["time_seconds"])
        return pending

    def get_status(self) -> Dict:
        """Return current registration state snapshot."""
        active, pending, auto = [], [], []
        for spec in PROVIDERS:
            key = os.environ.get(spec["env_var"], "").strip()
            if key and key.lower() not in ("none", "pending", "your_value_here"):
                active.append(spec["id"])
            elif spec.get("auto"):
                auto.append(spec["id"])
            else:
                pending.append(spec["id"])
        return {
            "active":  active,
            "pending_manual": pending,
            "can_auto": auto,
            "last_run": self._state.get("last_run"),
        }

    # ── Auto-registration implementations ─────────────────────────────────────

    def _auto_register(self, provider_id: str, spec: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Attempt programmatic registration. Returns (success, key, error)."""
        method = f"_register_{provider_id}"
        fn = getattr(self, method, None)
        if fn is None:
            return False, None, "no auto-register method"
        try:
            return fn(spec)
        except Exception as e:
            return False, None, str(e)

    def _register_huggingface(self, spec: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Create a HuggingFace account and token via the public API.
        Requires DMAI_EMAIL and DMAI_HF_PASSWORD env vars.
        """
        email    = os.environ.get("DMAI_EMAIL", "").strip()
        password = os.environ.get("DMAI_HF_PASSWORD", "").strip()

        if not email or not password:
            return False, None, "DMAI_EMAIL or DMAI_HF_PASSWORD not set"

        username = re.sub(r"[^a-z0-9]", "-", email.split("@")[0].lower())[:39]

        # Step 1: Register account
        reg_resp = requests.post(
            "https://huggingface.co/api/users",
            json={"email": email, "name": username, "password": password},
            timeout=15,
            headers={"Content-Type": "application/json"},
        )

        if reg_resp.status_code not in (200, 201, 409):  # 409 = already exists
            return False, None, f"HF registration failed: {reg_resp.status_code} {reg_resp.text[:200]}"

        # Step 2: Login to get session token
        login_resp = requests.post(
            "https://huggingface.co/login",
            data={"username": username, "password": password},
            timeout=15,
            allow_redirects=True,
        )
        cookies = login_resp.cookies

        # Step 3: Create API token
        token_resp = requests.post(
            "https://huggingface.co/settings/tokens/new",
            data={
                "name":        "dmai-auto",
                "type":        "write",
                "accessTo":    "all",
            },
            cookies=cookies,
            timeout=15,
            allow_redirects=True,
        )

        # Extract token from response
        token_match = re.search(r'(hf_[A-Za-z0-9]{32,})', token_resp.text)
        if token_match:
            return True, token_match.group(1), None

        return False, None, "Could not extract HF token from response"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _loop(self):
        """Background daemon loop."""
        while not self._stop.is_set():
            try:
                self.register_all()
            except Exception as e:
                logger.error("AutoRegistrar loop error: %s", e)
            self._stop.wait(self.CHECK_INTERVAL)

    def _load_state(self) -> Dict:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"registered": {}, "last_run": None}

    def _save_state(self):
        self._state["last_run"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2)
            tmp.rename(STATE_FILE)
        except Exception as e:
            logger.warning("AutoRegistrar state save failed: %s", e)

    def _record_registered(self, provider_id: str, key: str):
        self._state.setdefault("registered", {})[provider_id] = {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "key_prefix":    key[:8] + "...",
        }
