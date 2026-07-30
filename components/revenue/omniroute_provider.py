"""
OmniRoute Gateway — free AI provider aggregator for DMAI.

OmniRoute (https://omniroute.ai) provides a single API endpoint that
routes to multiple free/open-source LLM providers. DMAI uses this to
supplement paid providers and ensure she never hits "no provider available".

Integration approach:
1. DMAI researches OmniRoute's current API structure
2. Registers as a provider in the AI Hub's fallback chain
3. Uses OmniRoute before falling back to web search
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("OmniRoute")

# Known OmniRoute endpoints (DMAI will research and update these)
OMNIROUTE_CONFIG = {
    "repo": "https://github.com/diegosouzapw/OmniRoute",
    "base_url": "https://api.omniroute.ai/v1",
    "chat_endpoint": "/chat/completions",
    "models": ["llama-3.1-70b", "mixtral-8x7b", "gemma-2-9b", "qwen-2-72b"],
    "free_tier": True,
    "rate_limit": "100 requests/hour",
    "requires_key": False,
    "status": "needs_research",  # DMAI will verify and update
}


class OmniRouteProvider:
    """OmniRoute AI Gateway provider for DMAI's AI Hub."""

    def __init__(self):
        self.config = OMNIROUTE_CONFIG
        self.available = False
        self.last_check = None
        self.consecutive_failures = 0

    def research_and_update(self) -> Dict[str, Any]:
        """Use DMAI's web search to research current OmniRoute status.
        Updates config with latest endpoint and model information.
        """
        try:
            from dmai_core_complete import _ai_chat
            
            prompt = """Analyze the OmniRoute GitHub repository (github.com/diegosouzapw/OmniRoute).
This is a free AI gateway that provides access to multiple LLMs through a single API.
Research and determine:
1. Current API base URL and endpoint structure from the code
2. Available models and their capabilities
3. Rate limits and any authentication requirements
4. How to send chat completion requests (headers, payload format)
5. Is it currently active and usable?

Return as JSON with keys: base_url, chat_endpoint, models, rate_limit, requires_key, 
auth_header, example_payload, status (active/inactive/needs_key), notes.
Only include verified information found in the repository code."""
            
            response = _ai_chat(prompt)
            if response:
                # Let DMAI parse and update her own config
                self._update_from_research(response)
                return {"status": "researched", "available": self.available,
                        "models": self.config.get("models", [])}
        except Exception as e:
            logger.warning("OmniRoute research failed: %s", e)
        return {"status": "research_failed", "available": False}

    def _update_from_research(self, research: str):
        """Parse research results and update config."""
        import re
        try:
            json_match = re.search(r'\{.*\}', research, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                self.config.update(data)
                self.available = data.get("status") == "active"
                self.last_check = datetime.now(timezone.utc).isoformat()
        except Exception:
            pass

    def chat(self, messages: list, model: str = None, **kwargs) -> Optional[str]:
        """Send a chat request through OmniRoute."""
        if self.consecutive_failures >= 3:
            return None

        import requests as _req
        try:
            url = f"{self.config['base_url']}{self.config['chat_endpoint']}"
            payload = {
                "model": model or self.config["models"][0],
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            headers = {"Content-Type": "application/json"}
            
            r = _req.post(url, json=payload, headers=headers, timeout=30)
            
            if r.status_code == 200:
                data = r.json()
                self.consecutive_failures = 0
                self.available = True
                return data["choices"][0]["message"]["content"]
            else:
                self.consecutive_failures += 1
                logger.debug("OmniRoute returned %d: %s", r.status_code, r.text[:200])
                if self.consecutive_failures >= 3:
                    self.available = False
                return None
        except Exception as e:
            self.consecutive_failures += 1
            logger.debug("OmniRoute request failed: %s", e)
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider": "omniroute",
            "available": self.available,
            "models": self.config.get("models", []),
            "rate_limit": self.config.get("rate_limit", "unknown"),
            "free_tier": self.config.get("free_tier", True),
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.last_check,
        }
