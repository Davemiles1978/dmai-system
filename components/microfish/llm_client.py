"""
LLM adapter -> routes Microfish prompts through DMAI's 13-provider waterfall.
No external SDK dependencies; uses dmai_core_complete._direct_provider_chat.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MicrofishLLM:
    """Thin LLM client. Calls DMAI's existing provider waterfall."""

    def __init__(self, chat_fn=None):
        self._chat_fn = chat_fn  # lazily resolved if None

    def _resolve_chat(self):
        if self._chat_fn is not None:
            return self._chat_fn
        try:
            from dmai_core_complete import _direct_provider_chat  # type: ignore
            self._chat_fn = _direct_provider_chat
            return self._chat_fn
        except Exception as e:
            logger.warning("MicrofishLLM: could not resolve _direct_provider_chat: %s", e)
            return None

    def chat(self, prompt: str, *, system: Optional[str] = None) -> Optional[str]:
        fn = self._resolve_chat()
        if fn is None:
            return None
        full = f"{system}\n\n{prompt}" if system else prompt
        try:
            out = fn(full)
            if isinstance(out, tuple):
                out = out[0] if out else None
            return out if isinstance(out, str) else (str(out) if out else None)
        except Exception as e:
            logger.warning("MicrofishLLM.chat failed: %s", e)
            return None

    def chat_json(self, prompt: str, *, system: Optional[str] = None,
                  default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Chat and parse JSON. Strips code fences, tolerates trailing junk."""
        sys_msg = (system or "") + "\nRespond with valid JSON only. No prose, no code fences."
        raw = self.chat(prompt, system=sys_msg.strip())
        if not raw:
            return default or {}
        # strip code fences if present
        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        # find first { ... } or [ ... ] block
        m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if m:
            s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            # one more attempt: remove trailing commas
            s2 = re.sub(r",(\s*[}\]])", r"\1", s)
            try:
                return json.loads(s2)
            except Exception as e:
                logger.warning("MicrofishLLM.chat_json parse failed: %s | raw=%r", e, raw[:200])
                return default or {}
