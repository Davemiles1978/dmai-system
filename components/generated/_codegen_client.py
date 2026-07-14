"""OpenRouter cascade client for capability code generation.

Two models, cheap first:

1. ``openai/gpt-4o-mini`` for the initial attempt.
2. ``anthropic/claude-sonnet-4.5`` for a retry when the first
   attempt failed validation. Sonnet is invoked at most once per
   candidate.

Requires ``OPENROUTER_API_KEY`` in the environment. If it's missing
the client returns ``None`` from every call - the caller must treat
that as a hard failure. This mirrors ``knowledge_acquirer._try_llm``.

Everything is a thin wrapper around ``urllib.request`` so no extra
dependency is added. Timeouts are strict (30s per call).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_PRIMARY  = "openai/gpt-4o-mini"
MODEL_FALLBACK = "anthropic/claude-sonnet-4.5"

REQUEST_TIMEOUT_SEC = 30
MAX_TOKENS_DEFAULT  = 1500


# ── Prompt template ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You write Python 3.11 capability modules for DMAI, an autonomous "
    "AI system. Every module you produce MUST follow this exact "
    "shape:\n"
    "\n"
    '"""One-paragraph module docstring: what this capability does '
    "and why it is useful to DMAI. The docstring is inspected by "
    'DMAI\'s self-judge, so state the concept plainly."""\n'
    "\n"
    "from __future__ import annotations\n"
    "# ...imports (see allow-list below)...\n"
    "\n"
    "def run(**kwargs):\n"
    '    """Entry point. Accept keyword args, return a serialisable value."""\n'
    "    ...\n"
    "\n"
    "Constraints:\n"
    "- Only import from this allow-list: math, statistics, json, re, "
    "hashlib, datetime, collections, itertools, functools, operator, "
    "dataclasses, enum, typing, uuid, random, decimal, string, "
    "textwrap, difflib, bisect, heapq, copy, sqlite3, "
    "components.knowledge, components.self_judge.\n"
    "- No file writes, no subprocess, no networking, no eval/exec/open.\n"
    "- Pure function of its kwargs and (optionally) read-only SQLite.\n"
    "- If SQLite is used, open with `sqlite3.connect(db_path)` where "
    "`db_path` is a kwarg; never hardcode a path.\n"
    "- Include full type hints on `run` and on any helper functions.\n"
    "- Return only the Python source. No prose, no fences, no "
    "explanation."
)


USER_PROMPT_TEMPLATE = (
    "CAPABILITY CONCEPT: {concept}\n"
    "\n"
    "SUPPORTING INSIGHT: {insight}\n"
    "\n"
    "CAPABILITY TYPE: {capability_type}\n"
    "\n"
    "REQUIRED PUBLIC API:\n"
    "  def run(**kwargs) -> <serialisable>\n"
    "\n"
    "HAPPY-PATH KWARGS (must succeed with these): {happy_kwargs}\n"
    "\n"
    "Write the module. Start with the docstring; end with the last "
    "line of the last function."
)


RETRY_HINT_TEMPLATE = (
    "The previous attempt was rejected. Reasons:\n"
    "{reasons}\n"
    "\n"
    "Regenerate the module. Address every reason above. Keep the "
    "docstring aligned with the capability concept."
)


# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class CodegenAttempt:
    ok: bool
    source: Optional[str] = None
    model: str = ""
    reason: str = ""
    usage: dict = field(default_factory=dict)


# ── HTTP ──────────────────────────────────────────────────────────────────

def _post_openrouter(model: str, messages: List[dict],
                     max_tokens: int) -> Optional[dict]:
    """POST to OpenRouter. Returns parsed JSON or ``None`` on failure."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.info("codegen_client: OPENROUTER_API_KEY unset - skipping")
        return None

    payload = json.dumps({
        "model":      model,
        "messages":   messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://dmai-web.onrender.com",
            "X-Title":       "DMAI capability materialiser",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        logger.warning("codegen_client: HTTP failure for %s: %s", model, e)
        return None
    except json.JSONDecodeError as e:
        logger.warning("codegen_client: non-JSON response from %s: %s",
                       model, e)
        return None


# ── Code extractor ────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip optional Markdown fences. If none present, return as-is."""
    m = _FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


# ── Public API ────────────────────────────────────────────────────────────

def request_code(concept: str,
                 insight: str,
                 capability_type: str,
                 happy_kwargs: dict,
                 *,
                 model: str = MODEL_PRIMARY,
                 retry_reasons: Optional[Iterable[str]] = None,
                 max_tokens: int = MAX_TOKENS_DEFAULT) -> CodegenAttempt:
    """Generate a candidate module source string.

    On the first attempt pass ``retry_reasons=None``. On a retry pass
    the validator/sandbox reasons so the LLM knows what to fix.
    """
    user = USER_PROMPT_TEMPLATE.format(
        concept=concept[:400],
        insight=insight[:800],
        capability_type=capability_type[:80],
        happy_kwargs=json.dumps(happy_kwargs, default=str)[:400],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    if retry_reasons:
        messages.append({
            "role": "user",
            "content": RETRY_HINT_TEMPLATE.format(
                reasons="\n".join(f"  - {r}" for r in retry_reasons)[:2000]
            ),
        })

    resp = _post_openrouter(model, messages, max_tokens=max_tokens)
    if resp is None:
        return CodegenAttempt(
            ok=False, model=model, reason="http_or_auth_failure",
        )
    try:
        choice0 = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return CodegenAttempt(
            ok=False, model=model, reason="malformed_response",
            usage=resp.get("usage") or {},
        )
    source = _extract_code(choice0)
    if not source or "def run" not in source:
        return CodegenAttempt(
            ok=False, model=model, reason="no_run_in_response",
            source=source, usage=resp.get("usage") or {},
        )
    return CodegenAttempt(
        ok=True, model=model, source=source,
        usage=resp.get("usage") or {},
    )


def request_code_cascade(concept: str,
                         insight: str,
                         capability_type: str,
                         happy_kwargs: dict,
                         *,
                         primary_reasons: Optional[Iterable[str]] = None
                         ) -> List[CodegenAttempt]:
    """Try the primary model, retry with fallback if primary fails.

    Returns a list with one or two attempts. The caller decides
    which (if any) actually passed downstream validation.
    """
    first = request_code(
        concept, insight, capability_type, happy_kwargs,
        model=MODEL_PRIMARY, retry_reasons=None,
    )
    if first.ok:
        return [first]
    second = request_code(
        concept, insight, capability_type, happy_kwargs,
        model=MODEL_FALLBACK,
        retry_reasons=primary_reasons or [first.reason or "primary_failed"],
    )
    return [first, second]


__all__ = [
    "MODEL_PRIMARY",
    "MODEL_FALLBACK",
    "CodegenAttempt",
    "request_code",
    "request_code_cascade",
]
