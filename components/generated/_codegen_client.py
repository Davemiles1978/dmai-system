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
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

MODEL_PRIMARY  = "openai/gpt-4o-mini"
MODEL_FALLBACK = "anthropic/claude-sonnet-4.5"

REQUEST_TIMEOUT_SEC = 30
MAX_TOKENS_DEFAULT  = 1500

# PR WW: floor below which we treat a request as guaranteed-fail. If
# even with the credit-adaptive resize we can't afford this many
# tokens, codegen can't produce a valid module docstring + run()
# function and we should skip the attempt instead of burning it.
MIN_VIABLE_MAX_TOKENS = 400


# PR WW: parse the OpenRouter 402 body to learn how many tokens we
# can actually afford right now.
_402_AFFORD_RE = re.compile(
    r"can only afford\s+(\d+)\s+tokens",
    re.IGNORECASE,
)


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
    # PR VV: capture the real HTTP/exception info instead of swallowing
    # everything into an opaque None. Downstream now sees a dict with
    # __error__ set on any failure, so CodegenAttempt.reason can be
    # a concrete diagnosis ('openrouter_401_invalid_key', '429_quota',
    # etc.) instead of 'http_or_auth_failure'.
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            logger.warning("codegen_client: non-JSON response from %s: %s",
                           model, e)
            return {"__error__": "non_json_response",
                    "http_status": 200,
                    "body_snippet": (body or "")[:400]}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", "replace")
        except Exception:
            err_body = ""
        logger.warning("codegen_client: HTTP %s from %s: %s",
                       e.code, model, err_body[:200])
        return {"__error__": f"http_{e.code}",
                "http_status": e.code,
                "http_reason": e.reason or "",
                "body_snippet": err_body[:400]}
    except urllib.error.URLError as e:
        logger.warning("codegen_client: URLError for %s: %s", model, e)
        return {"__error__": "url_error",
                "exception_msg": str(getattr(e, "reason", e))[:300]}
    except OSError as e:
        logger.warning("codegen_client: OSError for %s: %s", model, e)
        return {"__error__": "os_error",
                "exception_msg": str(e)[:300]}


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

    PR XX-1: try local template synthesis first for well-understood
    capability shapes. Only if the local path can't handle this type,
    or if a retry is requested (meaning a previous attempt failed
    validation and we need a smarter path), do we fall through to the
    external LLM. This is Layer 1 of DMAI's coding self-sufficiency.
    """
    # PR XX-1: local template synthesis (tier 1).
    # Skipped on retry: if the caller is asking for a retry, the
    # first attempt (which was likely also templated) didn't pass
    # smoke test, so escalate to the LLM instead of retemplating.
    if not retry_reasons:
        try:
            from components.local_codegen import (
                can_template, generate_from_template,
            )
            if can_template(capability_type):
                tr = generate_from_template(
                    concept=concept,
                    insight=insight,
                    capability_type=capability_type,
                    happy_kwargs=happy_kwargs,
                )
                if tr.ok:
                    logger.info(
                        "codegen_client: tier-1 template used (%s)",
                        tr.template_id,
                    )
                    return CodegenAttempt(
                        ok=True,
                        model=tr.template_id,
                        source=tr.source,
                        reason="",
                        usage={"template": tr.template_id,
                               "local": True},
                    )
                logger.info(
                    "codegen_client: tier-1 template declined (%s): %s",
                    capability_type, tr.reason,
                )
        except Exception as e:  # noqa: BLE001
            # Non-fatal: fall through to external LLM.
            logger.info(
                "codegen_client: local template raised, falling "
                "back to LLM: %s", e,
            )
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

    # PR WW: attempt-with-adaptive-retry on HTTP 402 ("you can only
    # afford N tokens"). The self-heal loop:
    #   1. Try the requested max_tokens.
    #   2. If OpenRouter says we can only afford N (< requested),
    #      re-attempt with max_tokens=N (once).
    #   3. If N < MIN_VIABLE_MAX_TOKENS, mark as credit_exhausted and
    #      skip - don't burn a real attempt on a doomed request.
    #
    # This keeps the picker productive on days when the OpenRouter
    # balance is low but non-zero, and produces a clean, actionable
    # 'credit_exhausted' reason (not 'http_or_auth_failure') when it
    # actually runs out.
    effective_max = max_tokens
    resp = _post_openrouter(model, messages, max_tokens=effective_max)
    if resp is None:
        # Only path that still returns None: OPENROUTER_API_KEY unset.
        return CodegenAttempt(
            ok=False, model=model, reason="openrouter_key_unset",
        )
    if resp.get("__error__"):
        err = resp["__error__"]
        hs = resp.get("http_status")
        snip = resp.get("body_snippet") or resp.get("exception_msg") or ""

        # PR WW self-heal branch: HTTP 402 with a concrete affordable-
        # token count -> retry once at that size.
        if hs == 402:
            m = _402_AFFORD_RE.search(snip or "")
            if m:
                afford = int(m.group(1))
                if afford < MIN_VIABLE_MAX_TOKENS:
                    return CodegenAttempt(
                        ok=False, model=model,
                        reason=(
                            f"credit_exhausted: openrouter can only "
                            f"afford {afford} tokens (below viable floor "
                            f"{MIN_VIABLE_MAX_TOKENS}). Top up credits or "
                            f"switch MODEL_PRIMARY to a cheaper model."
                        ),
                    )
                # Retry at the affordable size (leave 8-token headroom
                # so we don't tickle the same 402 back on a rounding).
                effective_max = max(MIN_VIABLE_MAX_TOKENS, afford - 8)
                logger.info(
                    "codegen_client: 402 - retrying %s at max_tokens=%d",
                    model, effective_max,
                )
                resp = _post_openrouter(
                    model, messages, max_tokens=effective_max,
                )
                if resp is None:
                    return CodegenAttempt(
                        ok=False, model=model,
                        reason="openrouter_key_unset_on_retry",
                    )
                if resp.get("__error__"):
                    err = resp["__error__"]
                    hs = resp.get("http_status")
                    snip = (
                        resp.get("body_snippet")
                        or resp.get("exception_msg")
                        or ""
                    )
                    reason = f"codegen_{err}"
                    if hs:
                        reason += f"_status_{hs}"
                    if snip:
                        reason += f": {snip[:200]}"
                    return CodegenAttempt(
                        ok=False, model=model, reason=reason,
                    )
                # Fall through into the normal success path with the
                # retry response.
            else:
                # 402 with no parseable token count = quota-of-a-
                # different-flavour. Report cleanly.
                return CodegenAttempt(
                    ok=False, model=model,
                    reason=(
                        f"credit_exhausted: openrouter returned 402 "
                        f"but token affordance not parseable. Snippet: "
                        f"{snip[:200]}"
                    ),
                )
        else:
            # Non-402 error path -> report as PR VV did.
            reason = f"codegen_{err}"
            if hs:
                reason += f"_status_{hs}"
            if snip:
                reason += f": {snip[:200]}"
            return CodegenAttempt(ok=False, model=model, reason=reason)
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
    # PR WW: skip the fallback model when the primary failed because
    # of credit exhaustion. MODEL_FALLBACK is more expensive per token
    # than MODEL_PRIMARY, so retrying there guarantees another 402 and
    # wastes wall-clock time + writes a second failed row for the same
    # underlying reason.
    reason_lc = (first.reason or "").lower()
    if "credit_exhausted" in reason_lc or "status_402" in reason_lc:
        logger.info(
            "codegen_client: primary hit credit_exhausted - skipping "
            "more-expensive MODEL_FALLBACK",
        )
        return [first]
    second = request_code(
        concept, insight, capability_type, happy_kwargs,
        model=MODEL_FALLBACK,
        retry_reasons=primary_reasons or [first.reason or "primary_failed"],
    )
    return [first, second]


# PR WW: pre-flight credit check so the materialiser can skip a tick
# cleanly when the OpenRouter balance is below the minimum viable
# request size, rather than burning a picked candidate to discover it.
def get_openrouter_credits() -> Optional[dict]:
    """Query OpenRouter's /api/v1/credits endpoint. Returns a dict
    with 'usage', 'limit', and 'balance' (in USD), or None on failure.

    Returns None if OPENROUTER_API_KEY is unset. Any HTTP or network
    failure also returns None so the caller can safely continue with
    the old (attempt-then-fail-back) path.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    req = urllib.request.Request(
        OPENROUTER_CREDITS_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
        data = json.loads(body)
        d = data.get("data") if isinstance(data, dict) else None
        if isinstance(d, dict):
            usage   = float(d.get("total_usage", 0.0) or 0.0)
            limit   = d.get("total_credits", None)
            balance = None
            if isinstance(limit, (int, float)):
                balance = float(limit) - usage
            return {"usage": usage, "limit": limit, "balance": balance}
        return None
    except Exception as e:
        logger.info("codegen_client: credit check failed: %s", e)
        return None


__all__ = [
    "MODEL_PRIMARY",
    "MODEL_FALLBACK",
    "CodegenAttempt",
    "request_code",
    "request_code_cascade",
    "get_openrouter_credits",
    "MIN_VIABLE_MAX_TOKENS",
]
