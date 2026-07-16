"""
KaizenExecutor — reads kaizen_proposals.jsonl, for HIGH-priority proposals
generates a code fix via LLM, creates a branch, commits it, opens a PR.
Human review required before merge.
"""

import os
import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("KaizenExecutor")

KAIZEN_QUEUE_PATH = os.getenv("KAIZEN_QUEUE_PATH", "data/kaizen_proposals.jsonl")
KAIZEN_PROCESSED_PATH = os.getenv("KAIZEN_PROCESSED_PATH", "data/kaizen_processed.json")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN_MAIN") or os.getenv("GITHUB_TOKEN_SECONDARY") or ""
GITHUB_REPO = os.getenv("GITHUB_REPO", "Davemiles1978/dmai-system")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
POLL_INTERVAL_SECONDS = int(os.getenv("KAIZEN_POLL_SECONDS", "3600"))  # 1 hour

GH_API = "https://api.github.com"
GH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _gh(method: str, path: str, **kwargs):
    url = f"{GH_API}{path}"
    r = requests.request(method, url, headers=GH_HEADERS, timeout=30, **kwargs)
    if r.status_code >= 400:
        logger.error("GitHub %s %s → %s: %s", method, path, r.status_code, r.text[:300])
        r.raise_for_status()
    return r.json() if r.text else {}


def _load_processed() -> set:
    p = Path(KAIZEN_PROCESSED_PATH)
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            pass
    return set()


def _save_processed(ids: set):
    Path(KAIZEN_PROCESSED_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(KAIZEN_PROCESSED_PATH).write_text(json.dumps(list(ids)))


def _load_queue() -> list:
    p = Path(KAIZEN_QUEUE_PATH)
    if not p.exists():
        return []
    proposals = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            proposals.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return proposals


def _generate_fix(proposal: dict) -> Optional[str]:
    """
    Ask the LLM (Groq fast model) to produce a code patch for the proposal.
    Returns the patch text or None on failure.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — cannot generate fix")
        return None

    description = proposal.get("description", "")
    component = proposal.get("component", "unknown")
    error_msg = proposal.get("error_message", "")
    suggested_fix = proposal.get("suggested_fix", "")

    prompt = f"""You are a Python developer working on the DMAI autonomous AI system.
A self-improvement proposal has been raised. Your job is to generate the minimal, correct code change.

Component: {component}
Description: {description}
Error message: {error_msg}
Suggested fix: {suggested_fix}

Respond ONLY with a unified diff patch (--- a/... +++ b/... format) that applies the fix.
If you cannot generate a safe, targeted patch, respond with: CANNOT_FIX: <reason>
Do not include any explanation outside the patch."""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
            },
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("CANNOT_FIX:"):
            logger.info("LLM declined fix: %s", content)
            return None
        return content
    except Exception as e:
        logger.error("LLM fix generation failed: %s", e)
        return None


def _get_main_sha() -> str:
    data = _gh("GET", f"/repos/{GITHUB_REPO}/git/ref/heads/main")
    return data["object"]["sha"]


def _create_branch(branch_name: str, sha: str):
    _gh("POST", f"/repos/{GITHUB_REPO}/git/refs", json={
        "ref": f"refs/heads/{branch_name}",
        "sha": sha,
    })


def _commit_patch(branch: str, patch_text: str, proposal: dict) -> bool:
    """
    Writes the patch as a file under data/kaizen_patches/ and commits it.
    (Applying a unified diff programmatically over the API is complex;
    instead we commit the patch file itself for a human to review and apply.)
    """
    import base64
    proposal_id = proposal.get("id", hashlib.sha1(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:8])
    file_path = f"data/kaizen_patches/patch_{proposal_id}.diff"
    content_b64 = base64.b64encode(patch_text.encode()).decode()

    # Check if file already exists (get its sha for update)
    sha_param = {}
    try:
        existing = _gh("GET", f"/repos/{GITHUB_REPO}/contents/{file_path}", params={"ref": branch})
        sha_param = {"sha": existing["sha"]}
    except Exception:
        pass  # new file

    _gh("PUT", f"/repos/{GITHUB_REPO}/contents/{file_path}", json={
        "message": f"kaizen: add patch for proposal {proposal_id}",
        "content": content_b64,
        "branch": branch,
        **sha_param,
    })
    return True


def _open_pr(branch: str, proposal: dict, proposal_id: str) -> str:
    description = proposal.get("description", "No description")
    component = proposal.get("component", "unknown")
    priority = proposal.get("priority", "HIGH")
    error_msg = proposal.get("error_message", "")

    body = f"""## Kaizen Self-Improvement Proposal

**Proposal ID:** `{proposal_id}`
**Component:** `{component}`
**Priority:** {priority}
**Raised:** {proposal.get("timestamp", "unknown")}

### Description
{description}

### Error Message
```
{error_msg}
```

### Suggested Fix
{proposal.get("suggested_fix", "See patch file in data/kaizen_patches/")}

---

### Review Checklist
- [ ] Patch applies cleanly
- [ ] Logic is correct and safe
- [ ] No regressions introduced
- [ ] Tested locally or in staging

> ⚠️ **Human review required before merge.** This PR was auto-generated by KaizenExecutor.
"""

    # Ensure labels exist
    for label in ["auto-generated", "kaizen"]:
        try:
            _gh("POST", f"/repos/{GITHUB_REPO}/labels", json={
                "name": label,
                "color": "0075ca" if label == "auto-generated" else "e4e669",
            })
        except Exception:
            pass  # already exists

    pr = _gh("POST", f"/repos/{GITHUB_REPO}/pulls", json={
        "title": f"[Kaizen] {description[:80]}",
        "head": branch,
        "base": "main",
        "body": body,
        "draft": False,
    })
    pr_url = pr.get("html_url", "")

    # Add labels
    pr_number = pr.get("number")
    if pr_number:
        try:
            _gh("POST", f"/repos/{GITHUB_REPO}/issues/{pr_number}/labels",
                json={"labels": ["auto-generated", "kaizen"]})
        except Exception:
            pass

    return pr_url


def _process_proposal(proposal: dict) -> bool:
    """Returns True if a PR was successfully opened."""
    proposal_id = proposal.get(
        "id",
        hashlib.sha1(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:8]
    )
    priority = proposal.get("priority", "MEDIUM").upper()
    if priority not in ("HIGH", "MEDIUM"):
        logger.info("Skipping low-priority proposal %s", proposal_id)
        return False

    logger.info("Processing Kaizen proposal %s (priority=%s)", proposal_id, priority)

    patch = _generate_fix(proposal)
    if not patch:
        logger.warning("No patch generated for proposal %s", proposal_id)
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    branch = f"auto/kaizen-{today}-{proposal_id[:6]}"

    try:
        main_sha = _get_main_sha()
        _create_branch(branch, main_sha)
        _commit_patch(branch, patch, proposal)
        pr_url = _open_pr(branch, proposal, proposal_id)
        logger.info("PR opened: %s", pr_url)
        return True
    except Exception as e:
        logger.error("Failed to create PR for proposal %s: %s", proposal_id, e)
        return False


def run_once():
    """Run one pass of the Kaizen executor."""
    proposals = _load_queue()
    if not proposals:
        logger.info("Kaizen queue empty — nothing to do")
        return

    processed = _load_processed()
    new_prs = []

    for proposal in proposals:
        proposal_id = proposal.get(
            "id",
            hashlib.sha1(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:8]
        )
        if proposal_id in processed:
            continue

        success = _process_proposal(proposal)
        processed.add(proposal_id)

        if success:
            new_prs.append(proposal_id)

    _save_processed(processed)
    logger.info("Kaizen run complete. New PRs opened: %d", len(new_prs))


# PR QQ: cooperative-stop machinery so the daemon loop can be shut down
# cleanly on process exit or explicit stop() (instead of a bare while-True).
_KAIZEN_STOP_EVENT = threading.Event()
_KAIZEN_THREAD: "threading.Thread | None" = None


def stop_background_loop(join_timeout: float = 5.0) -> None:
    """Signal the Kaizen background loop to stop and (optionally) join."""
    global _KAIZEN_THREAD
    _KAIZEN_STOP_EVENT.set()
    if _KAIZEN_THREAD is not None and _KAIZEN_THREAD.is_alive():
        _KAIZEN_THREAD.join(timeout=join_timeout)
    _KAIZEN_THREAD = None


def start_background_loop():
    """Start KaizenExecutor as a daemon thread (idempotent, stoppable)."""
    global _KAIZEN_THREAD
    if _KAIZEN_THREAD is not None and _KAIZEN_THREAD.is_alive():
        logger.info("KaizenExecutor loop already running; skip duplicate start")
        return _KAIZEN_THREAD
    _KAIZEN_STOP_EVENT.clear()

    def loop():
        logger.info("KaizenExecutor background loop started (interval=%ds)", POLL_INTERVAL_SECONDS)
        while not _KAIZEN_STOP_EVENT.is_set():
            try:
                if GITHUB_TOKEN:
                    run_once()
                else:
                    logger.warning("GITHUB_TOKEN not set — KaizenExecutor disabled")
            except Exception as e:
                logger.error("KaizenExecutor loop error: %s", e)
            # Interruptible sleep: returns True if stop was requested.
            if _KAIZEN_STOP_EVENT.wait(POLL_INTERVAL_SECONDS):
                break
        logger.info("KaizenExecutor background loop stopped cleanly")

    t = threading.Thread(target=loop, daemon=True, name="KaizenExecutor")
    t.start()
    _KAIZEN_THREAD = t
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_once()
