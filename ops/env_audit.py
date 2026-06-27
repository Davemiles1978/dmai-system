#!/usr/bin/env python3
"""
DMAI Environment Audit Trail
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pulls the current Render environment variable configuration (values redacted),
diffs against the last snapshot, flags unapproved changes, commits a new
version-controlled snapshot to GitHub, and generates a recovery plan.

Usage:
    python3 ops/env_audit.py [--approve] [--recover] [--dry-run]

    --approve     Mark all current changes as approved (updates baseline)
    --recover     Print the full recovery checklist to stdout
    --dry-run     Run everything but skip the GitHub commit

Environment variables required:
    RENDER_API_KEY          Render API key  (https://dashboard.render.com/u/settings#api-keys)
    GITHUB_TOKEN_MAIN       GitHub PAT with repo scope
    GITHUB_REPO             Target repo  (default: Davemiles1978/dmai-system)
    RENDER_SERVICE_NAME     Render service name (default: dmai-web)
"""

import os
import sys
import json
import hashlib
import hmac
import base64
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_REPO         = os.environ.get("GITHUB_REPO", "Davemiles1978/dmai-system")
RENDER_SERVICE_NAME = os.environ.get("RENDER_SERVICE_NAME", "dmai-web")
SNAPSHOT_DIR        = "ops/snapshots"          # path inside the repo
BASELINE_FILE       = "ops/snapshots/baseline.json"
RECOVERY_TEMPLATE   = "ops/RECOVERY_PLAN.md"

# Variables that should ALWAYS be present (critical set from render.yaml)
REQUIRED_VARS = [
    "MASTER_PASSWORD", "JWT_SECRET", "PORT", "RENDER", "LOG_LEVEL", "DATA_PATH",
    "GROQ_API_KEY", "GOOGLE_AI_STUDIO_KEY", "CEREBRAS_API_KEY", "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY", "OPENROUTER_API_KEY", "GITHUB_TOKEN_MAIN",
]

# Variables whose presence is optional but tracked
OPTIONAL_VARS = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "PERPLEXITY_API_KEY",
    "HUGGINGFACE_API_KEY", "COHERE_API_KEY", "GITHUB_MODELS_TOKEN",
    "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "XAI_API_KEY",
    "BRAVE_SEARCH_API_KEY", "DATABASE_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL",
    "ELEVENLABS_API_KEY", "STABILITY_API_KEY", "RUNWAY_API_KEY", "REPLICATE_API_KEY",
    "TOGETHER_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_ENVIRONMENT",
    "GEMINI_API_KEY", "GEMINI_GEMS_KEY", "IMAGEN_API_KEY", "NOTEBOOKLM_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
    "KDP_EMAIL", "KDP_PASSWORD", "AMAZON_SELLER_ID",
    "WEBHOOK_SECRET", "GITHUB_TOKEN_SECONDARY",
]

ALL_KNOWN_VARS = REQUIRED_VARS + OPTIONAL_VARS

# Signup URLs for recovery plan
SIGNUP_URLS = {
    "GROQ_API_KEY":          "https://console.groq.com/keys",
    "GOOGLE_AI_STUDIO_KEY":  "https://aistudio.google.com/apikey",
    "CEREBRAS_API_KEY":      "https://cloud.cerebras.ai",
    "DEEPSEEK_API_KEY":      "https://platform.deepseek.com/api_keys",
    "TAVILY_API_KEY":        "https://tavily.com/#api",
    "OPENROUTER_API_KEY":    "https://openrouter.ai/keys",
    "OPENAI_API_KEY":        "https://platform.openai.com/api-keys",
    "ANTHROPIC_API_KEY":     "https://console.anthropic.com/settings/keys",
    "MISTRAL_API_KEY":       "https://console.mistral.ai",
    "PERPLEXITY_API_KEY":    "https://docs.perplexity.ai",
    "HUGGINGFACE_API_KEY":   "https://huggingface.co/settings/tokens",
    "COHERE_API_KEY":        "https://dashboard.cohere.com/api-keys",
    "GITHUB_MODELS_TOKEN":   "https://github.com/settings/tokens",
    "GITHUB_TOKEN_MAIN":     "https://github.com/settings/tokens",
    "GITHUB_TOKEN_SECONDARY":"https://github.com/settings/tokens",
    "CLOUDFLARE_API_KEY":    "https://dash.cloudflare.com/profile/api-tokens",
    "XAI_API_KEY":           "https://console.x.ai",
    "BRAVE_SEARCH_API_KEY":  "https://api.search.brave.com",
    "DATABASE_URL":          "https://render.com/docs/databases",
    "TELEGRAM_BOT_TOKEN":    "https://t.me/BotFather",
    "ALPACA_API_KEY":        "https://alpaca.markets",
    "ALPACA_SECRET_KEY":     "https://alpaca.markets",
    "ELEVENLABS_API_KEY":    "https://elevenlabs.io",
    "STABILITY_API_KEY":     "https://platform.stability.ai/account/keys",
    "RUNWAY_API_KEY":        "https://app.runwayml.com",
    "REPLICATE_API_KEY":     "https://replicate.com/account/api-tokens",
    "TOGETHER_API_KEY":      "https://api.together.ai/settings/api-keys",
    "PINECONE_API_KEY":      "https://app.pinecone.io",
    "AWS_ACCESS_KEY_ID":     "https://console.aws.amazon.com/iam",
    "RENDER_API_KEY":        "https://dashboard.render.com/u/settings#api-keys",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _redact(value: str) -> str:
    """Return a safe fingerprint: first 2 chars + SHA-1 suffix (8 hex). Never the real value."""
    if not value:
        return "__EMPTY__"
    prefix = value[:2] if len(value) >= 2 else value[0]
    digest = hashlib.sha1(value.encode()).hexdigest()[:8]
    return f"{prefix}****[{digest}]"

def _presence(value: str) -> str:
    """SET or UNSET."""
    return "SET" if value else "UNSET"


# ─────────────────────────────────────────────────────────────────────────────
# RENDER API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_render_env(render_api_key: str, service_name: str) -> dict:
    """
    Pull current env vars from Render API.
    Returns {var_name: {"status": "SET|UNSET", "fingerprint": "...", "source": "render_api"}}
    Falls back to local os.environ inspection if API key not available.
    """
    import urllib.request, urllib.error

    env_state = {}

    if render_api_key:
        try:
            # Step 1: find the service ID
            req = urllib.request.Request(
                "https://api.render.com/v1/services?limit=20",
                headers={"Authorization": f"Bearer {render_api_key}", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                services = json.loads(resp.read())

            service_id = None
            for item in services:
                svc = item.get("service", item)
                if svc.get("name") == service_name:
                    service_id = svc.get("id")
                    break

            if not service_id:
                print(f"  ⚠  Service '{service_name}' not found via Render API — falling back to render.yaml list")
            else:
                # Step 2: fetch env vars for that service
                req2 = urllib.request.Request(
                    f"https://api.render.com/v1/services/{service_id}/env-vars?limit=100",
                    headers={"Authorization": f"Bearer {render_api_key}", "Accept": "application/json"}
                )
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    env_vars = json.loads(resp2.read())

                for item in env_vars:
                    ev = item.get("envVar", item)
                    key   = ev.get("key", "")
                    value = ev.get("value", "")
                    if key:
                        env_state[key] = {
                            "status":      _presence(value),
                            "fingerprint": _redact(value) if value else "__UNSET__",
                            "source":      "render_api",
                        }
                print(f"  ✓  Fetched {len(env_state)} env vars from Render API for service '{service_name}'")
                return env_state

        except Exception as e:
            print(f"  ⚠  Render API error: {e} — falling back to known-var list")

    # ── Fallback: use the known var list + local environment ─────────────────
    print("  ℹ  No RENDER_API_KEY set — auditing against known var list + local environment")
    for key in ALL_KNOWN_VARS:
        val = os.environ.get(key, "")
        env_state[key] = {
            "status":      _presence(val),
            "fingerprint": _redact(val) if val else "__UNSET__",
            "source":      "local_env_fallback",
        }
    return env_state


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(env_state: dict) -> dict:
    """Combine env state with metadata into a snapshot document."""
    missing_required = [k for k in REQUIRED_VARS if env_state.get(k, {}).get("status") == "UNSET"]
    set_count   = sum(1 for v in env_state.values() if v["status"] == "SET")
    unset_count = sum(1 for v in env_state.values() if v["status"] == "UNSET")

    return {
        "schema_version": "1.1",
        "snapshot_at":    _now_utc(),
        "service":        RENDER_SERVICE_NAME,
        "repo":           GITHUB_REPO,
        "summary": {
            "total_tracked": len(env_state),
            "set":           set_count,
            "unset":         unset_count,
            "missing_required": missing_required,
            "health": "CRITICAL" if missing_required else ("PARTIAL" if unset_count > 0 else "FULL"),
        },
        "variables": dict(sorted(env_state.items())),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIFF ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def diff_snapshots(baseline: dict, current: dict) -> dict:
    """
    Compare baseline vs current snapshot.
    Returns a diff report with added, removed, changed, and approved flags.
    """
    base_vars = baseline.get("variables", {})
    curr_vars = current.get("variables", {})
    approved_changes = baseline.get("approved_changes", {})

    added   = {}   # new keys not in baseline
    removed = {}   # keys present in baseline but gone
    changed = {}   # fingerprint or status changed

    all_keys = set(base_vars) | set(curr_vars)
    for key in sorted(all_keys):
        b = base_vars.get(key)
        c = curr_vars.get(key)

        if b is None and c is not None:
            added[key] = {"current": c, "approved": key in approved_changes}

        elif b is not None and c is None:
            removed[key] = {"baseline": b, "approved": key in approved_changes}

        elif b["fingerprint"] != c["fingerprint"] or b["status"] != c["status"]:
            changed[key] = {
                "baseline":  b,
                "current":   c,
                "approved":  key in approved_changes,
            }

    unapproved = {
        "added":   {k: v for k, v in added.items()   if not v["approved"]},
        "removed": {k: v for k, v in removed.items() if not v["approved"]},
        "changed": {k: v for k, v in changed.items() if not v["approved"]},
    }
    total_unapproved = sum(len(v) for v in unapproved.values())

    return {
        "baseline_snapshot_at": baseline.get("snapshot_at", "unknown"),
        "current_snapshot_at":  current["snapshot_at"],
        "added":   added,
        "removed": removed,
        "changed": changed,
        "unapproved": unapproved,
        "total_changes":    len(added) + len(removed) + len(changed),
        "total_unapproved": total_unapproved,
        "alert": total_unapproved > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY PLAN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_recovery_plan(snapshot: dict) -> str:
    """Generate a full Markdown recovery checklist from a snapshot."""
    ts   = snapshot["snapshot_at"]
    summ = snapshot["summary"]
    vars_ = snapshot["variables"]

    lines = [
        f"# DMAI Recovery Plan",
        f"> Generated from snapshot: `{ts}`  ",
        f"> Service: `{snapshot['service']}`  ",
        f"> Repo: `{snapshot['repo']}`  ",
        f"> Health: **{summ['health']}** — {summ['set']} set / {summ['unset']} unset / {summ['total_tracked']} tracked",
        "",
        "---",
        "",
        "## Step 1 — Redeploy on Render",
        "",
        "1. Go to [dashboard.render.com](https://dashboard.render.com)",
        f"2. Select service **{snapshot['service']}**",
        "3. Click **Manual Deploy → Deploy latest commit**",
        "4. Wait for build to complete (watch the logs for `DMAI v7.0.0 — Starting`)",
        "",
        "---",
        "",
        "## Step 2 — Restore Environment Variables",
        "",
        "Go to **Render → dmai-web → Environment** and set the following.",
        "Values are NOT stored here — retrieve them from your password manager or the signup links below.",
        "",
        "### 🔴 Critical (system will not function without these)",
        "",
    ]

    # Critical vars
    for key in REQUIRED_VARS:
        v = vars_.get(key, {})
        status_icon = "✅" if v.get("status") == "SET" else "❌"
        url = SIGNUP_URLS.get(key, "")
        url_str = f" — [Get key]({url})" if url else ""
        note = " ← **SET at last snapshot**" if v.get("status") == "SET" else " ← **MISSING at last snapshot**"
        lines.append(f"- [ ] `{key}` {status_icon}{note}{url_str}")

    lines += ["", "### 🟡 Secondary Providers", ""]

    optional_ai = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "PERPLEXITY_API_KEY",
        "HUGGINGFACE_API_KEY", "COHERE_API_KEY", "GITHUB_MODELS_TOKEN", "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID", "XAI_API_KEY", "BRAVE_SEARCH_API_KEY",
    ]
    for key in optional_ai:
        v = vars_.get(key, {})
        status_icon = "✅" if v.get("status") == "SET" else "⬜"
        url = SIGNUP_URLS.get(key, "")
        url_str = f" — [Get key]({url})" if url else ""
        lines.append(f"- [ ] `{key}` {status_icon}{url_str}")

    lines += ["", "### ⚪ Optional / When Ready", ""]

    optional_other = [
        "DATABASE_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL",
        "ELEVENLABS_API_KEY", "STABILITY_API_KEY", "RUNWAY_API_KEY",
        "REPLICATE_API_KEY", "TOGETHER_API_KEY",
        "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_ENVIRONMENT",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
        "GITHUB_TOKEN_SECONDARY",
    ]
    for key in optional_other:
        v = vars_.get(key, {})
        status_icon = "✅" if v.get("status") == "SET" else "⬜"
        url = SIGNUP_URLS.get(key, "")
        url_str = f" — [Get key]({url})" if url else ""
        lines.append(f"- [ ] `{key}` {status_icon}{url_str}")

    lines += [
        "",
        "---",
        "",
        "## Step 3 — Verify Deployment",
        "",
        "Run these checks after restoring env vars:",
        "",
        "```bash",
        "# System health",
        "curl https://dmai-web.onrender.com/api/status",
        "",
        "# Auth (replace YOUR_PASSWORD with your MASTER_PASSWORD value)",
        'curl -X POST https://dmai-web.onrender.com/api/admin/auth \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"password":"YOUR_PASSWORD"}\'',
        "",
        "# Heartbeat",
        "curl https://dmai-web.onrender.com/api/heartbeat",
        "",
        "# Provider health",
        "curl https://dmai-web.onrender.com/api/harvester/status",
        "```",
        "",
        "---",
        "",
        "## Step 4 — Re-run Audit After Restore",
        "",
        "```bash",
        "RENDER_API_KEY=your_key python3 ops/env_audit.py --approve",
        "```",
        "",
        "This sets the restored config as the new approved baseline.",
        "",
        "---",
        "",
        f"*Recovery plan auto-generated by `ops/env_audit.py` — snapshot `{ts}`*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB COMMIT
# ─────────────────────────────────────────────────────────────────────────────

def commit_to_github(
    snapshot: dict,
    diff: dict,
    recovery_md: str,
    dry_run: bool = False,
) -> str | None:
    """
    Commit snapshot JSON + recovery plan MD to the repo.
    Uses local git (GITHUB_TOKEN injected via credential helper).
    Returns the commit SHA or None on dry run / failure.
    """
    repo_root = Path(__file__).parent.parent
    snap_dir  = repo_root / SNAPSHOT_DIR
    snap_dir.mkdir(parents=True, exist_ok=True)

    today      = _today()
    snap_file  = snap_dir / f"snapshot_{today}.json"
    latest_sym = snap_dir / "latest.json"
    baseline_f = repo_root / BASELINE_FILE
    recovery_f = repo_root / RECOVERY_TEMPLATE

    # Write snapshot
    with open(snap_file, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  ✓  Snapshot written → {snap_file.relative_to(repo_root)}")

    # Write/overwrite latest.json
    with open(latest_sym, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Write recovery plan
    with open(recovery_f, "w") as f:
        f.write(recovery_md)
    print(f"  ✓  Recovery plan written → {recovery_f.relative_to(repo_root)}")

    # Write diff report
    diff_file = snap_dir / f"diff_{today}.json"
    with open(diff_file, "w") as f:
        json.dump(diff, f, indent=2)

    if dry_run:
        print("  ℹ  DRY RUN — skipping git commit")
        return None

    try:
        git = lambda *args: subprocess.run(
            ["git", "-C", str(repo_root)] + list(args),
            capture_output=True, text=True, check=True
        )
        git("config", "user.email", "milesd040@gmail.com")
        git("config", "user.name",  "David Miles")
        git("add",
            str(snap_file.relative_to(repo_root)),
            str(latest_sym.relative_to(repo_root)),
            str(recovery_f.relative_to(repo_root)),
            str(diff_file.relative_to(repo_root)),
        )

        alert_flag = "⚠ UNAPPROVED CHANGES" if diff.get("alert") else "✓ clean"
        msg = (
            f"[audit] Config snapshot {today} — "
            f"{diff['total_changes']} change(s), {diff['total_unapproved']} unapproved [{alert_flag}]"
        )
        git("commit", "-m", msg)
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        sha = result.stdout.strip()[:10]
        git("push", "origin", "main")
        print(f"  ✓  Committed and pushed → {sha}")
        return sha
    except subprocess.CalledProcessError as e:
        print(f"  ✗  Git error: {e.stderr.strip()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(snapshot: dict, diff: dict) -> None:
    summ = snapshot["summary"]
    sep  = "─" * 60

    print(f"\n{sep}")
    print(f"  DMAI Config Audit  ·  {snapshot['snapshot_at']}")
    print(sep)
    print(f"  Service  : {snapshot['service']}")
    print(f"  Health   : {summ['health']}  ({summ['set']} set / {summ['unset']} unset)")

    if summ["missing_required"]:
        print(f"\n  ❌  MISSING REQUIRED VARS ({len(summ['missing_required'])}):")
        for k in summ["missing_required"]:
            url = SIGNUP_URLS.get(k, "")
            print(f"       {k:<35} {url}")

    print(f"\n  Changes vs baseline snapshot ({diff['baseline_snapshot_at']}):")
    print(f"    Added   : {len(diff['added'])}")
    print(f"    Removed : {len(diff['removed'])}")
    print(f"    Changed : {len(diff['changed'])}")

    if diff["total_unapproved"] == 0:
        print(f"\n  ✅  No unapproved changes — config matches approved baseline")
    else:
        print(f"\n  ⚠   UNAPPROVED CHANGES ({diff['total_unapproved']}):")
        for section, items in diff["unapproved"].items():
            if items:
                print(f"\n    [{section.upper()}]")
                for key, detail in items.items():
                    if section == "added":
                        print(f"      + {key} ({detail['current']['status']})")
                    elif section == "removed":
                        print(f"      - {key} (was {detail['baseline']['status']})")
                    else:
                        b = detail["baseline"]
                        c = detail["current"]
                        print(f"      ~ {key}  {b['status']} → {c['status']}  fp: {b['fingerprint']} → {c['fingerprint']}")
        print(f"\n  Run with --approve to mark these as the new baseline.")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# APPROVE (update baseline)
# ─────────────────────────────────────────────────────────────────────────────

def approve_baseline(snapshot: dict, repo_root: Path, dry_run: bool) -> None:
    """Write current snapshot as the new approved baseline."""
    baseline_f = repo_root / BASELINE_FILE
    baseline_f.parent.mkdir(parents=True, exist_ok=True)

    approved_snapshot = dict(snapshot)
    approved_snapshot["approved_at"] = _now_utc()
    approved_snapshot["approved_changes"] = {
        k: _now_utc() for k in snapshot["variables"]
        if snapshot["variables"][k]["status"] == "SET"
    }

    if not dry_run:
        with open(baseline_f, "w") as f:
            json.dump(approved_snapshot, f, indent=2)
        print(f"  ✓  Baseline updated → {BASELINE_FILE}")
    else:
        print("  ℹ  DRY RUN — baseline not written")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DMAI Environment Audit Trail")
    parser.add_argument("--approve",  action="store_true", help="Approve current state as new baseline")
    parser.add_argument("--recover",  action="store_true", help="Print recovery plan to stdout")
    parser.add_argument("--dry-run",  action="store_true", help="Skip git commit")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    baseline_f = repo_root / BASELINE_FILE

    render_api_key = os.environ.get("RENDER_API_KEY", "")

    print("\n🔍  DMAI Environment Audit — starting")
    print(f"    Service  : {RENDER_SERVICE_NAME}")
    print(f"    Repo     : {GITHUB_REPO}")
    print(f"    Render API key: {'present' if render_api_key else 'NOT SET (fallback mode)'}\n")

    # ── 1. Fetch current env state ───────────────────────────────────────────
    print("[ 1/5 ] Fetching environment state from Render...")
    env_state = fetch_render_env(render_api_key, RENDER_SERVICE_NAME)

    # ── 2. Build snapshot ────────────────────────────────────────────────────
    print("[ 2/5 ] Building snapshot...")
    current_snapshot = build_snapshot(env_state)

    # ── 3. Load baseline and diff ────────────────────────────────────────────
    print("[ 3/5 ] Comparing against baseline...")
    if baseline_f.exists():
        with open(baseline_f) as f:
            baseline = json.load(f)
        print(f"        Baseline from: {baseline.get('snapshot_at', 'unknown')}")
    else:
        print("        No baseline found — treating current state as first snapshot")
        baseline = dict(current_snapshot)
        baseline["snapshot_at"] = "INITIAL"
        baseline["variables"]   = {}  # empty so everything shows as "added"

    diff = diff_snapshots(baseline, current_snapshot)

    # ── 4. Generate recovery plan ────────────────────────────────────────────
    print("[ 4/5 ] Generating recovery plan...")
    recovery_md = generate_recovery_plan(current_snapshot)

    if args.recover:
        print("\n" + "═" * 60)
        print(recovery_md)
        print("═" * 60 + "\n")

    # ── 5. Commit to GitHub ──────────────────────────────────────────────────
    print("[ 5/5 ] Committing snapshot to GitHub...")
    sha = commit_to_github(current_snapshot, diff, recovery_md, dry_run=args.dry_run)

    # ── Approve baseline if requested ────────────────────────────────────────
    if args.approve:
        print("\n[approve] Writing new approved baseline...")
        approve_baseline(current_snapshot, repo_root, args.dry_run)
        if not args.dry_run:
            # Commit the baseline update too
            try:
                git = lambda *args: subprocess.run(
                    ["git", "-C", str(repo_root)] + list(args),
                    capture_output=True, text=True, check=True
                )
                git("add", BASELINE_FILE)
                git("commit", "-m", f"[audit] Approve baseline — {_today()}")
                git("push", "origin", "main")
                print("  ✓  Baseline approval committed")
            except subprocess.CalledProcessError as e:
                print(f"  ✗  Baseline commit failed: {e.stderr.strip()}")

    # ── Print report ─────────────────────────────────────────────────────────
    print_report(current_snapshot, diff)

    # ── Exit code: non-zero if unapproved changes exist ──────────────────────
    if diff["total_unapproved"] > 0 and not args.approve:
        sys.exit(1)


if __name__ == "__main__":
    main()
