#!/usr/bin/env python3
"""PR DDD-2: Weekly promoter-drift diagnostic.

Runs inside the GitHub Actions runner (see ../promoter-drift.yml).
Collects DMAI admin status, computes diversity metrics, decides
whether to alert, composes an HTML + text email, and POSTs it to
/api/cron/promoter-drift/email on dmai-web where DMAI delivers it
via Resend or Slack fallback.

Trigger rule (matches the original Perplexity cron):
  * If diversity ratio dropped >10% vs 28-day baseline -> send email.
  * Otherwise -> log a one-line "ok" entry to workspace/promoter_drift_log.md.

Stdlib only. No third-party deps needed on the runner.

Exit codes:
  0 = ran successfully (regardless of whether alert triggered)
  2 = failed to collect enough data to compute the ratio
  3 = failed to POST to the email endpoint AND diversity was
      degraded enough to warrant alerting - GitHub Actions surfaces
      this as a red run so we notice the delivery failure.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import request as _urlreq
from urllib import error as _urlerr


BASE_URL = os.environ["DMAI_BASE_URL"].rstrip("/")
CRON_SECRET = os.environ["DMAI_CRON_SECRET"]
LOG_PATH = pathlib.Path(os.environ.get("DRIFT_LOG_PATH", "workspace/promoter_drift_log.md"))
EMAIL_TO = os.environ.get("DRIFT_EMAIL_TO", "milesd040@gmail.com")
ALERT_THRESHOLD_PCT = -10.0  # trigger if delta <= -10%


def get_json(path: str, headers: Optional[Dict[str, str]] = None,
             timeout: int = 30) -> Optional[Dict[str, Any]]:
    """GET a DMAI admin endpoint and return parsed JSON, or None on failure."""
    url = f"{BASE_URL}{path}"
    req = _urlreq.Request(url, headers=headers or {})
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except (_urlerr.HTTPError, _urlerr.URLError, json.JSONDecodeError) as e:
        print(f"[warn] {path}: {e}", file=sys.stderr)
        return None


def shannon_diversity(counts: Dict[str, int]) -> float:
    """Return H / log(k) or 0.0 if not enough types."""
    total = sum(counts.values())
    k = len([c for c in counts.values() if c > 0])
    if total == 0 or k <= 1:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            H -= p * math.log(p)
    return H / math.log(k)


def load_previous_snapshot() -> Optional[Dict[str, Any]]:
    """Pull the most recent snapshot JSON from the log file, or None."""
    if not LOG_PATH.exists():
        return None
    text = LOG_PATH.read_text(encoding="utf-8")
    # Snapshots are appended as fenced JSON blocks: ```json ... ```
    start = text.rfind("```json")
    if start == -1:
        return None
    end = text.find("```", start + 7)
    if end == -1:
        return None
    try:
        return json.loads(text[start + 7:end].strip())
    except json.JSONDecodeError:
        return None


def collect_metrics() -> Dict[str, Any]:
    """Pull the four admin endpoints in parallel-ish (serial is fine at
    this scale, ~1s each). Fail-soft: missing fields become empty."""
    return {
        "fresh_blood": get_json("/api/admin/fresh-blood-status") or {},
        "capability_promoter": get_json("/api/admin/capability-promoter-status") or {},
        "insight_promoter": get_json("/api/admin/insight-promoter-status") or {},
        "materialiser": get_json("/api/admin/capability-materialiser-status") or {},
        "gaps": get_json("/api/self-evolution/gaps") or {},
    }


def compute_ratio_and_delta(metrics: Dict[str, Any],
                            previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return dict with current_ratio, prev_ratio, delta_pct, distribution."""
    # Capability distribution comes from capability_promoter status.
    dist_raw = metrics["capability_promoter"].get("distribution", {})
    distribution: Dict[str, int] = {k: int(v or 0) for k, v in dist_raw.items()}
    current_ratio = shannon_diversity(distribution)
    prev_ratio = float(previous.get("current_ratio", 0.0)) if previous else 0.0
    if prev_ratio > 0:
        delta_pct = (current_ratio - prev_ratio) / prev_ratio * 100.0
    else:
        delta_pct = 0.0  # first run - no baseline yet
    return {
        "current_ratio": round(current_ratio, 4),
        "prev_ratio": round(prev_ratio, 4),
        "delta_pct": round(delta_pct, 2),
        "distribution": distribution,
        "first_run": previous is None,
    }


def stagnant_types(distribution: Dict[str, int],
                   previous: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of {name, current, previous, pct_of_prev} for types
    whose count fell below 25% of their previous value."""
    prev_dist = (previous or {}).get("distribution", {}) or {}
    stagnant = []
    for k, current in distribution.items():
        prev = int(prev_dist.get(k, 0))
        if prev >= 4 and current < prev * 0.25:  # only flag ones that had signal
            stagnant.append({
                "name": k,
                "current": current,
                "previous": prev,
                "pct_of_prev": round(current / prev * 100.0, 1),
            })
    stagnant.sort(key=lambda x: x["pct_of_prev"])
    return stagnant[:5]


def build_email_bodies(ratio_info: Dict[str, Any],
                       stagnant: List[Dict[str, Any]],
                       metrics: Dict[str, Any]) -> Dict[str, str]:
    """Return {subject, html_body, text_body, slack_summary}."""
    d = ratio_info["delta_pct"]
    cur = ratio_info["current_ratio"]
    prev = ratio_info["prev_ratio"]
    fb_count = metrics["fresh_blood"].get("last_7d_count", "?")
    top_stag = ", ".join(s["name"] for s in stagnant[:2]) or "(none)"

    subject = (
        f"DMAI Promoter Drift Alert - diversity {d:+.1f}% WoW · "
        f"stagnant: {top_stag}"
    )

    lines = [
        f"Diversity ratio: {cur:.4f} (was {prev:.4f}, delta {d:+.2f}%).",
        f"Fresh-blood insights last 7d: {fb_count}.",
        "",
        "Stagnant capability types (this week vs last week):",
    ]
    for s in stagnant:
        lines.append(f"  * {s['name']}: {s['current']} (was {s['previous']}, "
                     f"{s['pct_of_prev']}% of prev)")
    if not stagnant:
        lines.append("  * (none)")
    lines += [
        "",
        f"Full snapshot: {BASE_URL}/api/admin/capability-promoter-status",
        f"Run at: {datetime.now(timezone.utc).isoformat()}",
    ]
    text_body = "\n".join(lines)
    html_body = (
        "<html><body>"
        f"<h2>DMAI Promoter Drift Alert</h2>"
        f"<p><b>Diversity ratio:</b> {cur:.4f} (was {prev:.4f}, "
        f"<b>delta {d:+.2f}%</b>).</p>"
        f"<p><b>Fresh-blood insights last 7d:</b> {fb_count}</p>"
        f"<h3>Stagnant capability types</h3><ul>"
        + "".join(
            f"<li><b>{s['name']}</b>: {s['current']} "
            f"(was {s['previous']}, {s['pct_of_prev']}% of prev)</li>"
            for s in stagnant
        )
        + ("<li>(none)</li>" if not stagnant else "")
        + "</ul>"
        f"<p>Full snapshot: <a href=\"{BASE_URL}/api/admin/capability-promoter-status\">"
        f"{BASE_URL}/api/admin/capability-promoter-status</a></p>"
        f"<p><small>Run at: {datetime.now(timezone.utc).isoformat()}</small></p>"
        "</body></html>"
    )
    slack_summary = f"DMAI drift {d:+.1f}% WoW · stagnant: {top_stag}"
    return {
        "subject": subject,
        "html_body": html_body,
        "text_body": text_body,
        "slack_summary": slack_summary,
    }


def post_email(bodies: Dict[str, str], to: List[str]) -> Dict[str, Any]:
    """POST the composed bodies to /api/cron/promoter-drift/email."""
    url = f"{BASE_URL}/api/cron/promoter-drift/email"
    payload = dict(bodies)
    payload["to"] = to
    req = _urlreq.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-Cron-Secret": CRON_SECRET,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return {"ok": True, "status": resp.status, "body": json.loads(body)}
    except _urlerr.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode("utf-8", "replace")[:400]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": 0, "error": str(e)[:400]}


def append_log(ratio_info: Dict[str, Any], stagnant: List[Dict[str, Any]],
               triggered: bool, delivery: Optional[Dict[str, Any]]) -> None:
    """Append a one-line entry + snapshot JSON block to the log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    header = (
        f"\n## {now} · delta {ratio_info['delta_pct']:+.2f}% · "
        f"triggered={triggered}"
    )
    if triggered and delivery:
        header += f" · via={delivery.get('body', {}).get('delivered_via', '?')}"
    snapshot = {
        "run_ts": now,
        "current_ratio": ratio_info["current_ratio"],
        "prev_ratio": ratio_info["prev_ratio"],
        "delta_pct": ratio_info["delta_pct"],
        "triggered": triggered,
        "distribution": ratio_info["distribution"],
        "stagnant": [s["name"] for s in stagnant],
    }
    block = f"{header}\n```json\n{json.dumps(snapshot, indent=2)}\n```\n"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(block)


def main() -> int:
    print(f"[info] Collecting metrics from {BASE_URL} ...")
    metrics = collect_metrics()
    previous = load_previous_snapshot()

    dist_raw = metrics["capability_promoter"].get("distribution")
    if not dist_raw:
        print("[error] capability distribution missing; cannot compute ratio",
              file=sys.stderr)
        return 2

    ratio_info = compute_ratio_and_delta(metrics, previous)
    stagnant = stagnant_types(ratio_info["distribution"], previous)

    triggered = (
        not ratio_info["first_run"]
        and ratio_info["delta_pct"] <= ALERT_THRESHOLD_PCT
    )
    print(
        f"[info] ratio={ratio_info['current_ratio']} "
        f"prev={ratio_info['prev_ratio']} "
        f"delta={ratio_info['delta_pct']:+.2f}% "
        f"first_run={ratio_info['first_run']} triggered={triggered}"
    )

    delivery: Optional[Dict[str, Any]] = None
    if triggered:
        bodies = build_email_bodies(ratio_info, stagnant, metrics)
        delivery = post_email(bodies, [EMAIL_TO])
        print(f"[info] delivery result: {json.dumps(delivery)[:400]}")
    else:
        print("[info] No alert triggered - logging only.")

    append_log(ratio_info, stagnant, triggered, delivery)

    if triggered and (not delivery or not delivery.get("ok")):
        # Alert needed but delivery failed; exit non-zero so GitHub
        # highlights the run.
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
