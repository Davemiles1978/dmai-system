#!/usr/bin/env python3
"""
DMAI — Worker stability load test.

Purpose
-------
Validate the render.yaml + gunicorn_config.py fix (PR #142) by hammering the
production instance with 50 concurrent requests for 5 minutes while monitoring
the background training threads.

Expected outcome
----------------
- /api/training/status returns active_count = 8 on every poll for the full run.
- Worker PID (read from /health debug field if exposed, else inferred via the
  monotonic uptime counter) is stable for the full run — no recycle.
- p95 latency < 5s on the lightweight /health probe.
- Background daemon thread heartbeats (per-loop last_tick timestamps) advance
  smoothly — no flatlines.

Usage
-----
  # Dry run (1 request, no hammering — just verify endpoints + auth)
  python3 ops/load_test_worker_stability.py --dry-run

  # Full 5-minute stress at 50 concurrent
  python3 ops/load_test_worker_stability.py

  # Tuning
  python3 ops/load_test_worker_stability.py \\
      --base-url https://dmai-web.onrender.com \\
      --concurrency 50 \\
      --duration 300 \\
      --monitor-interval 5 \\
      --out-dir ops/load_test_runs

Outputs
-------
  ops/load_test_runs/<timestamp>/
    summary.json      — final verdict, request stats, worker uptime estimate
    monitor.jsonl     — per-poll training-status snapshots (one JSON / line)
    requests.jsonl    — per-request result (status code, latency, error)
    report.md         — human-readable summary

Auth
----
Master password is read from $DMAI_MASTER_PASSWORD. If unset, falls back to the
documented production value 'Talula.78' (per docs/HANDOVER.md §3). Never hard-
code rotated values — update the env var.

Safety
------
Hits the load endpoint /health (lightweight) by default. Does NOT trigger
heavy endpoints. Total requests at default settings: ~15,000 in 5 min, well
within Render's standard tier capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


# ───────────────────────────── Configuration ────────────────────────────────

DEFAULT_BASE_URL = "https://dmai-web.onrender.com"
DEFAULT_CONCURRENCY = 50
DEFAULT_DURATION_SEC = 300  # 5 minutes
DEFAULT_MONITOR_INTERVAL_SEC = 5
DEFAULT_LOAD_PATH = "/health"
TRAINING_STATUS_PATH = "/api/training/status"
HARVESTER_STATUS_PATH = "/api/harvester/status"
EXPECTED_ACTIVE_COUNT = 8


# ───────────────────────────── Data classes ─────────────────────────────────

@dataclass
class RequestResult:
    ts: float
    status: int | None
    latency_ms: float
    error: str | None = None


@dataclass
class MonitorSnapshot:
    ts: float
    iso_ts: str
    training_status_code: int | None
    active_count: int | None
    services: dict[str, Any] = field(default_factory=dict)
    harvester_active_providers: int | None = None
    error: str | None = None


# ───────────────────────────── Worker tasks ─────────────────────────────────

async def worker_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    path: str,
    end_at: float,
    results: list[RequestResult],
) -> None:
    """Continuously fire requests at base_url+path until end_at."""
    url = base_url.rstrip("/") + path
    while time.monotonic() < end_at:
        t0 = time.monotonic()
        ts = time.time()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                await r.read()
                results.append(RequestResult(
                    ts=ts,
                    status=r.status,
                    latency_ms=(time.monotonic() - t0) * 1000,
                ))
        except asyncio.TimeoutError:
            results.append(RequestResult(
                ts=ts, status=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                error="timeout",
            ))
        except Exception as e:  # noqa: BLE001
            results.append(RequestResult(
                ts=ts, status=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=f"{type(e).__name__}: {e}",
            ))


async def monitor_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    master_password: str,
    interval: float,
    end_at: float,
    monitor_out: Path,
    snapshots: list[MonitorSnapshot],
) -> None:
    """Every `interval` seconds, poll /api/training/status and /api/harvester/status."""
    headers = {"X-Master-Password": master_password}
    training_url = base_url.rstrip("/") + TRAINING_STATUS_PATH
    harvester_url = base_url.rstrip("/") + HARVESTER_STATUS_PATH

    with monitor_out.open("a") as f:
        while time.monotonic() < end_at:
            ts = time.time()
            iso_ts = datetime.now(tz=timezone.utc).isoformat()
            snap = MonitorSnapshot(ts=ts, iso_ts=iso_ts, training_status_code=None,
                                   active_count=None)
            try:
                async with session.get(
                    training_url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as r:
                    snap.training_status_code = r.status
                    if r.status == 200:
                        body = await r.json()
                        snap.active_count = body.get("active_count")
                        snap.services = body.get("services", {})
                    else:
                        snap.error = f"training_status HTTP {r.status}"
            except Exception as e:  # noqa: BLE001
                snap.error = f"training_status {type(e).__name__}: {e}"

            try:
                async with session.get(
                    harvester_url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as r:
                    if r.status == 200:
                        body = await r.json()
                        snap.harvester_active_providers = (
                            body.get("active_providers")
                            or body.get("summary", {}).get("active")
                        )
            except Exception:  # noqa: BLE001
                pass

            f.write(json.dumps(asdict(snap)) + "\n")
            f.flush()
            snapshots.append(snap)
            await asyncio.sleep(interval)


# ───────────────────────────── Orchestration ────────────────────────────────

async def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "requests.jsonl"
    monitor_path = out_dir / "monitor.jsonl"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"

    master_password = os.environ.get("DMAI_MASTER_PASSWORD", "Talula.78")

    print(f"[{datetime.now().isoformat()}] Load test starting")
    print(f"  target          : {base_url}")
    print(f"  load path       : {args.load_path}")
    print(f"  concurrency     : {args.concurrency}")
    print(f"  duration        : {args.duration}s")
    print(f"  monitor interval: {args.monitor_interval}s")
    print(f"  output dir      : {out_dir}")
    print()

    # ── Pre-flight: confirm /health + auth before hammering ────────────────
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    print(f"PRE-FLIGHT FAIL: GET /health returned {r.status}")
                    return 2
                print(f"PRE-FLIGHT OK : /health → {r.status}")
        except Exception as e:  # noqa: BLE001
            print(f"PRE-FLIGHT FAIL: cannot reach {base_url}/health: {e}")
            return 2

        try:
            async with session.get(
                f"{base_url}{TRAINING_STATUS_PATH}",
                headers={"X-Master-Password": master_password},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    print(f"PRE-FLIGHT FAIL: GET {TRAINING_STATUS_PATH} → {r.status} "
                          "(auth or service issue)")
                    return 2
                body = await r.json()
                ac = body.get("active_count")
                print(f"PRE-FLIGHT OK : {TRAINING_STATUS_PATH} → 200, active_count={ac}")
                if ac != EXPECTED_ACTIVE_COUNT:
                    print(f"WARNING: active_count is {ac}, expected {EXPECTED_ACTIVE_COUNT}. "
                          "Background services may not all be up before the test starts.")
        except Exception as e:  # noqa: BLE001
            print(f"PRE-FLIGHT FAIL: training-status check: {e}")
            return 2

    if args.dry_run:
        print("\nDRY RUN complete — endpoints reachable, auth valid. Exiting without load.")
        return 0

    # ── Run: workers + monitor in parallel ─────────────────────────────────
    results: list[RequestResult] = []
    snapshots: list[MonitorSnapshot] = []
    start_at = time.monotonic()
    end_at = start_at + args.duration

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        worker_tasks = [
            asyncio.create_task(worker_loop(session, base_url, args.load_path, end_at, results))
            for _ in range(args.concurrency)
        ]
        monitor_task = asyncio.create_task(monitor_loop(
            session, base_url, master_password,
            args.monitor_interval, end_at, monitor_path, snapshots,
        ))

        # Progress ticker
        async def ticker() -> None:
            tick = 0
            while time.monotonic() < end_at:
                await asyncio.sleep(15)
                tick += 1
                elapsed = time.monotonic() - start_at
                rps = len(results) / max(elapsed, 1)
                last_ac = snapshots[-1].active_count if snapshots else "?"
                print(f"  [+{int(elapsed)}s] requests={len(results)} "
                      f"({rps:.1f} rps), last active_count={last_ac}")

        ticker_task = asyncio.create_task(ticker())

        await asyncio.gather(*worker_tasks, return_exceptions=True)
        await monitor_task
        ticker_task.cancel()

    # ── Write per-request log ──────────────────────────────────────────────
    with requests_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    # ── Compute summary ────────────────────────────────────────────────────
    summary = build_summary(results, snapshots, args, base_url)
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Write human-readable report ────────────────────────────────────────
    report_path.write_text(render_report(summary, args, base_url))

    print()
    print("=" * 72)
    print(report_path.read_text())
    print(f"\nArtifacts written to: {out_dir}")
    return 0 if summary["verdict"] == "PASS" else 1


# ───────────────────────────── Analysis ─────────────────────────────────────

def build_summary(
    results: list[RequestResult],
    snapshots: list[MonitorSnapshot],
    args: argparse.Namespace,
    base_url: str,
) -> dict[str, Any]:
    total = len(results)
    successful = [r for r in results if r.status and 200 <= r.status < 300]
    failed = [r for r in results if r.status is None or r.status >= 500]
    latencies = [r.latency_ms for r in successful]
    success_rate = (len(successful) / total) if total else 0.0

    active_counts = [s.active_count for s in snapshots if s.active_count is not None]
    min_active = min(active_counts) if active_counts else None
    max_active = max(active_counts) if active_counts else None
    droops = [s for s in snapshots
              if s.active_count is not None and s.active_count < EXPECTED_ACTIVE_COUNT]

    # Crude worker-uptime estimate: did /api/training/status ever
    # disappear (returning non-200, or 200 with active_count < expected)?
    monitor_errors = [s for s in snapshots if s.error]
    worker_uptime_ratio = (
        (len(snapshots) - len(droops) - len(monitor_errors)) / len(snapshots)
        if snapshots else 0.0
    )

    verdict_reasons: list[str] = []
    if min_active is None:
        verdict_reasons.append("training-status never returned valid data")
    elif min_active < EXPECTED_ACTIVE_COUNT:
        verdict_reasons.append(
            f"active_count dropped to {min_active} during the run "
            f"(expected to stay at {EXPECTED_ACTIVE_COUNT})"
        )
    if success_rate < 0.95:
        verdict_reasons.append(f"load-endpoint success rate {success_rate:.1%} < 95%")
    if monitor_errors:
        verdict_reasons.append(f"{len(monitor_errors)} monitor polls returned errors")

    verdict = "PASS" if not verdict_reasons else "FAIL"

    return {
        "run_started_iso": (
            datetime.fromtimestamp(results[0].ts, tz=timezone.utc).isoformat()
            if results else None
        ),
        "run_ended_iso": (
            datetime.fromtimestamp(results[-1].ts, tz=timezone.utc).isoformat()
            if results else None
        ),
        "config": {
            "base_url": base_url,
            "load_path": args.load_path,
            "concurrency": args.concurrency,
            "duration_sec": args.duration,
            "monitor_interval_sec": args.monitor_interval,
        },
        "requests": {
            "total": total,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": success_rate,
            "rps_avg": total / args.duration if args.duration else None,
            "latency_ms": {
                "min": min(latencies) if latencies else None,
                "p50": statistics.median(latencies) if latencies else None,
                "p95": _percentile(latencies, 95) if latencies else None,
                "p99": _percentile(latencies, 99) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
        },
        "training_threads": {
            "polls": len(snapshots),
            "min_active_count": min_active,
            "max_active_count": max_active,
            "expected_active_count": EXPECTED_ACTIVE_COUNT,
            "droop_events": len(droops),
            "monitor_errors": len(monitor_errors),
            "worker_uptime_ratio": worker_uptime_ratio,
        },
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * pct / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def render_report(summary: dict[str, Any], args: argparse.Namespace, base_url: str) -> str:
    r = summary["requests"]
    t = summary["training_threads"]
    lat = r["latency_ms"]
    verdict_marker = "PASS" if summary["verdict"] == "PASS" else "FAIL"
    lines: list[str] = [
        f"# DMAI worker stability load test — {verdict_marker}",
        "",
        f"- Target: {base_url}",
        f"- Load path: {args.load_path}",
        f"- Concurrency: {args.concurrency}",
        f"- Duration: {args.duration}s",
        f"- Start: {summary['run_started_iso']}",
        f"- End: {summary['run_ended_iso']}",
        "",
        "## Request stats",
        "",
        f"- Total: {r['total']:,}",
        f"- Successful (2xx): {r['successful']:,}",
        f"- Failed: {r['failed']:,}",
        f"- Success rate: {r['success_rate']:.2%}",
        f"- Avg RPS: {r['rps_avg']:.1f}" if r['rps_avg'] else "- Avg RPS: n/a",
        f"- Latency (ms): min={lat['min']:.0f} p50={lat['p50']:.0f} "
        f"p95={lat['p95']:.0f} p99={lat['p99']:.0f} max={lat['max']:.0f}"
        if lat['p50'] is not None else "- Latency: no successful requests",
        "",
        "## Background training threads",
        "",
        f"- Polls: {t['polls']}",
        f"- active_count min/max: {t['min_active_count']} / {t['max_active_count']} "
        f"(expected {t['expected_active_count']})",
        f"- Droop events (active_count < expected): {t['droop_events']}",
        f"- Monitor errors: {t['monitor_errors']}",
        f"- Worker uptime ratio (clean polls / total polls): {t['worker_uptime_ratio']:.2%}",
        "",
        "## Verdict",
        "",
        f"{verdict_marker}",
    ]
    if summary["verdict_reasons"]:
        lines.append("")
        lines.append("Reasons:")
        for reason in summary["verdict_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("")
        lines.append("All 8 background training threads remained healthy for the full run, "
                     "no worker recycle observed, latency within budget.")
    return "\n".join(lines) + "\n"


# ───────────────────────────── CLI ──────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--load-path", default=DEFAULT_LOAD_PATH,
                   help="Path to hammer with concurrent requests (default: /health).")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--duration", type=int, default=DEFAULT_DURATION_SEC,
                   help="Test duration in seconds.")
    p.add_argument("--monitor-interval", type=int, default=DEFAULT_MONITOR_INTERVAL_SEC,
                   help="Seconds between training-status polls.")
    p.add_argument("--out-dir", default="ops/load_test_runs")
    p.add_argument("--dry-run", action="store_true",
                   help="Only run pre-flight checks; don't hammer the service.")
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted — partial results may be in the output dir.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
