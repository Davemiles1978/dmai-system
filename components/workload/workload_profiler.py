"""Workload self-profiler for the self-hosting feasibility study (PR J).

Records what DMAI actually consumes on Render so PR K's procurement
research skill can price a home-lab box with realistic numbers instead
of vendor-marketing peak-load estimates.

Design pointers:
- Fourth isolated SQLite file (``data/dmai_workload.db``). Never touches
  knowledge / ledger / treasury DBs.
- Every sample is a snapshot of the CURRENT process at the moment
  :func:`sample_now` is called; monotonic counters (CPU seconds, disk
  I/O, net I/O) are recorded as absolute values so rollups compute the
  delta between adjacent samples.
- No per-loop attribution. Process-wide only. Per-loop breakdown can be
  a future refinement if PR K's sizing math wants it.
- Zero-start: ``workload_state:install_ts`` is stamped on first init.
  Rollups may include a partial first bucket, but historical samples
  from before the profiler shipped simply do not exist.
- Optional dependency on ``psutil``. If unavailable, :func:`sample_now`
  returns ``None`` and logs a warning; nothing else in the codebase
  needs to change.
- Optional cross-DB size tracking: the profiler samples the on-disk
  size of the four DMAI SQLite files so PR K can size storage growth.
  If any of those DBs are missing at sample time, they're logged as
  ``NULL`` for that tick.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────

DEFAULT_DB_FILENAME = "dmai_workload.db"
KNOWN_DBS = ("dmai_knowledge.db", "dmai_ledger.db",
             "dmai_treasury.db", "dmai_workload.db")


def default_workload_path() -> str:
    """Return the default path for the workload DB (respects DATA_PATH)."""
    base = os.environ.get("DATA_PATH", "data")
    return str(Path(base) / DEFAULT_DB_FILENAME)


# ── schema ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workload_samples (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TEXT NOT NULL,
    cpu_percent           REAL,
    cpu_seconds_total     REAL,
    mem_rss_mb            REAL,
    mem_peak_rss_mb       REAL,
    disk_read_mb_total    REAL,
    disk_write_mb_total   REAL,
    net_sent_mb_total     REAL,
    net_recv_mb_total     REAL,
    open_fds              INTEGER,
    thread_count          INTEGER,
    uptime_seconds        REAL,
    knowledge_db_mb       REAL,
    ledger_db_mb          REAL,
    treasury_db_mb        REAL,
    workload_db_mb        REAL
);

CREATE INDEX IF NOT EXISTS ix_workload_samples_ts
    ON workload_samples(ts);

CREATE TABLE IF NOT EXISTS workload_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ── connection helpers ────────────────────────────────────────────────────

def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_workload_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=15.0)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA busy_timeout=15000;")
    c.execute("PRAGMA foreign_keys=ON;")
    c.row_factory = sqlite3.Row
    return c


def _get_state(c: sqlite3.Connection, key: str) -> Optional[str]:
    row = c.execute(
        "SELECT value FROM workload_state WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def _set_state(c: sqlite3.Connection, key: str, value: str) -> None:
    c.execute(
        "INSERT INTO workload_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# ── init ──────────────────────────────────────────────────────────────────

def init_workload_db(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Create the schema, stamp ``install_ts`` if missing, return state."""
    with _connect(db_path) as c:
        c.executescript(_SCHEMA)
        if _get_state(c, "install_ts") is None:
            _set_state(
                c, "install_ts",
                datetime.now(timezone.utc).isoformat(),
            )
        c.commit()
        return {
            "install_ts":       _get_state(c, "install_ts"),
            "last_sample_ts":   _get_state(c, "last_sample_ts") or "",
        }


def get_install_ts(db_path: Optional[str] = None) -> str:
    with _connect(db_path) as c:
        return _get_state(c, "install_ts") or ""


# ── sampling ──────────────────────────────────────────────────────────────

@dataclass
class Sample:
    ts: str
    cpu_percent: Optional[float]
    cpu_seconds_total: Optional[float]
    mem_rss_mb: Optional[float]
    mem_peak_rss_mb: Optional[float]
    disk_read_mb_total: Optional[float]
    disk_write_mb_total: Optional[float]
    net_sent_mb_total: Optional[float]
    net_recv_mb_total: Optional[float]
    open_fds: Optional[int]
    thread_count: Optional[int]
    uptime_seconds: Optional[float]
    knowledge_db_mb: Optional[float]
    ledger_db_mb: Optional[float]
    treasury_db_mb: Optional[float]
    workload_db_mb: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _sizeof_db(name: str) -> Optional[float]:
    """Return the on-disk size of a DMAI SQLite file in MB, or None."""
    base = os.environ.get("DATA_PATH", "data")
    p = Path(base) / name
    try:
        if p.exists():
            return round(p.stat().st_size / (1024.0 * 1024.0), 4)
    except Exception:
        pass
    return None


def _snapshot(process: Optional[Any] = None) -> Sample:
    """Take a psutil snapshot of the current process."""
    now = datetime.now(timezone.utc).isoformat()

    if psutil is None:
        return Sample(
            ts=now,
            cpu_percent=None, cpu_seconds_total=None,
            mem_rss_mb=None, mem_peak_rss_mb=None,
            disk_read_mb_total=None, disk_write_mb_total=None,
            net_sent_mb_total=None, net_recv_mb_total=None,
            open_fds=None, thread_count=None, uptime_seconds=None,
            knowledge_db_mb=_sizeof_db("dmai_knowledge.db"),
            ledger_db_mb=_sizeof_db("dmai_ledger.db"),
            treasury_db_mb=_sizeof_db("dmai_treasury.db"),
            workload_db_mb=_sizeof_db("dmai_workload.db"),
        )

    p = process or psutil.Process()

    # CPU
    try:
        cpu_pct = p.cpu_percent(interval=None)
    except Exception:
        cpu_pct = None
    try:
        cpu_times = p.cpu_times()
        cpu_secs = float(cpu_times.user + cpu_times.system)
    except Exception:
        cpu_secs = None

    # Memory
    try:
        mem = p.memory_info()
        rss_mb = round(mem.rss / (1024.0 * 1024.0), 3)
    except Exception:
        rss_mb = None
    try:
        peak = getattr(psutil.Process(), "memory_full_info", None)
        peak_mb = None
        if peak is not None:
            info = p.memory_info()
            # psutil doesn't expose peak RSS portably; use the current
            # RSS as a lower bound and let the loop track the maximum.
            peak_mb = round(info.rss / (1024.0 * 1024.0), 3)
    except Exception:
        peak_mb = None

    # Disk I/O (process-scoped; may be unavailable on some platforms)
    try:
        io = p.io_counters()
        d_read = round(io.read_bytes  / (1024.0 * 1024.0), 3)
        d_write = round(io.write_bytes / (1024.0 * 1024.0), 3)
    except Exception:
        d_read = d_write = None

    # Net I/O — psutil.net_io_counters is host-wide, but for
    # single-tenant Render web services host-wide == process-wide
    # for practical purposes.
    try:
        net = psutil.net_io_counters()
        n_sent = round(net.bytes_sent / (1024.0 * 1024.0), 3)
        n_recv = round(net.bytes_recv / (1024.0 * 1024.0), 3)
    except Exception:
        n_sent = n_recv = None

    # FDs + threads + uptime
    try:
        fds = p.num_fds() if hasattr(p, "num_fds") else None
    except Exception:
        fds = None
    try:
        threads = p.num_threads()
    except Exception:
        threads = None
    try:
        uptime = time.time() - p.create_time()
    except Exception:
        uptime = None

    return Sample(
        ts=now,
        cpu_percent=cpu_pct,
        cpu_seconds_total=cpu_secs,
        mem_rss_mb=rss_mb,
        mem_peak_rss_mb=peak_mb,
        disk_read_mb_total=d_read,
        disk_write_mb_total=d_write,
        net_sent_mb_total=n_sent,
        net_recv_mb_total=n_recv,
        open_fds=fds,
        thread_count=threads,
        uptime_seconds=uptime,
        knowledge_db_mb=_sizeof_db("dmai_knowledge.db"),
        ledger_db_mb=_sizeof_db("dmai_ledger.db"),
        treasury_db_mb=_sizeof_db("dmai_treasury.db"),
        workload_db_mb=_sizeof_db("dmai_workload.db"),
    )


def sample_now(db_path: Optional[str] = None,
               process: Optional[Any] = None) -> Sample:
    """Take a snapshot and persist it. Returns the Sample."""
    s = _snapshot(process=process)
    with _connect(db_path) as c:
        c.execute(
            "INSERT INTO workload_samples("
            "ts, cpu_percent, cpu_seconds_total, mem_rss_mb, "
            "mem_peak_rss_mb, disk_read_mb_total, disk_write_mb_total, "
            "net_sent_mb_total, net_recv_mb_total, open_fds, "
            "thread_count, uptime_seconds, knowledge_db_mb, "
            "ledger_db_mb, treasury_db_mb, workload_db_mb) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.ts, s.cpu_percent, s.cpu_seconds_total,
             s.mem_rss_mb, s.mem_peak_rss_mb,
             s.disk_read_mb_total, s.disk_write_mb_total,
             s.net_sent_mb_total, s.net_recv_mb_total,
             s.open_fds, s.thread_count, s.uptime_seconds,
             s.knowledge_db_mb, s.ledger_db_mb,
             s.treasury_db_mb, s.workload_db_mb),
        )
        _set_state(c, "last_sample_ts", s.ts)
        c.commit()
    return s


# ── readers ───────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def get_recent(hours: int = 24,
               db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return samples from the last ``hours`` hours, oldest first."""
    if hours <= 0:
        return []
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=hours)).isoformat()
    with _connect(db_path) as c:
        rows = c.execute(
            "SELECT * FROM workload_samples "
            "WHERE ts >= ? ORDER BY ts ASC", (cutoff,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_latest(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as c:
        row = c.execute(
            "SELECT * FROM workload_samples "
            "ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row) if row else None


def _safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def get_daily_rollup(days: int = 7,
                     db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return daily rollups (UTC calendar days) for the last ``days`` days.

    Each rollup contains:
        - ``day``: ISO calendar date (YYYY-MM-DD, UTC)
        - ``samples``: number of samples in the bucket
        - ``avg_cpu_percent``, ``peak_cpu_percent``
        - ``avg_rss_mb``, ``peak_rss_mb``
        - ``cpu_seconds_delta``: end-of-day - start-of-day
        - ``disk_read_mb_delta``, ``disk_write_mb_delta``
        - ``net_sent_mb_delta``, ``net_recv_mb_delta``
    """
    if days <= 0:
        return []
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=days)).isoformat()
    with _connect(db_path) as c:
        rows = c.execute(
            "SELECT * FROM workload_samples "
            "WHERE ts >= ? ORDER BY ts ASC", (cutoff,),
        ).fetchall()

    buckets: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        day = r["ts"][:10]
        buckets.setdefault(day, []).append(r)

    out: List[Dict[str, Any]] = []
    for day in sorted(buckets):
        bucket = buckets[day]
        cpu_pcts = [b["cpu_percent"] for b in bucket
                    if b["cpu_percent"] is not None]
        rss_vals = [b["mem_rss_mb"] for b in bucket
                    if b["mem_rss_mb"] is not None]
        first, last = bucket[0], bucket[-1]
        out.append({
            "day":                 day,
            "samples":             len(bucket),
            "avg_cpu_percent":     (round(sum(cpu_pcts) / len(cpu_pcts), 3)
                                    if cpu_pcts else None),
            "peak_cpu_percent":    (round(max(cpu_pcts), 3)
                                    if cpu_pcts else None),
            "avg_rss_mb":          (round(sum(rss_vals) / len(rss_vals), 3)
                                    if rss_vals else None),
            "peak_rss_mb":         (round(max(rss_vals), 3)
                                    if rss_vals else None),
            "cpu_seconds_delta":   _safe_delta(last["cpu_seconds_total"],
                                               first["cpu_seconds_total"]),
            "disk_read_mb_delta":  _safe_delta(last["disk_read_mb_total"],
                                               first["disk_read_mb_total"]),
            "disk_write_mb_delta": _safe_delta(last["disk_write_mb_total"],
                                               first["disk_write_mb_total"]),
            "net_sent_mb_delta":   _safe_delta(last["net_sent_mb_total"],
                                               first["net_sent_mb_total"]),
            "net_recv_mb_delta":   _safe_delta(last["net_recv_mb_total"],
                                               first["net_recv_mb_total"]),
        })
    return out


def get_db_growth(days: int = 7,
                  db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return DB size growth over the window.

    Returns a dict with per-DB growth (MB), plus a growth rate (MB/day).
    """
    if days <= 0:
        return {}
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=days)).isoformat()
    with _connect(db_path) as c:
        first = c.execute(
            "SELECT * FROM workload_samples "
            "WHERE ts >= ? ORDER BY ts ASC LIMIT 1", (cutoff,),
        ).fetchone()
        last = c.execute(
            "SELECT * FROM workload_samples "
            "WHERE ts >= ? ORDER BY ts DESC LIMIT 1", (cutoff,),
        ).fetchone()
    if not first or not last or first["id"] == last["id"]:
        return {"window_days": days, "samples": 0, "growth": {}}

    out: Dict[str, Any] = {
        "window_days":  days,
        "start_ts":     first["ts"],
        "end_ts":       last["ts"],
        "samples":      2,
        "growth":       {},
    }
    for col in ("knowledge_db_mb", "ledger_db_mb",
                "treasury_db_mb", "workload_db_mb"):
        delta = _safe_delta(last[col], first[col])
        per_day = None
        if delta is not None and days > 0:
            per_day = round(delta / days, 4)
        out["growth"][col.replace("_mb", "")] = {
            "start_mb":   first[col],
            "end_mb":     last[col],
            "delta_mb":   delta,
            "mb_per_day": per_day,
        }
    return out


def get_status(db_path: Optional[str] = None) -> Dict[str, Any]:
    """One-call summary used by the admin endpoint + digest."""
    with _connect(db_path) as c:
        n = c.execute(
            "SELECT COUNT(*) AS n FROM workload_samples"
        ).fetchone()["n"]
    return {
        "install_ts":     get_install_ts(db_path),
        "sample_count":   int(n),
        "latest":         get_latest(db_path),
        "rollup_24h":     get_daily_rollup(days=1, db_path=db_path),
        "rollup_7d":      get_daily_rollup(days=7, db_path=db_path),
        "db_growth_7d":   get_db_growth(days=7, db_path=db_path),
    }


__all__ = [
    "Sample", "default_workload_path", "init_workload_db",
    "get_install_ts", "sample_now", "get_recent", "get_latest",
    "get_daily_rollup", "get_db_growth", "get_status",
]
