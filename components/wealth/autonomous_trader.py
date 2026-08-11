"""
Autonomous trader loop for DMAI.

Wraps AggressiveTrader with:
  - Background thread that ticks every 5 minutes during US market hours
    (Mon-Fri 09:30-16:00 America/New_York).
  - Paper-mode hard-enforced unless TRADING_LIVE=true env var is set
    (re-checked on every tick; flag flip is honoured without restart).
  - Escalating risk caps: tier auto-promotes after a rolling 5-day +EV
    track record and demotes on drawdown. Starts CONSERVATIVE.
  - Microfish-driven +EV gate per signal (skip when below tier threshold).
  - Daily-deploy + daily-trade-count + daily-drawdown circuit breakers.
  - Full SQLite audit of every tick, decision, order, and tier change.

No live execution unless ALL of:
  TRADING_LIVE=true  AND  loop.enabled=True  AND  market open.

Tables (created on first init in shared dmai_knowledge.db):
  at_state           - singleton: enabled, tier, last_tick, today P&L
  at_ticks           - one row per loop tick (market_open, signals_seen, trades_placed)
  at_trades          - one row per executed trade (paper or live)
  at_tier_changes    - audit of every tier promotion/demotion
"""

import os
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)


# ── Risk tiers (conservative → moderate → aggressive) ─────────────────────────
TIERS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "max_pct_per_trade": 0.02,    # 2% of wealth wallet per trade
        "max_pct_per_day":   0.10,    # 10% deployed/day
        "max_trades_per_day": 3,
        "ev_gate":            0.07,   # require ≥7% expected edge
        "max_daily_drawdown": 0.03,   # halt loop if -3% on the day
    },
    "moderate": {
        "max_pct_per_trade": 0.05,
        "max_pct_per_day":   0.25,
        "max_trades_per_day": 6,
        "ev_gate":            0.05,
        "max_daily_drawdown": 0.05,
    },
    "aggressive": {
        "max_pct_per_trade": 0.10,
        "max_pct_per_day":   0.50,
        "max_trades_per_day": 10,
        "ev_gate":            0.03,
        "max_daily_drawdown": 0.08,
    },
}

TIER_ORDER = ["conservative", "moderate", "aggressive"]


def _bytes_to_str(v):
    """Decode bytes -> str so jsonify() doesn't TypeError on legacy DB rows."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v.decode("latin-1", errors="replace")
    return v


def _row_safe(row):
    """Convert a dict to a JSON-safe dict, decoding bytes values."""
    if row is None:
        return None
    try:
        d = dict(row)
    except Exception:
        return row
    return {k: _bytes_to_str(v) for k, v in d.items()}


def _norm_tier(value, default: str = "conservative") -> str:
    """Coerce a tier value to a known string key.

    Defensive against legacy DB rows that stored the value as bytes
    (SQLite blob affinity quirk) instead of TEXT — would raise
    KeyError on TIERS[tier] otherwise. Also normalises case/whitespace.
    """
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return default
    try:
        v = str(value).strip().lower()
    except Exception:
        return default
    return v if v in TIERS else default


# ── Paper / live execution mode (PR #166) ─────────────────────────────────────
# Distinct from the cadence "mode" (scheduled|live). This is the at_state.mode
# column that decides whether AggressiveTrader points at the Alpaca paper or live
# API. Defaults to 'paper' so a fresh boot never trades real money by accident;
# the user flips it to 'live' explicitly via POST /api/trader/at-mode.
AT_MODES = ("paper", "live")


def _norm_at_mode(value, default: str = "paper") -> str:
    """Coerce an at_state.mode value to 'paper' or 'live'. Bytes/NULL → default."""
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return default
    try:
        v = str(value).strip().lower()
    except Exception:
        return default
    return v if v in AT_MODES else default

# Promotion rule: ≥10 trades over last 5 sessions AND rolling P&L ≥ +3% → step up.
# Demotion rule: rolling P&L ≤ -2% over last 5 sessions → step down.
PROMOTE_TRADES   = 10
PROMOTE_PNL_PCT  = 0.03
DEMOTE_PNL_PCT   = -0.02
ROLLING_WINDOW_DAYS = 5

LOOP_INTERVAL_SECONDS = 300   # legacy default; scheduled cadence is now env-driven (2h)
MARKET_TZ_OFFSET_HOURS = -4   # EDT default; refined per tick via _us_market_open()

# ── Cadence (PR #163) ─────────────────────────────────────────────────────────
# Post-#162 the trader became the dominant write-mutex holder: a 5-min loop tick
# hitting c.execute() under SQLite write pressure produced a steady
# "database is locked" storm. We rate-limit to a default 2h cadence, with a
# dashboard-controlled live-mode override (30s) that auto-expires after 4h so it
# can't accidentally re-trigger the lock storm. All env reads happen at call
# time so a running loop honours changes without restart.
MODE_SCHEDULED = "scheduled"
MODE_LIVE = "live"
SLEEP_CHUNK_SECONDS = 10       # re-check mode file this often mid-sleep
HEARTBEAT_EVERY_SECONDS = 300  # refresh heartbeat ~every 5 min during sleep


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return v if v > 0 else default
    except Exception:
        return default


def _tick_interval_s() -> int:
    """Scheduled-mode cadence (default 2h)."""
    return _env_int("AUTONOMOUS_TRADER_TICK_INTERVAL_SECONDS", 7200)


def _live_interval_s() -> int:
    """Live-mode cadence while the dashboard trading window is open (default 30s)."""
    return _env_int("AUTONOMOUS_TRADER_LIVE_INTERVAL_SECONDS", 30)


def _live_max_minutes() -> int:
    """Live-mode auto-expiry: revert to scheduled after this long (default 4h)."""
    return _env_int("AUTONOMOUS_TRADER_LIVE_MAX_MINUTES", 240)
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS at_state (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        enabled         INTEGER NOT NULL DEFAULT 0,
        tier            TEXT    NOT NULL DEFAULT 'conservative',
        last_tick_ts    TEXT,
        last_tick_note  TEXT,
        today_date      TEXT,
        today_deployed_pct REAL NOT NULL DEFAULT 0,
        today_trades    INTEGER NOT NULL DEFAULT 0,
        today_open_eq   REAL,
        created_at      TEXT    NOT NULL DEFAULT (NOW()),
        updated_at      TEXT    NOT NULL DEFAULT (NOW())
    )""",
    """CREATE TABLE IF NOT EXISTS at_ticks (
        id              SERIAL PRIMARY KEY,
        ts              TEXT    NOT NULL DEFAULT (NOW()),
        market_open     INTEGER NOT NULL,
        tier            TEXT    NOT NULL,
        live            INTEGER NOT NULL,
        signals_seen    INTEGER NOT NULL DEFAULT 0,
        signals_passed  INTEGER NOT NULL DEFAULT 0,
        trades_placed   INTEGER NOT NULL DEFAULT 0,
        note            TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS at_trades (
        id              SERIAL PRIMARY KEY,
        ts              TEXT    NOT NULL DEFAULT (NOW()),
        symbol          TEXT    NOT NULL,
        side            TEXT    NOT NULL,
        qty             REAL,
        confidence      REAL,
        ev              REAL,
        tier            TEXT    NOT NULL,
        live            INTEGER NOT NULL,
        result_json     TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS at_tier_changes (
        id              SERIAL PRIMARY KEY,
        ts              TEXT    NOT NULL DEFAULT (NOW()),
        from_tier       TEXT    NOT NULL,
        to_tier         TEXT    NOT NULL,
        reason          TEXT    NOT NULL
    )""",
]


def _is_live() -> bool:
    """Hard re-check on every tick — env-driven, no caching."""
    return os.getenv("TRADING_LIVE", "").strip().lower() == "true"


def _us_market_open(now_utc: Optional[datetime] = None) -> bool:
    """US equity market hours check: Mon-Fri 09:30-16:00 ET.
    Approximates ET as UTC-4 (DST) or UTC-5 (standard).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    # Cheap DST approximation: Mar-Nov → UTC-4, else UTC-5.
    et_offset = -4 if 3 <= now_utc.month <= 11 else -5
    et = now_utc + timedelta(hours=et_offset)
    if et.weekday() >= 5:  # Sat/Sun
        return False
    minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


class AutonomousTrader:
    """Background loop that drives AggressiveTrader on a schedule."""

    def __init__(
        self,
        db_path: str,
        trader: Any,                    # AggressiveTrader instance
        prediction_engine: Optional[Any] = None,
        notifier: Optional[Any] = None,
        universe: Optional[List[str]] = None,
        loop_interval_s: Optional[int] = None,
        require_approval: bool = False,
    ):
        self.db_path = db_path
        self.trader = trader
        self.prediction_engine = prediction_engine or getattr(trader, "prediction_engine", None)
        self.notifier = notifier
        self.universe = universe or (
            getattr(trader, "conservative_pairs", []) + getattr(trader, "trading_pairs", [])
        )
        # Scheduled-mode cadence. Defaults to the env-driven 2h interval so the
        # TraderWatchdog's staleness threshold (2 * loop_interval_s + 30) scales
        # with the real cadence instead of the old 5-min value.
        self.loop_interval_s = loop_interval_s if loop_interval_s is not None else _tick_interval_s()
        self.require_approval = bool(require_approval)

        # Mode + heartbeat files live alongside the DB so the API layer and the
        # loop share one location, and tests can point at a temp dir.
        _data_dir = os.path.dirname(os.path.abspath(self.db_path))
        self.mode_file = os.path.join(_data_dir, "autonomous_trader_mode.txt")
        self.heartbeat_file = os.path.join(_data_dir, "autonomous_trader_heartbeat.txt")

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Once-per-instance guard for _ensure_tables. Every public entry used
        # to call _ensure_tables(), which runs a CREATE TABLE IF NOT EXISTS
        # loop through the write-gating proxy — each call briefly held the
        # process-wide write mutex on dmai_knowledge.db. Under contention that
        # made GET /api/trader/at-mode queue up to the 30 s write_mutex_timeout
        # (observed: 36.7 s latency post-boot). Since PR #175's boot-time
        # _ensure_kdb_schema already creates every trader table, this call is
        # redundant after the first invocation on this instance.
        self._tables_ensured = False

        self._init_db()
        self._ensure_state_row()

        # ExitManager + StrategyLab (autonomous exits + ongoing variant backtesting)
        self.exit_manager = None
        self.strategy_lab = None
        try:
            from components.wealth.exit_manager import ExitManager as _ExitMgr
            self.exit_manager = _ExitMgr(
                db_path=self.db_path,
                trader=self.trader,
                prediction_engine=self.prediction_engine,
                notifier=self.notifier,
            )
            logger.info("AutonomousTrader: ExitManager attached")
        except Exception as _e:
            logger.warning("AutonomousTrader: ExitManager unavailable: %s", _e)
        try:
            from components.wealth.strategy_lab import StrategyLab as _SLab
            self.strategy_lab = _SLab(db_path=self.db_path)
            try:
                self.strategy_lab.start()
            except Exception as _se:
                logger.debug("StrategyLab.start() skipped: %s", _se)
            logger.info("AutonomousTrader: StrategyLab attached + loop started")
        except Exception as _e:
            logger.warning("AutonomousTrader: StrategyLab unavailable: %s", _e)

        self._start_loop()

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL + self-heal on malformed DB.

        WAL mode is the cure for the concurrent-writer corruption that's been
        plaguing dmai_knowledge.db. If we hit 'database disk image is malformed'
        we quarantine the file, recreate schema, and seed at_state.
        """
        try:
            c = safe_open_kdb(self.db_path, timeout=30)
            c.row_factory = dict
            try:
                pass
            except Exception:
                pass
            return c
        except sqlite3.DatabaseError as e:
            if "malformed" not in str(e).lower():
                raise
            logger.error(
                "AutonomousTrader: malformed DB on connect, self-healing: %s", e
            )
            self._self_heal_db()
            c = safe_open_kdb(self.db_path, timeout=30)
            c.row_factory = dict
            try:
                pass
            except Exception:
                pass
            return c

    def _self_heal_db(self) -> None:
        """Quarantine a malformed DB file and recreate fresh schema + state row."""
        import os as _hos, time as _ht
        try:
            if _hos.path.exists(self.db_path):
                quarantine = self.db_path + f".malformed_{int(_ht.time())}"
                _hos.rename(self.db_path, quarantine)
                logger.warning(
                    "AutonomousTrader: quarantined malformed DB -> %s", quarantine
                )
            for sfx in ("-wal", "-shm"):
                p = self.db_path + sfx
                if _hos.path.exists(p):
                    try:
                        _hos.remove(p)
                    except Exception:
                        pass
        except Exception as he:
            logger.error("AutonomousTrader: self-heal rename failed: %s", he)
        try:
            fresh = safe_open_kdb(self.db_path, timeout=30)
            try:
                for ddl in SCHEMA:
                    fresh.execute(ddl)
                self._migrate_mode_column(fresh)
                row = fresh.execute(
                    "SELECT id FROM at_state WHERE id = 1"
                ).fetchone()
                if not row:
                    fresh.execute(
                        "INSERT INTO at_state(id, enabled, tier) "
                        "VALUES (1, 0, 'conservative')"
                    )
                fresh.commit()
            finally:
                fresh.close()
            logger.info(
                "AutonomousTrader: fresh schema laid down after self-heal"
            )
        except Exception as se:
            logger.error("AutonomousTrader: schema recreation failed: %s", se)

    @staticmethod
    def _migrate_mode_column(c) -> None:
        """Add at_state.mode if missing (idempotent). Backfill NULLs to 'paper'
        so no tick ever runs without an explicit paper/live mode (PR #166 A.1/A.4).
        """
        try:
            c.execute(
                "ALTER TABLE at_state ADD COLUMN mode TEXT NOT NULL DEFAULT 'paper' "
                "CHECK (mode IN ('paper', 'live'))"
            )
        except Exception:
            # Column already exists — re-run is a no-op.
            pass
        try:
            c.execute(
                "UPDATE at_state SET mode = 'paper' WHERE mode IS NULL AND id = 1"
            )
        except Exception:
            pass

    def _init_db(self) -> None:
        with self._conn() as c:
            for ddl in SCHEMA:
                c.execute(ddl)
            self._migrate_mode_column(c)
            c.commit()

    def _ensure_tables_once(self) -> None:
        """Run _ensure_tables at most once per instance, and only if the
        boot-time bootstrap actually failed to lay down the schema.

        Background: every public entry previously called self._ensure_tables(),
        which routes every CREATE TABLE / ALTER TABLE through the
        KeepOpenProxy write-gate. Under any background writer holding the
        SQLite file lock, even the first _ensure_tables call blocks on
        BUSY-wait for up to timeout=30s — turning GET /api/trader/at-mode
        into a 30-second stall (observed live 2026-07-12 via
        /api/admin/db-lock-status).

        no BUSY-wait — confirms the schema is in place, in which case we
        set the guard and skip _ensure_tables entirely. Fallback: only if
        boot bootstrap actually failed do we run the full ensure.
        """
        if self._tables_ensured:
            return
        try:
            if self._schema_ready_readonly():
                self._tables_ensured = True
                return
        except Exception as e:  # never let diagnostics break the app
            logger.debug("trader schema readonly check failed, falling back: %s", e)
        self._ensure_tables()
        self._tables_ensured = True

    def _schema_ready_readonly(self) -> bool:
        """Read-only check: does at_state exist with the 'mode' column?

        Runs on a fresh raw sqlite3 connection (NOT the KeepOpenProxy) so
        """
        import sqlite3 as _sq
        try:
            # Read-only URI mode: no journal-mode changes, cannot write,
            # cannot flap the on-disk locking state — safe alongside the
            # KeepOpenProxy's WAL connection (see PR #175).
            raw = _sq.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=2)
        except Exception:
            return False
        try:
            row = raw.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name='at_state'"
            ).fetchone()
            if not row:
                return False
            if "mode" not in cols:
                return False
            # Also confirm the singleton row exists so get_at_mode/set_at_mode
            # don't hit a missing-row path.
            r = raw.execute("SELECT 1 FROM at_state WHERE id = 1").fetchone()
            return r is not None
        finally:
            try:
                raw.close()
            except Exception:
                pass

    def _ensure_tables(self) -> None:
        """Idempotent table create + state-row seed. Safe to call on every public entry."""
        try:
            with self._conn() as c:
                for ddl in SCHEMA:
                    c.execute(ddl)
                self._migrate_mode_column(c)
                row = c.execute("SELECT id, tier FROM at_state WHERE id = 1").fetchone()
                if not row:
                    c.execute(
                        "INSERT INTO at_state(id, enabled, tier) VALUES (1, 0, 'conservative')"
                    )
                else:
                    # Self-heal: if tier was stored as BLOB (legacy bug) the
                    # comparison TIERS[tier] would KeyError. Rewrite as TEXT.
                    raw_tier = row["tier"] if hasattr(row, "keys") else row[1]
                    if isinstance(raw_tier, bytes) or (raw_tier and str(raw_tier) not in TIERS):
                        clean = _norm_tier(raw_tier)
                        c.execute(
                            "UPDATE at_state SET tier = ?, updated_at = NOW() WHERE id = 1",
                            (clean,),
                        )
                        logger.warning(
                            "AutonomousTrader: cleaned malformed tier value %r -> %r",
                            raw_tier, clean,
                        )
                c.commit()
        except Exception as e:
            logger.warning("AutonomousTrader._ensure_tables: %s", e)

    def _ensure_state_row(self) -> None:
        with self._conn() as c:
            row = c.execute("SELECT id FROM at_state WHERE id = 1").fetchone()
            if not row:
                c.execute(
                    "INSERT INTO at_state(id, enabled, tier) VALUES (1, 0, 'conservative')"
                )
                c.commit()

    # ── Public surface ────────────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        self._ensure_tables_once()
        with self._conn() as c:
            s = c.execute("SELECT * FROM at_state WHERE id = 1").fetchone()
            recent = c.execute(
                "SELECT ts, market_open, tier, live, signals_seen, signals_passed, "
                "trades_placed, note FROM at_ticks ORDER BY id DESC LIMIT 20"
            ).fetchall()
            trades = c.execute(
                "SELECT ts, symbol, side, qty, confidence, ev, tier, live "
                "FROM at_trades ORDER BY id DESC LIMIT 20"
            ).fetchall()
        s_safe = _row_safe(s)
        tier = _norm_tier(s_safe.get("tier")) if s_safe else "conservative"
        mode_info = self.mode_status(mutate_expiry=False)
        return {
            "enabled":          bool(s_safe.get("enabled")) if s_safe else False,
            "at_mode":          _norm_at_mode(s_safe.get("mode")) if s_safe else "paper",
            "tier":             tier,
            "tier_caps":        TIERS[tier],
            "live":             _is_live(),
            "market_open":      _us_market_open(),
            "loop_interval_s":  self.loop_interval_s,
            "mode":             mode_info.get("mode"),
            "mode_expires_at":  mode_info.get("expires_at"),
            "next_tick_at":     mode_info.get("next_tick_at"),
            "scheduled_interval_s": _tick_interval_s(),
            "live_interval_s":  _live_interval_s(),
            "require_approval": self.require_approval,
            "last_tick_ts":     s_safe.get("last_tick_ts") if s_safe else None,
            "last_tick_note":   s_safe.get("last_tick_note") if s_safe else None,
            "today_date":       s_safe.get("today_date") if s_safe else None,
            "today_deployed_pct": s_safe.get("today_deployed_pct") if s_safe else 0,
            "today_trades":     s_safe.get("today_trades") if s_safe else 0,
            "today_open_eq":    s_safe.get("today_open_eq") if s_safe else None,
            "universe":         self.universe,
            "recent_ticks":     [_row_safe(r) for r in recent],
            "recent_trades":    [_row_safe(r) for r in trades],
            "pending_count":    self._pending_count(),
        }

    def set_enabled(self, enabled: bool, reason: str = "manual") -> Dict[str, Any]:
        with self._conn() as c:
            c.execute(
                "UPDATE at_state SET enabled = ?, updated_at = NOW() WHERE id = 1",
                (1 if enabled else 0,),
            )
            c.commit()
        logger.info("AutonomousTrader: enabled=%s (%s)", enabled, reason)
        return self.status()

    # ── Paper / live execution mode (PR #166) ──────────────────────────────────
    def get_at_mode(self) -> Dict[str, Any]:
        """Current paper/live execution mode + enabled flag + next tick estimate.

        Shape: {"mode": "paper"|"live", "enabled": bool, "next_tick_at": iso}.
        """
        self._ensure_tables_once()
        with self._conn() as c:
            row = c.execute(
                "SELECT mode, enabled FROM at_state WHERE id = 1"
            ).fetchone()
        mode = _norm_at_mode(row["mode"]) if row else "paper"
        enabled = bool(row["enabled"]) if row else False
        return {
            "mode": mode,
            "enabled": enabled,
            "next_tick_at": self.mode_status(mutate_expiry=False).get("next_tick_at"),
        }

    def set_at_mode(self, mode: str, reason: str = "admin") -> Dict[str, Any]:
        """Flip the trader between paper and live execution. Persists to
        at_state.mode and logs the transition at WARNING so it's visible in the
        Render logs. Raises ValueError on an unknown value."""
        norm = str(mode or "").strip().lower()
        if norm not in AT_MODES:
            raise ValueError("mode must be 'paper' or 'live'")
        self._ensure_tables_once()
        with self._conn() as c:
            cur = c.execute("SELECT mode FROM at_state WHERE id = 1").fetchone()
            from_mode = _norm_at_mode(cur["mode"]) if cur else "paper"
            c.execute(
                "UPDATE at_state SET mode = ?, updated_at = NOW() WHERE id = 1",
                (norm,),
            )
            c.commit()
        if from_mode != norm:
            logger.warning(
                "AutonomousTrader mode changed: %s → %s by %s",
                from_mode, norm, reason,
            )
        return self.get_at_mode()

    def set_tier(self, tier: str, reason: str = "manual_override") -> Dict[str, Any]:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier: {tier}")
        with self._conn() as c:
            cur = c.execute("SELECT tier FROM at_state WHERE id = 1").fetchone()
            from_tier = _norm_tier(cur["tier"]) if cur else "conservative"
            if from_tier == tier:
                return self.status()
            c.execute(
                "UPDATE at_state SET tier = ?, updated_at = NOW() WHERE id = 1",
                (tier,),
            )
            c.execute(
                "INSERT INTO at_tier_changes(from_tier, to_tier, reason) VALUES (?, ?, ?)",
                (from_tier, tier, reason),
            )
            c.commit()
        logger.info("AutonomousTrader: tier %s -> %s (%s)", from_tier, tier, reason)
        if self.notifier:
            try:
                self.notifier.tier_change(from_tier, tier, reason)
            except Exception as e:
                logger.debug("notifier.tier_change failed: %s", e)
        return self.status()

    def set_require_approval(self, on: bool) -> Dict[str, Any]:
        self.require_approval = bool(on)
        return self.status()

    # ── Loop ──────────────────────────────────────────────────────────────────
    def _start_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="AutonomousTrader-loop", daemon=True)
        self._thread = t
        t.start()
        logger.info(
            "AutonomousTrader: loop started (scheduled=%ss, live=%ss, live_max=%dm)",
            _tick_interval_s(), _live_interval_s(), _live_max_minutes(),
        )

    def stop(self) -> None:
        self._stop.set()

    # ── Mode file (scheduled | live) ────────────────────────────────────────────
    def _read_mode(self) -> str:
        """Read the mode file, creating it as 'scheduled' if absent.

        Returns MODE_LIVE or MODE_SCHEDULED. Any unrecognised/garbled content is
        treated as scheduled (fail safe — never leave the fast loop running by
        accident)."""
        try:
            with open(self.mode_file, "r", encoding="utf-8") as fh:
                val = (fh.read() or "").strip().lower()
            return MODE_LIVE if val == MODE_LIVE else MODE_SCHEDULED
        except FileNotFoundError:
            self._write_mode(MODE_SCHEDULED)
            return MODE_SCHEDULED
        except Exception as e:
            logger.debug("AutonomousTrader: mode read failed (%s), assuming scheduled", e)
            return MODE_SCHEDULED

    def _write_mode(self, mode: str) -> None:
        """Atomically write the mode file. Rewrite bumps mtime → resets live expiry."""
        mode = MODE_LIVE if str(mode).strip().lower() == MODE_LIVE else MODE_SCHEDULED
        tmp = self.mode_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(mode)
            os.replace(tmp, self.mode_file)
        except Exception as e:
            logger.warning("AutonomousTrader: mode write failed: %s", e)

    def _mode_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.mode_file)
        except Exception:
            return None

    def mode_status(self, mutate_expiry: bool = False) -> Dict[str, Any]:
        """Current effective mode + timing. When mutate_expiry is True and live
        mode has aged past LIVE_MAX_MINUTES, revert the file to scheduled.

        Returns {"mode", "expires_at", "next_tick_at"} (ISO-8601 / None)."""
        now = datetime.now(timezone.utc)
        mode = self._read_mode()
        expires_at = None
        if mode == MODE_LIVE:
            mtime = self._mode_mtime()
            if mtime is not None:
                exp = datetime.fromtimestamp(mtime, timezone.utc) + timedelta(
                    minutes=_live_max_minutes()
                )
                if now >= exp:
                    if mutate_expiry:
                        self._write_mode(MODE_SCHEDULED)
                        logger.info(
                            "AutonomousTrader live mode expired, reverting to scheduled"
                        )
                    mode = MODE_SCHEDULED
                else:
                    expires_at = exp.isoformat()
        interval = _live_interval_s() if mode == MODE_LIVE else _tick_interval_s()
        return {
            "mode": mode,
            "expires_at": expires_at,
            "next_tick_at": (now + timedelta(seconds=interval)).isoformat(),
        }

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Set trader cadence mode. Called by POST /api/trader/mode.

        Writing 'live' bumps the file mtime so the 4h auto-expiry clock restarts.
        """
        norm = str(mode or "").strip().lower()
        if norm not in (MODE_LIVE, MODE_SCHEDULED):
            raise ValueError("mode must be 'live' or 'scheduled'")
        self._write_mode(norm)
        logger.info("AutonomousTrader: mode set to %s", norm)
        return self.mode_status(mutate_expiry=False)

    def _effective_interval(self) -> tuple:
        """(interval_seconds, mode) for the current sleep, applying auto-expiry."""
        st = self.mode_status(mutate_expiry=True)
        mode = st["mode"]
        interval = _live_interval_s() if mode == MODE_LIVE else _tick_interval_s()
        return interval, mode

    def _write_heartbeat(self, mode: str, remaining_s: float) -> None:
        """Atomically stamp the heartbeat file (consumed by PR #164 self_healer)."""
        tmp = self.heartbeat_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(datetime.now(timezone.utc).isoformat())
            os.replace(tmp, self.heartbeat_file)
        except Exception as e:
            logger.debug("AutonomousTrader: heartbeat write failed: %s", e)
        rem = max(0, int(remaining_s))
        logger.info(
            "AutonomousTrader alive, mode=%s, next tick in %dm %ds",
            mode, rem // 60, rem % 60,
        )

    def _run(self) -> None:
        # Stagger first tick by ~10s so the rest of the app finishes startup.
        time.sleep(10)
        while not self._stop.is_set():
            interval, mode = self._effective_interval()
            self._write_heartbeat(mode, interval)
            try:
                self.tick()
            except Exception as e:
                logger.exception("AutonomousTrader tick failed: %s", e)

            # Sleep in ~10s chunks so a dashboard flip to live mode is honoured
            # within one chunk even mid-way through a 2h scheduled sleep.
            slept = 0.0
            last_hb = time.monotonic()
            while slept < interval and not self._stop.is_set():
                chunk = min(SLEEP_CHUNK_SECONDS, interval - slept)
                time.sleep(chunk)
                slept += chunk
                new_interval, new_mode = self._effective_interval()
                # Mode changed (or shorter cadence now applies) → tick promptly.
                if new_mode != mode or new_interval < interval:
                    break
                if time.monotonic() - last_hb >= HEARTBEAT_EVERY_SECONDS:
                    self._write_heartbeat(new_mode, interval - slept)
                    last_hb = time.monotonic()

    # ── Tick (single iteration) ───────────────────────────────────────────────
    def tick(self) -> Dict[str, Any]:
        with self._lock:
            return self._tick_inner()

    def _tick_inner(self) -> Dict[str, Any]:
        live = _is_live()
        market_open = _us_market_open()

        # Reconcile the broker client with the persisted paper/live mode BEFORE
        # any execute_buy / execute_sell runs this tick (PR #166 A.2). A missing
        # or garbled value falls back to paper.
        at_mode = "paper"
        try:
            with self._conn() as c:
                row = c.execute("SELECT mode FROM at_state WHERE id = 1").fetchone()
            at_mode = _norm_at_mode(row["mode"]) if row else "paper"
        except Exception as e:
            logger.warning("AutonomousTrader: at_state.mode read failed (%s), using paper", e)
            at_mode = "paper"
        try:
            set_mode = getattr(self.trader, "set_mode", None)
            if callable(set_mode):
                set_mode(paper=(at_mode != "live"))
        except Exception as e:
            logger.warning("AutonomousTrader: trader.set_mode failed: %s", e)

        # Step 0: evaluate exits FIRST so capital is freed before new buys.
        exits_summary: Dict[str, Any] = {"checked": 0, "closed": 0}
        if self.exit_manager and market_open:
            try:
                exits_summary = self.exit_manager.evaluate(live=live) or exits_summary
                if exits_summary.get("closed"):
                    logger.info(
                        "AutonomousTrader: ExitManager closed %s position(s)",
                        exits_summary.get("closed"),
                    )
            except Exception as _e:
                logger.warning("AutonomousTrader: ExitManager.evaluate failed: %s", _e)


        with self._conn() as c:
            s = c.execute("SELECT * FROM at_state WHERE id = 1").fetchone()
            tier = _norm_tier(s["tier"]) if s else "conservative"
            enabled = bool(s["enabled"]) if s else False
            today = date.today().isoformat()
            # Roll daily counters at session boundary.
            if s["today_date"] != today:
                eq = self._equity_safe()
                c.execute(
                    "UPDATE at_state SET today_date = ?, today_deployed_pct = 0, "
                    "today_trades = 0, today_open_eq = ?, updated_at = NOW() "
                    "WHERE id = 1",
                    (today, eq),
                )
                c.commit()
                s = c.execute("SELECT * FROM at_state WHERE id = 1").fetchone()

        caps = TIERS[tier]
        note_parts: List[str] = []
        signals_seen = signals_passed = trades_placed = 0

        if not enabled:
            note_parts.append("disabled")
        if not market_open:
            note_parts.append("market_closed")

        if enabled and market_open:
            # Drawdown circuit breaker
            open_eq = s["today_open_eq"] or 0
            cur_eq = self._equity_safe()
            dd = 0.0
            if open_eq and cur_eq:
                dd = (cur_eq - open_eq) / open_eq
            if dd <= -caps["max_daily_drawdown"]:
                note_parts.append(f"drawdown_halt({dd:.2%})")
            elif s["today_trades"] >= caps["max_trades_per_day"]:
                note_parts.append("trade_cap_reached")
            elif s["today_deployed_pct"] >= caps["max_pct_per_day"]:
                note_parts.append("deploy_cap_reached")
            else:
                signals = self._collect_signals()
                signals_seen = len(signals)

                # PR ZZ-1d: paper-log EVERY signal to training_paper_trades
                # BEFORE the EV gate + trade-count cap. This captures the
                # full universe of signals the trader saw — not just the
                # ones that would be placed for money — so we can measure
                # signal quality end-to-end before switching to live.
                #
                # Entry price is synthetic (1.0 per share, qty=stake) so the
                # P/L math in ZZ-1e can express (exit_price - 1.0) * qty as
                # a normalised return against the sized paper stake. Real
                # market prices land in ZZ-1e's settlement pass.
                try:
                    from components.monetisation.training_ledger import (
                        paper_bankroll as _paper_bankroll,
                        record_paper_trade as _record_paper_trade,
                    )
                    _pb = _paper_bankroll()
                    for _sig in signals:
                        _sym = _sig.get("symbol")
                        _conf = float(_sig.get("confidence") or 0)
                        _ev = float(_sig.get("ev") or 0)
                        if not _sym or _conf <= 0:
                            continue
                        # Size like _maybe_execute would: paper_bankroll *
                        # max_pct_per_trade * confidence. Never zero — the
                        # ledger layer rejects anything <= 0.
                        _stake = round(
                            _pb * float(caps["max_pct_per_trade"]) * _conf, 4
                        )
                        if _stake <= 0:
                            continue
                        _record_paper_trade(
                            symbol=_sym,
                            side="buy",
                            entry_price=1.0,     # synthetic normalised unit
                            qty=_stake,          # 1 share per £ staked
                            stake=_stake,
                            paper_bankroll_amt=_pb,
                            confidence=_conf,
                            expected_value=_ev,
                            tier=tier,
                            passed_ev_gate=(_ev >= caps["ev_gate"]),
                            source="autonomous_trader",
                        )
                except Exception as _tl_err:
                    logger.warning("paper-trade write failed: %s", _tl_err)

                for sig in signals:
                    if signals_passed >= caps["max_trades_per_day"] - s["today_trades"]:
                        break
                    ev = float(sig.get("ev", 0))
                    if ev < caps["ev_gate"]:
                        continue
                    signals_passed += 1
                    placed = self._maybe_execute(sig, tier, caps, live, at_mode)
                    if placed:
                        trades_placed += 1
                        with self._conn() as c:
                            c.execute(
                                "UPDATE at_state SET today_trades = today_trades + 1, "
                                "today_deployed_pct = today_deployed_pct + ?, "
                                "updated_at = NOW() WHERE id = 1",
                                (caps["max_pct_per_trade"],),
                            )
                            c.commit()

        note = ",".join(note_parts) if note_parts else "ok"

        with self._conn() as c:
            c.execute(
                "INSERT INTO at_ticks(market_open, tier, live, signals_seen, "
                "signals_passed, trades_placed, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1 if market_open else 0, tier, 1 if live else 0,
                 signals_seen, signals_passed, trades_placed, note),
            )
            c.execute(
                "UPDATE at_state SET last_tick_ts = NOW(), last_tick_note = ?, "
                "updated_at = NOW() WHERE id = 1",
                (note,),
            )
            c.commit()

        # Evaluate tier escalation/demotion once per tick (cheap; reads recent rows).
        self._maybe_change_tier()

        return {
            "ts":             datetime.utcnow().isoformat() + "Z",
            "tier":           tier,
            "live":           live,
            "market_open":    market_open,
            "exits_checked":  exits_summary.get("checked", 0),
            "exits_closed":   exits_summary.get("closed", 0),
            "signals_seen":   signals_seen,
            "signals_passed": signals_passed,
            "trades_placed":  trades_placed,
            "note":           note,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _equity_safe(self) -> Optional[float]:
        try:
            acc = self.trader.get_account()
            return float(acc.get("equity") or 0) or None
        except Exception as e:
            logger.debug("AutonomousTrader: equity probe failed: %s", e)
            return None

    def _collect_signals(self) -> List[Dict[str, Any]]:
        """Pull Microfish-backed +EV signals for the universe. Falls back
        to AggressiveTrader.generate_signals() if the engine is unavailable."""
        out: List[Dict[str, Any]] = []
        engine = self.prediction_engine
        if engine is not None:
            for sym in self.universe:
                try:
                    sig = self.trader._predict_symbol(sym)  # uses Microfish
                except Exception as e:
                    logger.debug("predict %s failed: %s", sym, e)
                    sig = None
                if not sig:
                    continue
                action = (sig.get("action") or "").lower()
                conf = float(sig.get("confidence") or 0)
                if action != "buy" or conf <= 0:
                    continue
                # Treat (confidence - 0.5) * 2 as a rough EV proxy in [-1,1].
                ev = max(0.0, (conf - 0.5) * 2.0)
                out.append({"symbol": sym, "confidence": conf, "ev": ev,
                            "reason": sig.get("reason", "")})
        else:
            try:
                raw = self.trader.generate_signals() or []
            except Exception as e:
                logger.warning("generate_signals failed: %s", e)
                raw = []
            for r in raw:
                conf = float(r.get("confidence") or 0)
                out.append({"symbol": r.get("symbol"), "confidence": conf,
                            "ev": max(0.0, (conf - 0.5) * 2.0),
                            "reason": r.get("reason", "")})
        # Best EV first
        out.sort(key=lambda x: x.get("ev", 0), reverse=True)
        return out

    def _maybe_execute(self, sig: Dict[str, Any], tier: str,
                       caps: Dict[str, float], live: bool,
                       at_mode: str = "paper") -> bool:
        symbol = sig["symbol"]
        confidence = float(sig.get("confidence") or 0)
        ev = float(sig.get("ev") or 0)

        # Approval gate for live mode: queue + notify instead of executing.
        if self.require_approval and live:
            self._queue_pending(symbol, confidence, ev, tier)
            return False

        try:
            result = self.trader.execute_buy(symbol, confidence)
        except Exception as e:
            result = {"error": str(e)}
            if self.notifier:
                try:
                    self.notifier.error("execute_buy", f"{symbol}: {e}")
                except Exception:
                    pass
        success = isinstance(result, dict) and "error" not in result and \
            result.get("status") != "skipped"
        qty = float(result.get("qty") or 0) if isinstance(result, dict) else 0
        # Ledger mirror (PR #166 B.3): open row on success, error row on failure.
        # NEVER let a ledger write failure block or unwind a real trade.
        self._ledger_record_buy(symbol, qty, confidence, at_mode, result, success)
        with self._conn() as c:
            c.execute(
                "INSERT INTO at_trades(symbol, side, qty, confidence, ev, tier, live, "
                "result_json) VALUES (?, 'buy', ?, ?, ?, ?, ?, ?)",
                (symbol, qty, confidence, ev, tier, 1 if live else 0,
                 json.dumps(result)[:4000]),
            )
            c.commit()
        if success:
            logger.info("AutonomousTrader: %s buy %s conf=%.2f ev=%.2f live=%s",
                        tier, symbol, confidence, ev, live)
            if self.notifier:
                try:
                    self.notifier.trade({
                        "symbol": symbol, "qty": qty,
                        "confidence": confidence, "ev": ev,
                        "tier": tier, "live": live,
                    })
                except Exception as e:
                    logger.debug("notifier.trade failed: %s", e)
        return success

    def _ledger_record_buy(self, symbol, qty, confidence, at_mode,
                           result, success) -> None:
        """Mirror a buy into data/dmai_ledger.db. Isolated DB, best-effort:
        wrapped so a ledger failure can never block or unwind a real trade."""
        try:
            from components.ledger import ledger_db
            if success and isinstance(result, dict):
                entry_price = result.get("price")
                stake = result.get("value")
                ledger_db.insert_trade(
                    symbol=symbol, side="buy", qty=qty,
                    entry_price=entry_price, stake=stake,
                    mode=at_mode, status="open", confidence=confidence,
                    source="autonomous_trader",
                )
            else:
                err = None
                if isinstance(result, dict):
                    err = result.get("error") or result.get("reason")
                ledger_db.insert_trade(
                    symbol=symbol, side="buy", qty=qty or 0.0,
                    mode=at_mode, status="error", confidence=confidence,
                    source="autonomous_trader",
                    notes=(str(err)[:500] if err else "execute_buy failed"),
                )
        except Exception as le:
            logger.warning("AutonomousTrader: ledger buy-record failed: %s", le)

    def _maybe_change_tier(self) -> None:
        """Promote/demote tier based on rolling P&L + trade count."""
        since = (datetime.utcnow() - timedelta(days=ROLLING_WINDOW_DAYS)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT result_json FROM at_trades WHERE ts >= ?", (since,)
            ).fetchall()
            cur = c.execute("SELECT tier FROM at_state WHERE id = 1").fetchone()
        if not cur:
            return
        tier = _norm_tier(cur["tier"])
        idx = TIER_ORDER.index(tier)

        trade_count = len(rows)
        # Rough rolling P&L proxy: sum of filled_avg_price * qty deltas isn't
        # always available, so we use realised P&L from the broker if exposed.
        pnl_pct = 0.0
        try:
            acc = self.trader.get_account()
            # Alpaca paper accounts start at 100k; compute % change from that
            # if our state doesn't have an opening equity.
            base = 100000.0
            equity = float(acc.get("equity") or base)
            pnl_pct = (equity - base) / base
        except Exception:
            return

        if trade_count >= PROMOTE_TRADES and pnl_pct >= PROMOTE_PNL_PCT and idx < len(TIER_ORDER) - 1:
            self.set_tier(TIER_ORDER[idx + 1],
                          reason=f"auto_promote pnl={pnl_pct:.2%} trades={trade_count}")
        elif pnl_pct <= DEMOTE_PNL_PCT and idx > 0:
            self.set_tier(TIER_ORDER[idx - 1],
                          reason=f"auto_demote pnl={pnl_pct:.2%}")


    # ----- Pending-approval queue (live-mode safety) ------------------------
    def _ensure_pending_table(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS at_pending ("
                "id SERIAL PRIMARY KEY, "
                "ts TEXT NOT NULL DEFAULT (NOW()), "
                "symbol TEXT NOT NULL, confidence REAL, ev REAL, tier TEXT, "
                "status TEXT NOT NULL DEFAULT 'pending', "
                "resolved_ts TEXT, result_json TEXT)"
            )
            c.commit()

    def _queue_pending(self, symbol, confidence, ev, tier):
        self._ensure_pending_table()
        with self._conn() as c:
            c.execute(
                "INSERT INTO at_pending(symbol, confidence, ev, tier) "
                "VALUES (?, ?, ?, ?)",
                (symbol, confidence, ev, tier),
            )
            c.commit()
        if self.notifier:
            try:
                self.notifier.send(
                    "trade",
                    "PENDING APPROVAL: " + symbol + " (" + tier + ")",
                    "confidence=" + ("%.0f%%" % (confidence * 100)) +
                    " EV=" + ("%.1f%%" % (ev * 100)) +
                    " - approve in /monetisation UI",
                    meta={"symbol": symbol, "confidence": confidence,
                          "ev": ev, "tier": tier, "pending": True},
                )
            except Exception:
                pass

    def _pending_count(self):
        try:
            self._ensure_pending_table()
            with self._conn() as c:
                row = c.execute(
                    "SELECT COUNT(*) AS n FROM at_pending WHERE status = 'pending'"
                ).fetchone()
            return int(row["n"]) if row else 0
        except Exception:
            return 0

    def list_pending(self, limit=50):
        self._ensure_pending_table()
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, ts, symbol, confidence, ev, tier, status, resolved_ts "
                "FROM at_pending ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def approve_pending(self, pending_id):
        self._ensure_pending_table()
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM at_pending WHERE id = ? AND status = 'pending'",
                (pending_id,)
            ).fetchone()
        if not row:
            return {"error": "pending row not found or already resolved"}
        try:
            result = self.trader.execute_buy(row["symbol"], float(row["confidence"] or 0))
        except Exception as e:
            result = {"error": str(e)}
        success = isinstance(result, dict) and "error" not in result and             result.get("status") != "skipped"
        live = _is_live()
        qty = float(result.get("qty") or 0) if isinstance(result, dict) else 0
        with self._conn() as c:
            c.execute(
                "UPDATE at_pending SET status = ?, resolved_ts = NOW(), "
                "result_json = ? WHERE id = ?",
                ("approved" if success else "failed",
                 json.dumps(result)[:4000], pending_id),
            )
            c.execute(
                "INSERT INTO at_trades(symbol, side, qty, confidence, ev, tier, live, "
                "result_json) VALUES (?, 'buy', ?, ?, ?, ?, ?, ?)",
                (row["symbol"], qty, float(row["confidence"] or 0),
                 float(row["ev"] or 0), row["tier"], 1 if live else 0,
                 json.dumps(result)[:4000]),
            )
            c.commit()
        if success and self.notifier:
            try:
                self.notifier.trade({
                    "symbol": row["symbol"], "qty": qty,
                    "confidence": float(row["confidence"] or 0),
                    "ev": float(row["ev"] or 0),
                    "tier": row["tier"], "live": live, "approved": True,
                })
            except Exception:
                pass
        return {"status": "approved" if success else "failed", "result": result}

    def reject_pending(self, pending_id, reason="manual"):
        self._ensure_pending_table()
        with self._conn() as c:
            c.execute(
                "UPDATE at_pending SET status = 'rejected', resolved_ts = NOW(), "
                "result_json = ? WHERE id = ? AND status = 'pending'",
                (json.dumps({"reason": reason}), pending_id),
            )
            c.commit()
        return {"status": "rejected"}

    # ----- Daily P&L digest --------------------------------------------------
    def daily_summary(self):
        today = date.today().isoformat()
        self._ensure_tables_once()
        try:
            with self._conn() as c:
                s = c.execute("SELECT * FROM at_state WHERE id = 1").fetchone()
                trades = c.execute(
                    "SELECT COUNT(*) AS n FROM at_trades WHERE date(ts) = ?", (today,)
                ).fetchone()
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.warning("AutonomousTrader tables missing — reinitialising schema")
                try:
                    self._init_db()
                except Exception as ie:
                    logger.warning("_init_db failed: %s", ie)
                s = None
                trades = None
            else:
                raise
        equity = self._equity_safe()
        open_eq = s["today_open_eq"] if s else None
        pnl_pct = 0.0
        if open_eq and equity:
            pnl_pct = (equity - open_eq) / open_eq
        return {
            "date":         today,
            "tier":         _norm_tier(s["tier"]) if s else "conservative",
            "live":         _is_live(),
            "trades":       int(trades["n"]) if trades else 0,
            "deployed_pct": s["today_deployed_pct"] if s else 0,
            "win_rate_pct": None,
            "equity":       equity,
            "pnl_pct":      pnl_pct,
        }

    def send_daily_digest(self):
        summary = self.daily_summary()
        if self.notifier:
            try:
                self.notifier.digest(summary)
            except Exception as e:
                logger.debug("notifier.digest failed: %s", e)
        return summary

    # ----- Trade journal export ----------------------------------------------
    def export_journal_rows(self, days=30):
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        self._ensure_tables_once()
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, symbol, side, qty, confidence, ev, tier, live "
                "FROM at_trades WHERE ts >= ? ORDER BY ts ASC", (since,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ----- Prometheus-style metrics ------------------------------------------
    def metrics_text(self):
        s = self.status()
        caps = s.get("tier_caps", {})
        lines = [
            "# HELP dmai_trader_enabled 1 if loop enabled",
            "# TYPE dmai_trader_enabled gauge",
            "dmai_trader_enabled %d" % (1 if s.get("enabled") else 0),
            "# TYPE dmai_trader_live gauge",
            "dmai_trader_live %d" % (1 if s.get("live") else 0),
            "# TYPE dmai_trader_market_open gauge",
            "dmai_trader_market_open %d" % (1 if s.get("market_open") else 0),
            "# TYPE dmai_trader_trades_today gauge",
            "dmai_trader_trades_today %d" % (s.get("today_trades", 0) or 0),
            "# TYPE dmai_trader_deployed_pct gauge",
            "dmai_trader_deployed_pct %.6f" % float(s.get("today_deployed_pct", 0) or 0),
            "# TYPE dmai_trader_pending_count gauge",
            "dmai_trader_pending_count %d" % (s.get("pending_count", 0) or 0),
            "# TYPE dmai_trader_ev_gate gauge",
            "dmai_trader_ev_gate %.6f" % float(caps.get("ev_gate", 0) or 0),
            "dmai_trader_tier{tier=\"" + str(s.get("tier", "")) + "\"} 1",
        ]
        return "\n".join(lines) + "\n"


def get_autonomous_trader(db_path, trader, prediction_engine=None, notifier=None):
    return AutonomousTrader(db_path=db_path, trader=trader,
                            prediction_engine=prediction_engine,
                            notifier=notifier)
