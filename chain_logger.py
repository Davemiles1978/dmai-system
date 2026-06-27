"""
DMAI Chain Logger
=================
Step-by-step JSON logging for reasoning chains.

Each chain execution is logged as a structured JSON trace with:
  - chain_id (UUID)
  - start_time, end_time, duration_ms
  - steps: list of {step_num, action, input, output, tool, duration_ms, timestamp}
  - final_result
  - metadata: {model, prompt_tokens, total_steps, success}

Usage:
    logger = ChainLogger(data_path=Path("data/"))
    cid = logger.start_chain({"model": "gpt-4o"})
    logger.log_step(cid, "reasoning", input_data="user query")
    logger.log_step(cid, "tool_call", tool="search_web", output_data="results")
    trace = logger.complete_chain(cid, result="final answer", success=True)

Flask decorator:
    @log_chain(metadata={"endpoint": "/api/chat"})
    def my_route():
        ...
"""

import functools
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum characters kept for input/output summaries in each step
_SUMMARY_MAX_LEN = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(value: Any, max_len: int = _SUMMARY_MAX_LEN) -> str:
    """Truncate any value to a string no longer than max_len characters."""
    s = str(value) if value is not None else ""
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using temp file + os.replace() pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp",
        delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# ChainStep
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """
    Represents a single step within a reasoning chain trace.

    Fields:
        step_num        -- 1-based index of this step in the chain.
        action          -- Label such as "tool_call", "reasoning", "decision", "halt".
        tool            -- Name of the tool called, if applicable (empty string otherwise).
        input_summary   -- Truncated representation of the step input (max 200 chars).
        output_summary  -- Truncated representation of the step output (max 200 chars).
        duration_ms     -- Wall-clock duration of this step in milliseconds.
        timestamp       -- ISO 8601 UTC timestamp when this step was recorded.
    """

    step_num: int
    action: str
    tool: str = ""
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        """Serialise this step to a plain dict for JSON output."""
        return {
            "step_num":       self.step_num,
            "action":         self.action,
            "tool":           self.tool,
            "input_summary":  self.input_summary,
            "output_summary": self.output_summary,
            "duration_ms":    round(self.duration_ms, 2),
            "timestamp":      self.timestamp,
        }


# ---------------------------------------------------------------------------
# ChainTrace
# ---------------------------------------------------------------------------

@dataclass
class ChainTrace:
    """
    Full trace of a single reasoning chain execution.

    Fields:
        chain_id     -- UUID identifying this chain (set at construction).
        start_time   -- ISO 8601 UTC string recorded when the chain started.
        end_time     -- ISO 8601 UTC string recorded when complete() is called.
        duration_ms  -- Total wall-clock duration in milliseconds.
        steps        -- Ordered list of ChainStep dicts.
        final_result -- Truncated final answer or result string.
        success      -- Whether the chain completed successfully.
        metadata     -- Arbitrary key/value dict (model, prompt_tokens, etc.).
    """

    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str = field(default_factory=_now_iso)
    end_time: str = ""
    duration_ms: float = 0.0
    steps: List[dict] = field(default_factory=list)
    final_result: str = ""
    success: bool = False
    metadata: dict = field(default_factory=dict)

    # Internal wall-clock start for duration calculation
    _wall_start: float = field(default_factory=time.monotonic, repr=False, compare=False)

    def add_step(
        self,
        action: str,
        tool: str = "",
        input_data: Any = None,
        output_data: Any = None,
    ) -> "ChainTrace":
        """
        Append a new step to this trace and return self for chaining.

        The step timestamp is captured at call time. Duration of the step
        is computed as the difference from the previous step (or chain start).
        """
        step_num = len(self.steps) + 1
        ts = _now_iso()

        # Approximate step duration: difference from last step's wall time
        step_wall = time.monotonic()
        if self.steps:
            prev_ms = self.steps[-1].get("duration_ms", 0.0)
            # We track elapsed from chain start; each step gets incremental ms
            elapsed_ms = (step_wall - self._wall_start) * 1000.0
            step_dur_ms = elapsed_ms - sum(
                s.get("duration_ms", 0.0) for s in self.steps
            )
        else:
            step_dur_ms = (step_wall - self._wall_start) * 1000.0

        step = ChainStep(
            step_num=step_num,
            action=action,
            tool=tool,
            input_summary=_truncate(input_data),
            output_summary=_truncate(output_data),
            duration_ms=max(0.0, step_dur_ms),
            timestamp=ts,
        )
        self.steps.append(step.to_dict())
        return self

    def complete(self, result: Any = None, success: bool = True) -> "ChainTrace":
        """Mark chain complete, compute total duration, and record final result."""
        self.end_time = _now_iso()
        self.duration_ms = (time.monotonic() - self._wall_start) * 1000.0
        self.final_result = _truncate(result, max_len=500)
        self.success = success
        self.metadata["total_steps"] = len(self.steps)
        return self

    def to_dict(self) -> dict:
        """Serialise this trace to a plain dict for JSON output."""
        return {
            "chain_id":     self.chain_id,
            "start_time":   self.start_time,
            "end_time":     self.end_time,
            "duration_ms":  round(self.duration_ms, 2),
            "steps":        self.steps,
            "final_result": self.final_result,
            "success":      self.success,
            "metadata":     self.metadata,
        }


# ---------------------------------------------------------------------------
# ChainLogger
# ---------------------------------------------------------------------------

class ChainLogger:
    """
    Singleton-style chain logger. Writes JSON traces to data/chain_logs/.

    Thread-safety note: this implementation is single-process. For
    multi-process deployments, use an external log aggregator.
    """

    LOG_DIR = "chain_logs"
    MAX_LOG_FILES = 500  # rotate (delete oldest) after this many per day-dir

    def __init__(self, data_path: Path):
        """Initialise the chain logger, creating the log directory."""
        self.log_dir = Path(data_path) / self.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._active: Dict[str, ChainTrace] = {}  # chain_id -> ChainTrace

    def start_chain(self, metadata: Optional[dict] = None) -> str:
        """
        Start a new chain and return its chain_id.
        Optionally attach metadata (e.g. model name, endpoint, prompt_tokens).
        """
        trace = ChainTrace(metadata=dict(metadata or {}))
        self._active[trace.chain_id] = trace
        logger.debug("ChainLogger: started chain %s", trace.chain_id)
        return trace.chain_id

    def log_step(
        self,
        chain_id: str,
        action: str,
        tool: str = "",
        input_data: Any = None,
        output_data: Any = None,
    ) -> None:
        """
        Append a step to the active chain identified by chain_id.
        Silently ignores unknown chain_ids with a WARNING log.
        """
        trace = self._active.get(chain_id)
        if trace is None:
            logger.warning(
                "ChainLogger.log_step: unknown chain_id '%s'", chain_id
            )
            return
        trace.add_step(
            action=action, tool=tool,
            input_data=input_data, output_data=output_data,
        )

    def complete_chain(
        self, chain_id: str, result: Any = None, success: bool = True
    ) -> Optional[ChainTrace]:
        """
        Complete the chain, write its JSON log file, and return the trace.
        Returns None if the chain_id is unknown.
        """
        trace = self._active.pop(chain_id, None)
        if trace is None:
            logger.warning(
                "ChainLogger.complete_chain: unknown chain_id '%s'", chain_id
            )
            return None
        trace.complete(result=result, success=success)
        self._write_trace(trace)
        logger.debug(
            "ChainLogger: completed chain %s (%d steps, %.0fms, success=%s)",
            chain_id, len(trace.steps), trace.duration_ms, success,
        )
        return trace

    def get_recent(self, n: int = 20) -> List[dict]:
        """
        Return the n most-recent completed chain traces as dicts,
        read from log files (newest first).
        """
        files: List[Path] = sorted(
            self.log_dir.rglob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        results: List[dict] = []
        for f in files[:n]:
            try:
                results.append(json.loads(f.read_text()))
            except Exception as exc:
                logger.warning("ChainLogger.get_recent: could not read %s — %s", f, exc)
        return results

    def _write_trace(self, trace: ChainTrace) -> None:
        """
        Atomically write the trace to log_dir/YYYY-MM-DD/<chain_id>.json.
        Rotates (deletes oldest) if MAX_LOG_FILES is exceeded in the day dir.
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir  = self.log_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # Rotate if needed
        existing = sorted(day_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(existing) >= self.MAX_LOG_FILES:
            try:
                existing.pop(0).unlink()
            except Exception:
                break

        out_path = day_dir / (trace.chain_id + ".json")
        _atomic_write_json(out_path, trace.to_dict())
        logger.debug("ChainLogger: wrote trace to %s", out_path)


# ---------------------------------------------------------------------------
# Flask decorator
# ---------------------------------------------------------------------------

def log_chain(metadata: Optional[dict] = None):
    """
    Decorator that wraps a Flask route function in a chain trace.

    The chain is started before the route executes and completed after.
    The route's return value is used as the final_result (truncated).
    Any exception marks the chain as failed and re-raises.

    Usage:
        @app.route("/api/chat", methods=["POST"])
        @log_chain(metadata={"endpoint": "/api/chat"})
        def chat():
            ...
    """
    def decorator(fn):
        """Inner decorator that wraps the Flask route function."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Wrapper that starts, logs, and completes a chain around the route."""
            # Lazy import so this module works without Flask installed
            try:
                from flask import g, current_app  # type: ignore
                _has_flask = True
            except ImportError:
                _has_flask = False

            # Obtain ChainLogger from Flask app extensions if available,
            # otherwise create a temporary one in the current directory.
            chain_log: Optional[ChainLogger] = None
            if _has_flask:
                try:
                    chain_log = current_app.extensions.get("chain_logger")
                except RuntimeError:
                    pass  # outside application context
            if chain_log is None:
                chain_log = ChainLogger(data_path=Path("data"))

            meta = dict(metadata or {})
            meta.setdefault("route", fn.__name__)
            cid = chain_log.start_chain(meta)

            chain_log.log_step(cid, "request", input_data={"args": args, "kwargs": kwargs})

            try:
                response = fn(*args, **kwargs)
                chain_log.log_step(cid, "response", output_data=str(response)[:200])
                chain_log.complete_chain(cid, result=str(response)[:200], success=True)
                return response
            except Exception as exc:
                chain_log.log_step(cid, "error", output_data=str(exc))
                chain_log.complete_chain(cid, result=str(exc), success=False)
                raise

        return wrapper
    return decorator
