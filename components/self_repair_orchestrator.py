"""Layer 3: SelfRepairOrchestrator

Owns the "gap -> proposal -> enqueue" loop.

Chunk 3 scope:
- Provide a safe-to-import orchestrator with a run_once() method.
- It does NOT start a background thread yet.
- It is wired into dmai_core_complete.py init under try/except and records
  init failures to _STARTUP_ERRORS.

The orchestrator fetches gaps in-process (no HTTP loopback) and matches them
against the repair pattern registry. For each match, it creates a FixProposal
and enqueues the proposed code via SelfEditQueue.

Auto-approve and auto-commit are explicitly out-of-scope until later chunks.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.gap_fetcher import GapEntry, fetch_gaps
from components.repair_patterns import FixProposal, PATTERNS
from components.self_edit_queue import SelfEditQueue

logger = logging.getLogger(__name__)


class SelfRepairOrchestrator:
    def __init__(
        self,
        repo_root: str | Path = ".",
        data_path: str | Path = "data",
        notifier=None,
    ):
        self.repo_root = str(repo_root)
        self.data_path = str(data_path)
        self.notifier = notifier
        self.queue = SelfEditQueue(data_path=self.data_path, notifier=notifier)
        self.last_run: Dict[str, Any] = {}

    def _entry_to_gap_dict(self, entry: GapEntry) -> Dict[str, Any]:
        d = dict(entry.payload or {})
        # preserve category for pattern detection
        d.setdefault("kind", entry.category)
        return d

    def run_once(self, fresh: bool = True) -> Dict[str, Any]:
        """Run one evaluation pass.

        Returns a structured summary suitable for an API response.
        """
        entries, raw = fetch_gaps(fresh=fresh)

        summary: Dict[str, Any] = {
            "ok": True,
            "gaps_count": len(entries),
            "matches": [],
            "enqueued": [],
        }

        for entry in entries:
            gap_dict = self._entry_to_gap_dict(entry)
            for pattern in PATTERNS:
                try:
                    if not pattern.detect(gap_dict):
                        continue
                    proposal_obj = pattern.propose(gap_dict, self.repo_root)
                    if proposal_obj is None:
                        summary["matches"].append({
                            "pattern": pattern.name,
                            "gap": gap_dict,
                            "proposal": None,
                        })
                        continue

                    # Normalize to FixProposal
                    if isinstance(proposal_obj, FixProposal):
                        proposal = proposal_obj
                    elif isinstance(proposal_obj, dict):
                        proposal = FixProposal(**proposal_obj)  # type: ignore
                    else:
                        summary["matches"].append({
                            "pattern": pattern.name,
                            "gap": gap_dict,
                            "proposal": None,
                            "error": f"unexpected proposal type: {type(proposal_obj).__name__}",
                        })
                        continue

                    summary["matches"].append({
                        "pattern": pattern.name,
                        "gap": gap_dict,
                        "proposal": asdict(proposal),
                    })

                    # Enqueue full-file content (new_snippet is expected to be the complete file text)
                    edit_id = self.queue.enqueue(
                        capability=f"layer3:{pattern.name}",
                        target_file=proposal.file,
                        code=proposal.new_snippet,
                        rationale=proposal.description,
                    )
                    summary["enqueued"].append({
                        "id": edit_id,
                        "pattern": pattern.name,
                        "file": proposal.file,
                        "confidence": proposal.confidence,
                    })

                except Exception as e:
                    logger.warning("SelfRepairOrchestrator pattern %s failed: %s", pattern.name, e)
                    summary.setdefault("errors", []).append({
                        "pattern": pattern.name,
                        "error": str(e),
                    })

        self.last_run = {
            "summary": summary,
            "raw": raw if isinstance(raw, dict) else {},
        }
        return summary

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "last_run": self.last_run.get("summary") or {},
            "pending": self.queue.pending(),
            "history": self.queue.history(limit=25),
        }
