"""Layer 3 Self-Repair Orchestrator.

Chunk 6 introduces auto-approve guardrails.

This module is intentionally small and dependency-light so it can be imported in
startup flows without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from components.repair_patterns import FixProposal, PATTERNS, is_critical_file


@dataclass
class OrchestratorRunSummary:
    matched_patterns: List[str]
    proposals: List[FixProposal]
    enqueued_edit_ids: List[str]
    auto_approved_edit_ids: List[str]


class SelfRepairOrchestrator:
    """Bridges gap entries to concrete edits queued in SelfEditQueue."""

    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = repo_root
        self.last_run: Optional[OrchestratorRunSummary] = None

    def _patch_line_count(self, proposal: FixProposal) -> int:
        """Approximate patch size by counting changed lines in snippets."""

        old_lines = proposal.original_snippet.splitlines() if proposal.original_snippet else []
        new_lines = proposal.new_snippet.splitlines() if proposal.new_snippet else []
        # Conservative: count the larger side as the patch magnitude.
        return max(len(old_lines), len(new_lines))

    def _should_auto_approve(self, proposal: FixProposal) -> bool:
        """Auto-approve guardrails.

        Rules (fail-closed for critical files only):
        - confidence >= 0.85
        - patch < 30 lines
        - file is not a critical file
        """

        if proposal is None:
            return False
        if proposal.confidence < 0.85:
            return False
        if self._patch_line_count(proposal) >= 30:
            return False
        if is_critical_file(proposal.file):
            return False
        return True

    def run_once(self, auto_approve: bool = False) -> OrchestratorRunSummary:
        """Run a single gap->proposal->queue pass.

        In this incremental build, this method is safe and may return an empty
        run summary if gaps cannot be fetched yet.
        """

        matched: List[str] = []
        proposals: List[FixProposal] = []
        enqueued: List[str] = []
        auto_approved: List[str] = []

        try:
            # Gap fetcher is introduced in earlier chunks; keep imports lazy.
            from components.gap_fetcher import iter_gap_entries

            gap_entries = list(iter_gap_entries())
        except Exception:
            gap_entries = []

        for gap in gap_entries:
            for pat in PATTERNS:
                try:
                    if not pat.detect(gap):
                        continue
                    matched.append(pat.name)
                    proposal = pat.propose(gap, self.repo_root)
                    if isinstance(proposal, FixProposal):
                        proposals.append(proposal)
                except Exception:
                    continue

        # Queue wiring may not exist yet; keep this safe.
        try:
            from components.self_edit_queue import SelfEditQueue

            q = SelfEditQueue(repo_root=self.repo_root)
            for prop in proposals:
                edit_id = q.enqueue(
                    file_path=prop.file,
                    description=prop.description,
                    original_snippet=prop.original_snippet,
                    new_snippet=prop.new_snippet,
                    proposed_by="self_repair_orchestrator",
                    confidence=prop.confidence,
                    meta=prop.meta or {},
                )
                enqueued.append(str(edit_id))
                if auto_approve and self._should_auto_approve(prop):
                    # Auto-commit wire is added in chunk 7.
                    auto_approved.append(str(edit_id))
        except Exception:
            pass

        self.last_run = OrchestratorRunSummary(
            matched_patterns=matched,
            proposals=proposals,
            enqueued_edit_ids=enqueued,
            auto_approved_edit_ids=auto_approved,
        )
        return self.last_run
