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
        # chunk 10: best-effort import of any Layer 3 self-bootstrapping
        # seeder modules so their idempotent setup work runs on cold boot
        # without requiring a dmai_core_complete.py wiring change.
        try:
            import importlib
            for _mod in ("components.empty_tables_seeder",):
                try:
                    importlib.import_module(_mod)
                except Exception:
                    pass
        except Exception:
            pass

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

    def run_once(self, auto_approve: bool = False, **_ignored: Any) -> OrchestratorRunSummary:
        """Run a single gap->proposal->queue pass.

        In this incremental build, this method is safe and may return an empty
        run summary if gaps cannot be fetched yet.

        chunk 7.6: accept and ignore unknown kwargs (e.g. ``fresh=True`` passed
        by the /api/self-evolution/repair-gap route) so the public surface is
        forgiving of caller drift.
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

        # chunk 7.6: caller may pass fresh=True to indicate they expect a
        # full re-scan; current implementation always re-fetches so this is a
        # no-op, but we record it for parity with the route contract.
        _ = bool(_ignored.get("fresh", False))

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

            # chunk 10.2: SelfEditQueue.__init__ takes (data_path, notifier),
            # not repo_root. The earlier chunk-7.5 hotfix passed repo_root which
            # raised TypeError and was silently swallowed by the outer
            # try/except — explaining why every prior live run had zero
            # enqueued_edit_ids despite valid proposals.
            q = SelfEditQueue(data_path="data")
            # chunk 10.2: dedupe identical proposals by (file, hash). When 3
            # empty_tables gaps all fire the same seeder proposer, we only
            # want a single enqueue/approve/commit cycle.
            import hashlib as _hashlib
            _seen_props: set = set()
            for prop in proposals:
                _key = (prop.file, _hashlib.sha256((prop.new_snippet or "").encode()).hexdigest())
                if _key in _seen_props:
                    continue
                _seen_props.add(_key)
                # chunk 7.5: validate proposed code parses before enqueueing.
                # Refuse syntactically broken proposals at the orchestrator
                # boundary so they never reach apply-time.
                try:
                    import ast
                    ast.parse(prop.new_snippet or "")
                except SyntaxError:
                    continue

                # chunk 7.5: match real SelfEditQueue.enqueue signature
                # (capability, target_file, code, rationale).
                rationale = (
                    f"{prop.description} "
                    f"(confidence={prop.confidence:.2f}, "
                    f"proposed_by=self_repair_orchestrator)"
                )
                edit_id = q.enqueue(
                    capability="self_repair_orchestrator",
                    target_file=prop.file,
                    code=prop.new_snippet,
                    rationale=rationale,
                )
                enqueued.append(str(edit_id))
                if auto_approve and self._should_auto_approve(prop):
                    try:
                        res = q.approve(str(edit_id), decided_by="self_repair_orchestrator")
                        if isinstance(res, dict) and res.get("ok") is True:
                            auto_approved.append(str(edit_id))
                            # chunk 10: self-bootstrapping seeder modules
                            # execute their work on first import. After a
                            # successful auto-approve of such a module,
                            # import it in-process so the seed runs now
                            # instead of waiting for the next deploy.
                            try:
                                tgt = (prop.file or "").replace("\\", "/")
                                if tgt.startswith("components/") and tgt.endswith(".py"):
                                    mod_name = tgt[:-3].replace("/", ".")
                                    import importlib
                                    try:
                                        m = importlib.import_module(mod_name)
                                        importlib.reload(m)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        self.last_run = OrchestratorRunSummary(
            matched_patterns=matched,
            proposals=proposals,
            enqueued_edit_ids=enqueued,
            auto_approved_edit_ids=auto_approved,
        )
        return self.last_run

    # chunk 7.6: status surface for /api/self-evolution/repair-status.
    # The route expects a JSON-serialisable dict. Wrap the last run summary
    # and any queue history we can fetch defensively (the queue may not be
    # importable yet, or may not have history methods in early chunks).
    def status(self) -> Dict[str, Any]:
        """Return last-run summary plus queue snapshot (best-effort).

        Shape:
            {
              "ok": True,
              "last_run": {matched_patterns, proposals_count, enqueued,
                            auto_approved} | None,
              "queue": {pending: int, history: [...]} | None,
            }
        """

        last_run_payload: Optional[Dict[str, Any]] = None
        if self.last_run is not None:
            last_run_payload = {
                "matched_patterns": list(self.last_run.matched_patterns),
                "proposals_count": len(self.last_run.proposals),
                "enqueued_edit_ids": list(self.last_run.enqueued_edit_ids),
                "auto_approved_edit_ids": list(
                    self.last_run.auto_approved_edit_ids
                ),
            }

        queue_payload: Optional[Dict[str, Any]] = None
        try:
            from components.self_edit_queue import SelfEditQueue

            # chunk 10.2: same fix as in run_once().
            q = SelfEditQueue(data_path="data")
            pending: List[Any] = []
            history: List[Any] = []
            for attr in ("pending", "list_pending", "get_pending"):
                fn = getattr(q, attr, None)
                if callable(fn):
                    try:
                        pending = list(fn() or [])
                        break
                    except Exception:
                        pass
            for attr in ("history", "recent", "list_recent"):
                fn = getattr(q, attr, None)
                if callable(fn):
                    try:
                        history = list(fn() or [])[:20]
                        break
                    except Exception:
                        pass
            queue_payload = {
                "pending_count": len(pending),
                "history": history,
            }
        except Exception:
            queue_payload = None

        return {
            "ok": True,
            "last_run": last_run_payload,
            "queue": queue_payload,
        }
