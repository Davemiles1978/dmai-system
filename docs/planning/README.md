# DMAI / Aevora Collated Requirements — File Index

This drop consolidates 20 prior planning documents (12 Mac-side markdown files, 2 extracted docx references, 4 architecture attachments, 8 open GitHub issues, 30 merged PRs) plus a direct gap analysis against the live repo (`/tmp/dmai`, branch `main`) into five files. Read them in this order.

## 1. `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` — the full source of truth (human-readable)

The complete 13-section document. This is the one to actually read end-to-end if you want the full picture: vision, architecture consolidation (with contradictions flagged), capability inventory, verified ground-truth state (30 PRs + 8 issues), the unified roadmap, environment reference, cross-reference/lineage table, discrepancies, source manifest, the full 62-row gap analysis (Section 11), the gap summary with buildability split (Section 12), and the DMAI autonomous operating model — Workflow A (autonomous) vs. Workflow B (drafted-then-approved) — in Section 13.

**Read this first if:** you want the whole story, need to trace where a roadmap item came from, or want to understand what's actually built vs. aspirational vs. explicitly out of scope (dark-web/hacking-engine content — see Section 2 and Section 9).

## 2. `GITHUB_ISSUE_BODY.md` — condensed, paste-ready version (human-readable)

A ~230-line condensed version of the same material, formatted to paste directly as a GitHub issue body. Contains the executive summary, ground-truth state, the roadmap (Sections 6.1–6.4 only, condensed), the gap analysis summary + buildability split, the autonomous operating model summary, and a `supersedes: #208, #212, #213, #216, #218, #219, #220, #221 ?` line — the question mark is intentional; David hasn't confirmed this doc should replace those issues as the tracking surface yet.

**Read this if:** you're about to open the GitHub issue, or want the short version without the full lineage/discrepancy detail.

## 3. `self_gen_backlog.jsonl` — machine-readable backlog feed (computer-readable)

One JSON object per line, one row per NOT-STARTED/PARTIAL gap item from Section 11 of the master doc. Each row is a full spec: `id`, `name`, `capability_type`, `description`, `priority` (1 = blocks other work, 2 = standard, 3 = nice-to-have), `provenance: "gap_driven"`, `runtime_mode: "stub"`, `judge_confidence`, `target_kpi`, `acceptance_criteria`, `depends_on`, `source_docs`, `blocker`, `estimated_scope`, `notes`, `buildable_by_self_gen`, and `workflow` (`"A"` = autonomous, `"B"` = drafted-then-approved by a human).

**This is the file DMAI's self-generation loop is meant to ingest** — but it can't yet, because the ingestion endpoint doesn't exist. See item #4 below.

## 4. `self_gen_backlog_manifest.json` — backlog metadata (computer-readable)

Top-level counts for the JSONL file above: total items, breakdown by priority/type/scope, the `buildability_split` (how many items self-gen can build unassisted vs. need a human PR, and why), and the `workflow_split` (Workflow A vs. B counts, with both workflows defined inline). Also carries the `seed_command` field, which spells out the chicken-and-egg situation: the ingestion endpoint that would consume this file doesn't exist yet — building it is backlog item `gap_seed_backlog_ingestion_endpoint`, priority 1, workflow B (it touches `dmai_core_complete.py`, so it needs a human PR).

## How to actually use the JSONL once the seed endpoint exists

1. **Build the endpoint first** (`gap_seed_backlog_ingestion_endpoint` in the JSONL — a small, human-authored PR, ~50 LOC, adds `POST /api/admin/self-generation/seed-backlog` to `dmai_core_complete.py`).
2. **Call it** — `POST /api/admin/self-generation/seed-backlog?path=data/self_gen_backlog.jsonl` (copy the JSONL into the repo's `data/` directory first, or point the endpoint at wherever it's mounted). The endpoint reads each line and inserts it into the `capabilities` table as a `stub` row with `provenance='gap_driven'` and the row's `judge_confidence`.
3. **Watch the picker pick them up** — the existing fresh-blood → self-judge → capability-promoter → capability-materialiser chain (already live, PRs H through JJ) will start selecting these stubs on its normal cadence, same as any other `gap_driven` row.
4. **Two outcomes per item, depending on its `workflow` field:**
   - `workflow: "A"` (2 of 21 items today) — the materialiser writes the module straight to `components/generated/live/`, the verifier tests it, and if it passes it goes live with no further human step.
   - `workflow: "B"` (19 of 21 items today) — the change is drafted to a sandbox branch, tested there, and if it passes, DMAI opens a `[DMAI-DRAFT]`-titled PR and notifies the user. A human reviews and merges or closes it. This is the fix-proposer pattern from Issue #220, generalised to the whole `buildable_by_self_gen: no_*` bucket.
5. **Depend-on chains matter** — some rows have a non-empty `depends_on` list (e.g. the metric-contract audit depends on the seed endpoint existing first). The picker/planner should respect these when deciding what to surface next; nothing enforces this automatically yet.

## Quick reference: which file answers which question

| Question | File |
|---|---|
| "What is DMAI/Aevora and what's the current state?" | `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` §1–2 |
| "What's actually built vs. just planned?" | `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` §11 (Gap Analysis) |
| "Where did roadmap item X come from?" | `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` §8 (Cross-Reference Table) |
| "What do the docs disagree on?" | `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` §9 (Discrepancies) |
| "I need to paste something into GitHub right now" | `GITHUB_ISSUE_BODY.md` |
| "What should DMAI build next, autonomously or via PR?" | `self_gen_backlog.jsonl` (sort by `priority`, filter by `workflow`) |
| "How many items are self-gen-buildable vs. need a human?" | `self_gen_backlog_manifest.json` → `buildability_split` / `workflow_split` |
| "What's the target operating model for handing work to DMAI?" | `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` §13, or `GITHUB_ISSUE_BODY.md`'s "DMAI Autonomous Operating Model" section |

## Explicit scope note

Per the legal-hard-limits mandate stated in the master doc's Section 2: no Tor/dark-web infrastructure, and compliance with UK CMA, PoCA, UK GDPR, FCA, and Betfair's terms is non-negotiable for any bet/trade capability. Two historical planning docs (`11_DMAI_FINAL_HANDOVER_v8.md`, `12_HANDOVER_v7.md`) once proposed a "Dark Web Engine" and "Hacking Engine" — these are retired, explicitly excluded from every file in this drop, and confirmed never to have been built as functioning code (see Section 9, discrepancy #3, and the corresponding NOT STARTED rows in Section 11).
