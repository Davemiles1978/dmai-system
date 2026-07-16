# Unified DMAI system requirements and roadmap — 2026-07-16 snapshot

**Full detail:** see `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md` in the same drop (10 sections + gap analysis + gap summary). This issue is the condensed, paste-ready version.

**supersedes: #208, #212, #213, #216, #218, #219, #220, #221 ?** *(question mark intentional — not yet confirmed by David; do not close those issues until he confirms this doc should replace them as the tracking surface.)*

---

## Executive Summary

DMAI (Dynamic Meta-Adaptive Intelligence) is a self-hosted, self-evolving AI system on Render with hybrid SQLite/Postgres persistence; **Aevora** is the evolving external/commercial name for the same lineage, currently scoped as a bounded, human-approved starter product rather than the full autonomous system. As of 2026-07-15/16: ~29% learning coverage (carried forward, not independently re-verified this pass), the self-generation loop was **just unblocked via PRs GG→JJ** (schema migration + DB-lock fix), 20,694+ capability rows are referenced in code/diagnostics, **zero live generated modules** exist yet in `components/generated/live/`, and nightly R2 backups of SQLite + Postgres are running (PR P). Full vision, architecture consolidation, and legal-scope caveats (no dark-web/hacking-engine work — see full doc §2 and §9) are in the companion document.

---

## Ground-Truth State

**30 PRs merged 2026-07-14 → 2026-07-15** (H through JJ), grouped:
- **Self-gen loop:** H, W, X, Y/Y-fix/Y-fix-2, Z, BB, CC, DD, EE/EE-fix, FF, GG/GG-hotfix, HH, II, JJ
- **Infra/procurement/treasury:** I, J, K, K.1, L
- **Backups/migrations/monitoring:** M, N, O, R, R.1, T, P, V-fast

**8 open issues:**
| # | Title | Status |
|---|---|---|
| #208 | 3D knowledge graph visualisation | Deferred until self-gen loop stable 48h |
| #212 | Bet/Trade training tracker table | Not started |
| #213 | Self-generation roadmap — goal-directed planner | Not started; gated on ≥20 live modules/3d + #212 |
| #216 | Weekly training refresh sweep | Not started |
| #218 | Self-gen scope boundary + suggest-a-fix + consistency cron | Rule established; follow-ups not built |
| #219 | Metric Contract Audit (self-healing) | Not started; blocked on #221 |
| #220 | Fix-Proposer Loop | Not started; depends on #218 (done) + #219 (not started) |
| #221 | First-widget-manifest walkthrough | Not started; blocks #219 |

**Self-gen loop status:** unblocked but not yet producing live modules. Root cause: (1) capabilities table schema mismatch → **fixed by PR HH**; (2) DB lock contention → **fixed by PR JJ** via `safe_open_kdb`; (3) verification endpoint shipped by PR II (`/api/admin/self-generation/force-tick`, confirmed present). **Next step: verify a force-tick actually produces a real module** — `components/generated/live/` is still empty as of this snapshot.

---

## Unified Roadmap (condensed)

### 6.1 Immediate (next 7 days) — self-gen productivity
- [ ] Verify force-tick actually generates a real module post-PR JJ
- [ ] SI 67% vs 0.005% dashboard mismatch (scale bug — fraction rendered as %; still unfixed)
- [ ] Empty-queue diagnosis re-verification (partially fixed by PR DD + PR HH)
- [ ] Fresh-blood + capability-promoter `get_*_status` export bug (found during PR GG diagnostic)
- [ ] Capability value gate (new — evaluation/gap-analysis system; only integrate capabilities with measurable system improvement, permanently mark/remove the rest)
- [ ] **Build `/api/admin/self-generation/seed-backlog` endpoint** (one small PR, ~50 LOC) — reads `self_gen_backlog.jsonl` (companion machine-readable feed, see below) and inserts each row into the `capabilities` table as a `stub` with correct `provenance`/`judge_confidence`/`runtime_mode`. This is the chicken-and-egg bootstrap: one human-written endpoint, then self-gen takes over ingesting the rest of the backlog.
- [ ] **Ingest `self_gen_backlog.jsonl` into the capabilities table** via the endpoint above once built.
- [ ] **Watch the first ticks pick + materialise from the backlog** — confirms the seeded rows actually flow through fresh-blood → judge → promoter → materialiser → live.

### 6.2 Short-term (this month) — hardening
- [ ] Bet/Trade training tracker (#212)
- [ ] Goal-directed roadmap planner (#213)
- [ ] 3D knowledge graph (#208)
- [ ] Weekly training refresh sweep (#216)
- [ ] PR V-real Postgres migration (full hot-table cutover; migration endpoint already shipped via PR R/R.1)
- [ ] Schema bootstrap warnings (needs direct log inspection — not re-verified this pass)
- [ ] PR "U" logs endpoint / PR "Q" checkpoints — **numbering gap**, neither appears in the 30 merged-PR list; needs GitHub history check beyond the most-recent-30 window
- [ ] Perplexity key regen (not independently re-verified this pass)
- [ ] Cron migrations / consolidation audit
- [ ] Self-healing scope boundary (#218), metric contract audit (#219), fix-proposer loop (#220), manifest walkthrough (#221)
- [ ] White-hat pentest integration (deferred; "hackingtool-plugin substrate" — **must be scoped strictly defensive/own-infra-only**, not conflated with the retired Dark Web/Hacking Engine concepts from handover v7/v8, which are explicitly out of scope — see full doc §2, §9)

### 6.3 Medium-term (this quarter) — self-gen goal-directed
- [ ] Goal-directed planner (#213) — picker quota integration (`roadmap_driven` pool)
- [ ] Roadmap engine DMAI itself owns (`GET /api/self-generation/roadmap-progress`)
- [ ] 20+ verified live modules milestone (currently 0)
- [ ] Revenue-gen stream go-live (bet/trade) — gated on #212 + #216

### 6.4 Long-term / vision
- [ ] Full AGI/SI targets from `04_DMAI_AUTONOMY_ROADMAP.md` (surface-web only — dark-web excluded)
- [ ] Dual Recovery Engine architecture (never built)
- [ ] Identity/financial accounts, cloud propagation, stealth/camouflage (claimed "complete" in handover v8 — **contradicted by repo evidence**, see full doc §9 discrepancy #2)
- [ ] Hall of Fame, 30 Base Systems, Investment Engine, 60/40 split, extended income streams, hardware phase, biometric security — all aspirational, mostly unbuilt
- [ ] **Dark Web Engine / Hacking Engine — explicitly excluded from vision going forward.** Historical record only; must not be actioned (legal-hard-limits mandate: no Tor/dark-web, UK CMA/PoCA/GDPR/FCA/Betfair compliance).

---

## Gap Analysis Summary

Full 62-row table with per-item evidence is in the companion doc, §11. Ordered NOT STARTED → PARTIAL → DONE, cross-checked against `/tmp/dmai` (branch `main`, HEAD `241dc07`), 30 merged PRs, and 8 open issues.

**Counts by status:** NOT STARTED = 27 · PARTIAL = 10 · DONE = 20 · UNKNOWN = 5 *(total rows = 62; some capabilities span more than one row, e.g. self-gen loop has both a DONE row for the built chain and a PARTIAL row for its still-unrealized live-module output)*

**Top 10 highest-leverage NOT STARTED items:**
1. Fresh-blood / capability-promoter `get_*_status` exports (tiny fix, blocks accurate dashboard reporting)
2. SI 67% vs 0.005% scale-mismatch fix (tiny fix, actively misleading right now)
3. Metric contract manifest + audit (#219, #221) — unlocks the whole self-healing pipeline
4. Capability value gate — addresses the 20,694-row duplication problem
5. Bet/Trade training tracker (#212) — hard gate on any revenue go-live claim
6. Weekly training refresh sweep (#216) — hard gate on any "training complete" claim
7. Fix-Proposer Loop (#220) — the actual payoff of the self-healing investment
8. Goal-directed roadmap planner (#213) — turns self-gen from reactive to convergent
9. Biometric security — zero progress since March 2026
10. Consistency-assertion cron (#218 follow-up) — would have auto-caught the SI-score and training-panel bugs

**Top 5 PARTIAL items closest to done:**
1. Self-generation loop end-to-end — fully wired; only the force-tick verification is outstanding
2. Bet/trade advisory capability — substantive code exists; blocked only by #212 + #216 gating
3. Self-gen scope boundary enforcement — policy documented; needs a lightweight automated guard
4. Knowledge graph 3D-ready data layer — 2D live; 3D deferred by design, not blocked technically
5. PR V-real Postgres cutover — migration endpoint already shipped; remaining work is execution, not engineering

### Buildability (self-gen vs. human-PR split)

Across the 62 gap-analysis rows, classified by whether DMAI's self-generation loop could build the item itself under current house rules (net-new modules under `components/generated/live/` only; no edits to `dmai_core_complete.py`/UI/config; no new secrets; no modification of existing code semantics):

| Classification | Count | Meaning |
|---|---|---|
| `yes` | 8 | Self-gen can build unassisted |
| `no_touches_main_app` | 12 | Needs a `dmai_core_complete.py` change — off-limits to self-gen |
| `no_touches_ui` | 1 | Needs a `static/*.html`/JS change |
| `no_needs_secrets` | 6 | Needs a new API key/credential |
| `no_bug_fix` | 10 | Needs to modify existing semantics — self-gen only adds net-new |
| `no_infra` | 15 | Needs DB migration, config, or deploy change |
| `partial` | 10 | Part self-gen-able, wiring/integration is human |

**Reading:** only **8 of 62 items (13%)** are cleanly self-gen-buildable under the current, correctly-conservative #218 scope boundary. The other 87% need a human-authored PR. This is expected — it confirms the scope boundary is containing blast radius — but it also means most of this roadmap will not clear itself; self-gen contributes at the margins (new standalone modules, content-gen pieces, monitoring/tracking utilities) while humans drive the rest.

---

## DMAI Autonomous Operating Model (target state)

**User's stated target (verbatim, 2026-07-16):** *"set DMAI self-generation to produce all that it can. As system progresses i'd like to be able to hand nearly all off to DMAI to heal, create, or fix. Anything created for she cannot do should create and test fix, then ask user permission to apply, assuming fix correct."*

Every backlog item is tagged with a `workflow`:

- **Workflow A — Autonomous** (`buildable_by_self_gen == "yes"`): materialiser picks the stub → codegen writes the module under `components/generated/live/` → verifier tests it → if verified, promoted to live automatically, no human step. If it fails, quarantined + auto-retried (existing PR CC behaviour).
- **Workflow B — Drafted-then-approved** (`buildable_by_self_gen` starts with `no_`): same picker/queue, but codegen writes the change to a **sandbox branch only** → sandbox tests run (the Issue #220 fix-proposer pattern, generalised beyond just metric-audit anomalies) → if green, DMAI opens a `[DMAI-DRAFT]`-titled PR with description, test evidence, and a confidence score → in-app notification to the user ("Draft fix ready for review: PR #NNN") → user merges (approve) or closes (reject) → DMAI records the outcome to improve future confidence scoring. The merge button always stays human.

Current backlog split: **2 items Workflow A, 19 items Workflow B** (see `self_gen_backlog_manifest.json`'s `workflow_split`). As the Fix-Proposer Loop (#220) itself lands, more of the Workflow-B bucket gets purpose-built tooling — the goal per the user's stated model is to maximize what DMAI can propose and prove unassisted, while keeping final apply/merge decisions with the user for anything touching the main app, UI, infra, secrets, or existing code semantics. Full detail in the companion doc's §13.

---

## Machine-Readable Companion Feed

Two companion files carry the NOT STARTED / PARTIAL gap items as a self-gen-ingestable backlog:
- `self_gen_backlog.jsonl` — one JSON object per gap item, schema includes `id`, `capability_type`, `priority` (1–3), `provenance: "gap_driven"`, `runtime_mode: "stub"`, `judge_confidence`, `target_kpi`, `acceptance_criteria`, `depends_on`, `source_docs`, `estimated_scope`, and `buildable_by_self_gen`.
- `self_gen_backlog_manifest.json` — top-level metadata: counts by priority, type, scope, and buildability split.

These are meant to be ingested via the new `/api/admin/self-generation/seed-backlog` endpoint (priority-1 item in the backlog itself — see 6.1 above).

---

## Related

- Cross-links: #208, #212, #213, #216, #218, #219, #220, #221 (all referenced above; see "supersedes" line at top)
- Full source manifest, architecture consolidation, environment-variable reference, and discrepancy log: `DMAI_COLLATED_REQUIREMENTS_AND_ROADMAP.md`
