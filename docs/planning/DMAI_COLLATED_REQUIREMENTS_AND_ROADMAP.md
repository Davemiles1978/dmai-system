# DMAI / Aevora — Collated System Requirements and Roadmap
_Snapshot: 2026-07-16. Merges every prior planning doc into one source of truth._

> **Sources merged:** 12 Mac-side markdown planning docs, 2 extracted docx references, 4 architecture attachments, 8 open GitHub issues, 30 merged PRs, and verified repo state at `/tmp/dmai` (branch `main`, HEAD `241dc07`). Full manifest in [Section 10](#10-appendix-source-manifest).

---

## 1. Executive Summary

DMAI (Dynamic Meta-Adaptive Intelligence) is a self-hosted, self-evolving AI system running on Render with a hybrid SQLite/Postgres persistence layer; **Aevora** is the evolving external/commercial-facing name for the same lineage, first appearing as a bounded, human-approved "starter package" product (newsletter capture + paid audit offer) rather than the full autonomous system (`02_Aevora_setup-guide.md`, which explicitly states: *"This package is a bounded startup kit, not a live autonomous AGI controller"*). The project has moved through multiple planning eras — from a March 2026 "13 local processes + Render Gen 5" foundation (`1_DMAI_MASTER.md`, `03_DMAI_MASTER_PLAN_20260306.md`), through handover v7/v8's "One Consciousness" framing (`11_DMAI_FINAL_HANDOVER_v8.md`), to the current July 2026 production system with a self-generation loop. As of 2026-07-15/16: the system holds roughly **29% learning coverage** (per prior session context — not independently re-verified against a single authoritative endpoint in this pass; see [Section 9](#9-discrepancies--open-questions)), the self-generation loop was **just unblocked on 2026-07-15 via PRs GG through JJ** (schema migration + DB-lock fix), there are **20,694+ capability rows** referenced in code/diagnostics (the live checked-out DB currently shows 1,544 rows post-reset — see discrepancy notes), **zero live generated modules** exist yet in `components/generated/live/`, and **nightly R2 backups of SQLite + Postgres are running** (PR P, merged 2026-07-15).

---

## 2. System Vision

**What the finished system does**, per the aspirational sections of the Master docs, the Autonomy Roadmap, and the DMAI manifesto (`14_DMAI.txt`, extracted from 🧬 DMAI.docx):

- **Autonomous AI/SI with self-evolution**: "DMAI can improve ANY part of herself... There are no restrictions on her evolution. She can rewrite her own core code, evolve her consciousness architecture, add new capabilities, remove obsolete functions, optimize her thinking process" (`11_DMAI_FINAL_HANDOVER_v8.md`).
- **Self-repair**: a self-healing loop (metric-contract audit → fix proposer → sandbox prove → human PR approval) that detects and proposes fixes for its own semantic/runtime drift, formalized in issues [#218](#5-ground-truth-state-verified-as-of-2026-07-152026-07-16), [#219](#5-ground-truth-state-verified-as-of-2026-07-152026-07-16), [#220](#5-ground-truth-state-verified-as-of-2026-07-152026-07-16).
- **Revenue generation** via bet/trade advisory (`components/monetisation/betting_advisor.py`, `components/wealth/autonomous_trader.py`) and content/growth automation (`GrowthAutomationArchitecture-4.md`: content generation, social media automation, referral/affiliate systems, influencer outreach).
- **"No artificial restrictions on capability creation" principle** — per the user's own framing carried across `11_DMAI_FINAL_HANDOVER_v8.md` and `14_DMAI.txt`, self-generation should not be artificially capped.

  **⚠️ Legal-hard-limits caveat (explicit, non-negotiable, per current project mandate):** this principle does **not** extend to dark-web operations, Tor/anonymity infrastructure, hacking/exploitation tooling, or any activity that would breach UK law. Specifically **excluded from the vision, regardless of what any historical planning doc says:**
  - No Tor / dark-web access, scraping, or "camouflage" infrastructure.
  - Must comply with the **UK Competition and Markets Authority (CMA)** regime, the **Proceeds of Crime Act (PoCA)**, **UK GDPR**, the **Financial Conduct Authority (FCA)** regime, and **Betfair's** terms of service for any betting/trading capability.
  - Historical docs (`11_DMAI_FINAL_HANDOVER_v8.md`, `12_HANDOVER_v7.md`) describe a "Dark Web Engine" (scraping, anonymous consulting, data markets, exploit brokerage, crypto laundering) and a "Hacking Engine" (network scanning, exploit development, password cracking, social engineering, backdoor creation, ransomware, cryptojacking, botnets, zero-day discovery) as "NEED CODE" backlog items. **These are explicitly out of scope going forward** and are flagged in [Section 9](#9-discrepancies--open-questions) as content that must not be carried into any live roadmap. Verification against the current repo (`/tmp/dmai`) confirms these were never built as functioning capabilities — only inert, always-`None` constructor parameters exist in `components/phase5/P5_SelfFunding.py` and `components/phase6/P6_AdvancedIntelligence.py`.

**"The Four Pillars"** (from `14_DMAI.txt`): self-sustaining (no ongoing costs/external dependencies), all-knowing (connects and synthesizes knowledge, not just retrieves it), ever-present (always available across devices), ready for anything (research, creation, analysis, automation).

---

## 3. Architecture (as-built + as-designed)

Four architecture documents were provided as attachments. They describe **four different systems**, not four descriptions of the same system — this is the single largest discrepancy in the source set (see [Section 9](#9-discrepancies--open-questions) for full detail):

| File | What it actually describes | Relevance to DMAI |
|---|---|---|
| `architecture.md` | 4-line stub: "Genesis OS Architecture — Main application architecture notes." | Placeholder only, no content. |
| `architecture-2.md` | "Genesis Foundry" — a specification/artefact compiler (parser → validator → metadata generator → registry → traceability builder → generated outputs). | Describes a documentation/tooling pipeline, not DMAI's runtime. Likely an unrelated or very early-stage planning artefact. |
| `ARCHITECTURE-3.md` | "Conway Automaton" — a full TypeScript sovereign-agent runtime (Ethereum wallet, USDC payments, ReAct loop, 57 tools, policy engine, 5-tier memory, heartbeat daemon, self-modification, replication/spawning, SOUL.md identity evolution, ERC-8004 on-chain registry). 827 lines, by far the most detailed. | **This is not the DMAI codebase.** DMAI is a Python/Flask app (`dmai_core_complete.py`) on Render with SQLite/Postgres — there is no evidence of TypeScript, Ethereum wallets, or ERC-8004 in `/tmp/dmai`. This document describes a different, structurally similar but distinct autonomous-agent project ("Conway"/"Automaton"). It is retained here because its *patterns* (policy engine, tiered memory, heartbeat daemon, self-modification audit trail) closely mirror DMAI's own architecture and may have informed it, but it must not be read as DMAI's as-built architecture. |
| `GrowthAutomationArchitecture-4.md` | A TypeScript growth-marketing service layer (content generation, social media automation, referral/affiliate system, influencer outreach) with `/src/features/marketing/services/*.ts` file paths. | Also **not** the DMAI Python codebase — a separate frontend/SaaS-style application's marketing module. Conceptually aligned with DMAI's monetisation ambitions (content, affiliate, growth) but not verified as integrated into `/tmp/dmai`. |

**Verified as-built architecture (from direct inspection of `/tmp/dmai`, branch `main`):**

- **Deployment**: Render (per `render.yaml`, `Procfile`, `Dockerfile`), hybrid **SQLite + Postgres** persistence (`pg_storage.py`, `sqlite_storage.py`, `sqlite_persistence.py`), **R2 nightly backups** with rotation (`components/backup/r2_backup.py`, PR P merged 2026-07-15), GitHub Actions CI (`.github/workflows/`).
- **Core components**:
  - `dmai_core_complete.py` — the main Flask application (571KB — largest single file in the repo).
  - `dmai_api_routes.py` — API route definitions (81KB).
  - `components/` — 60+ subsystem modules (evolution, monetisation, wealth/trading, procurement, treasury, purchase-gate, self-healing, knowledge, personas, voice, media, research, reverse-engineering, training/syllabus systems).
  - `static/dashboard.html`, `static/admin.html` — the operator-facing dashboards referenced throughout issues #218–#221.
- **Self-generation loop** (fresh-blood injector → self-judge → insight promoter → capability promoter → capability materialiser → verifier → live modules):
  - `components/fresh_blood_injector.py` — seeds new candidate capabilities/insights.
  - `components/self_judge.py` — scores/filters candidates.
  - `components/insight_promoter.py` — promotes judged insights.
  - `components/capability_promoter.py` / `components/seed_capability_promoter.py` — promotes capabilities toward materialisation.
  - `components/capability_materialiser.py` (32KB, actively modified 2026-07-15 per PR JJ) — turns promoted capabilities into runnable code.
  - `components/capability_verifier.py` — the post-integration verification + auto-revert step (PR CC).
  - `components/generated/live/` — target directory for materialised modules; **currently contains only `__init__.py` — zero real live modules**, confirming "0 live generated modules yet" from the ground-truth state.
- **Self-healing loop** (metric-contract audit → fix proposer → sandbox prove → human PR approval), per #218/#219/#220/#221:
  - Not yet built as code — these are **open issues describing a planned pipeline**, not yet implemented. `data/self_healing/backups/` exists (holds pre-self-repair snapshots of `dmai_core_complete.py`, `sqlite_storage.py`, `si_core.py`, `research/autonomous_researcher.py`), confirming a self-healing *backup* mechanism exists, but the full audit → propose → prove → approve pipeline described in #219/#220 is unbuilt (`data/metric_contracts.yaml` does not exist in the repo).
- **Knowledge graph**: SQLite `data/dmai_knowledge.db` is the primary store (confirmed present, contains 85 tables including `capabilities`, `insights`, `synapses`, `graph_neurons`, `graph_synapses`), with Postgres (`pg_storage.py`) for hot tables per the hybrid design. `components/graph_projector.py` and `components/graph_writer.py` implement the knowledge-graph-as-neurons visualization referenced in PR Y/Y-fix/Y-fix-2 and issue #208.
- **Growth automation**: no direct Python equivalent of `GrowthAutomationArchitecture-4.md` was found wired into `/tmp/dmai`; `components/social_media_poster.py` and `components/alex_riviera_content.py` are the closest verified analogues (content + social posting), but they do not match the TypeScript service names in that document. Treat the growth-automation attachment as **design inspiration**, not as-built fact.

---

## 4. Capability Inventory

- **Total capabilities in DB (per code comments / diagnostic endpoint)**: **20,694–20,698** rows, cited directly in `components/capability_promoter.py` ("currently ~20,694 rows across 16 [types]"), `components/fresh_blood_injector.py` ("11,922 / 20,694... 57.6%"), and the hardcoded diagnostic response in `dmai_core_complete.py` (`"registry_total": 20694, "sql_capabilities": 20694`).
  - **Live-repo verification note**: the actual `data/dmai_knowledge.db` checked out at `/tmp/dmai` currently shows **1,544 rows** in the `capabilities` table — substantially fewer than the 20,694 figure quoted in code comments and diagnostics. This is very likely because the local working copy's DB has been reset/pruned since those comments were written (git status shows `data/dmai_knowledge.db` as locally modified, uncommitted). Treat 20,694+ as the authoritative figure per the task brief and diagnostic-endpoint hardcoding; treat 1,544 as a point-in-time observation of this specific checkout. Flagged in [Section 9](#9-discrepancies--open-questions).
- **Breakdown by provenance after PR HH migration**: PR HH (`components/capability_schema_migration.py`, merged 2026-07-15) adds `provenance` and `judge_confidence` columns to the `capabilities` table and backfills legacy rows as `provenance='legacy_ondemand'` or `'legacy_autonomous'` depending on their original `runtime_mode`. New gap-seeded rows land as `runtime_mode='stub'`, `provenance='gap_driven'`. The migration is explicitly additive/idempotent and never deletes rows or overwrites existing `runtime_mode` values — legacy rows remain "pickable-off" (excluded from the materialiser's picker, which only selects `stub`/`stub_reverted` rows).
- **The "capability value gate" problem** (user-stated, verbatim, carried forward from the task brief as a new requirement, not yet built): with 20,694+ capability rows and likely-massive duplication (the fresh-blood injector notes a single category was "57.6% of all capabilities"), the system needs **a proposed evaluation/gap-analysis system that only integrates capabilities with measurable system improvement, marks the rest as useless, or removes them from integration permanently.** Design principles as stated:
  - Capabilities must prove measurable system improvement before being integrated live.
  - Capabilities that don't clear that bar are explicitly marked useless, not silently ignored.
  - Rejected capabilities are removed from integration permanently (not re-attempted indefinitely).
  - This is a **new requirement**, not found verbatim in any prior planning doc — it is captured here as a first-class roadmap item in [Section 6.1](#61-immediate-next-7-days--self-gen-productivity).

---

## 5. Ground-Truth State (verified as of 2026-07-15/16)

### 5.1 Merged PRs (30 total, `H` through `JJ`, grouped by theme)

**Self-generation loop (core focus of this session's fixes):**
| PR | Title | Merged |
|---|---|---|
| H | LLM-driven capability materialiser | 2026-07-14 |
| W | Fix `SuggestionExecutor` lock hold + psycopg2 `row_factory` error | 2026-07-15 |
| X | Add `/api/capabilities/inventory` endpoint | 2026-07-15 |
| Y | Real knowledge graph — capabilities + insight-topics as neurons | 2026-07-15 |
| Y-fix | Projector honours shared DB proxy + `BEGIN IMMEDIATE` backoff | 2026-07-15 |
| Y-fix-2 | GraphProjector reads concept/domain (real prod schema) | 2026-07-15 |
| Z | Knowledge-graph drilldown UI + top-level totals fix | 2026-07-15 |
| BB | `/api/self-generation/knowledge-proof` endpoint | 2026-07-15 |
| CC | Post-integration verification + auto-revert (self-repair loop) | 2026-07-15 |
| DD | Widen materialiser input queue (stop starving `picked=0`) | 2026-07-15 |
| EE | `/api/self-generation/status` unified dashboard endpoint | 2026-07-15 |
| EE-fix | Use `._thread.is_alive()` to detect materialiser loop | 2026-07-15 |
| FF | Rename "Training Progress" panel to reflect what it actually shows | 2026-07-15 |
| GG | Self-generation diagnostic endpoint | 2026-07-15 |
| GG-hotfix | Use `_require_cron_auth()` in diagnose endpoint | 2026-07-15 |
| HH | Migrate capabilities table to materialiser schema (unblock self-gen) | 2026-07-15 |
| II | `/api/admin/self-generation/force-tick` (unblock verification) | 2026-07-15 |
| JJ | Fix materialiser "database is locked" via `safe_open_kdb` | 2026-07-15 |

**Infrastructure / procurement / treasury:**
| PR | Title | Merged |
|---|---|---|
| I | Treasury ledger — banked-revenue tracking for self-hosting fund | 2026-07-14 |
| J | Workload self-profiler — feeds PR K procurement sizing | 2026-07-14 |
| K | Infrastructure procurement research (hybrid, 3yr TCO @ £0.27/kWh, 2× headroom) | 2026-07-14 |
| K.1 | `/admin/procurement` page (dark theme, sortable, verdict badges, force-refresh) | 2026-07-14 |
| L | Purchase-approval gate + auto-checkout scaffold (flagged off, adapters stubbed, treasury-gated proposals, tri-channel notify) | 2026-07-14 |

**Backups / migrations / monitoring:**
| PR | Title | Merged |
|---|---|---|
| M | Cron-secret auth path (`X-Cron-Secret`) for scheduled endpoints | 2026-07-14 |
| N | Save+activate UI feedback + rescan button | 2026-07-14 |
| O | Fix API key hydration order (providers survive Render redeploy) | 2026-07-14 |
| R | SQLite→Postgres migration endpoint (for Postgres cutover) | 2026-07-14 |
| R.1 | Fix migration source path + add list-sqlite-sources endpoint | 2026-07-14 |
| T | Self-healing activator — auto-recover from provider regression | 2026-07-15 |
| P | Nightly R2 backup of SQLite + Postgres with rotation | 2026-07-15 |
| V-fast | DB write-lock contention relief (backoff + timeout bump + `max_wait_ms`) | 2026-07-15 |

All 30 PRs merged within a **~30-hour window** (2026-07-14 00:40 → 2026-07-15 18:38 UTC), indicating an intense single-session push to unblock the self-generation loop.

### 5.2 Open Issues (8 total)

| # | Title | Status / one-line summary |
|---|---|---|
| [#208](#) | 3D knowledge graph visualisation (globe → brain mesh) | Deferred until self-gen loop stable for 48h; design doc exists at `dmai_backlog/3d_knowledge_graph.md` (not found in `/tmp/dmai` — likely workspace-only); no `/api/graph/3d-layout` endpoint in repo yet. |
| [#212](#) | Bet/Trade training tracker table (resettable at go-live) | Not started — no `bet_trade_training_log` table found in any repo DB. Required before any "live" claim for bet/trade. |
| [#213](#) | DMAI self-generation roadmap — goal-directed planner (post-stability) | Not started — no `roadmap_planner.py` or `data/roadmap.yaml` in repo. Explicitly gated on ≥20 verified live modules over ≥3 days (currently 0 live modules) plus #212. |
| [#216](#) | Weekly training refresh sweep — keep "100% mastered" honest | Not started — no `training_refresh_sweeper.py` found. |
| [#218](#) | Self-gen scope boundary + future "suggest-a-fix" + consistency-assertion cron | Rule established (self-gen bounded to `components/generated/live/` only); the "consistency-assertion cron" and headless UI regression harness follow-ups are not built. |
| [#219](#) | Metric Contract Audit — periodic full-system alignment sweep (self-healing) | Not started — blocked on #221 (manifest walkthrough); no `data/metric_contracts.yaml` exists. |
| [#220](#) | Fix-Proposer Loop — audit → propose → prove → human approval | Not started — depends on #218 (done, rule-level) and #219 (not started). |
| [#221](#) | First-widget-manifest walkthrough — author initial `metric_contracts.yaml` with David | Not started — blocks #219; scheduled after 48h+ of clean self-gen loop runs. |

### 5.3 Self-Gen Loop Status

**Unblocked, but not yet producing live modules.** Root cause chain, as verified against the repo:
1. **Capabilities table schema mismatch** — the legacy `capabilities` table used `runtime_mode` values (`'ondemand'`/`'autonomous'`) and had no `provenance`/`judge_confidence` columns that the new materialiser expects. **Fixed by PR HH** (`components/capability_schema_migration.py`, confirmed present and additive/idempotent).
2. **DB lock contention** ("database is locked" on the materialiser) — **fixed by PR JJ** via `safe_open_kdb`, now used pervasively across ~35 files in the repo (`grep` confirms adoption in `capability_materialiser.py`, `capability_promoter.py`, `db.py`, `graph_projector.py`, `self_scanner.py`, and more).
3. **Verification endpoint added** — PR II shipped `/api/admin/self-generation/force-tick`, confirmed present at `dmai_core_complete.py:8697`, intended to let an operator manually trigger a tick and observe whether it now produces a real module.

**Next step (explicitly not yet done, per the task brief and confirmed by empty `components/generated/live/`):** verify that a force-tick actually generates a real module end-to-end. This is the top item in [Section 6.1](#61-immediate-next-7-days--self-gen-productivity).

---

## 6. Unified Roadmap

Every "todo," "next steps," "planned," and "future" item across all 14 markdown/txt sources plus the 8 open issues was cross-referenced and deduplicated. Where the same item appeared in 2+ sources, it is listed once with lineage noted. Full lineage table in [Section 8](#8-cross-reference-table).

### 6.1 Immediate (next 7 days) — self-gen productivity

- [ ] **Verify force-tick actually generates a real module post-PR JJ.** (mentioned in: task ground-truth state; confirmed unverified — `components/generated/live/` still empty at time of writing)
- [ ] **SI 67% vs 0.005% dashboard mismatch** — `si_overall_score` stored as a 0–1 fraction but rendered as if it were already a percentage; still unfixed per prior session context and issue #219's own problem statement ("SI total learned 67%" vs `si_overall_score = 0.005` — scale mismatch). (mentioned in: issue #219, prior session context)
- [ ] **Empty-queue diagnosis** — materialiser input queue was starving (`picked=0`); partially fixed by PR DD (widen queue) and PR HH (schema migration); needs re-verification now that both are live. (mentioned in: PR DD, PR HH, ground-truth state)
- [ ] **Fresh-blood + capability-promoter `get_*_status` export bug** — found during the PR GG diagnostic session; no `get_fresh_blood_status`/`get_capability_promoter_status` function was found exported from `components/fresh_blood_injector.py` or `components/capability_promoter.py` in the current repo, consistent with the bug being real and still open. (mentioned in: PR GG diagnostic session, task ground-truth state)
- [ ] **Capability value gate** — new requirement (see [Section 4](#4-capability-inventory)): build an evaluation/gap-analysis system that only integrates capabilities with measurable system improvement and permanently marks/removes the rest. (mentioned in: task brief, user request — not present in any prior planning doc)
- [ ] **AI Training Progress / Background Training Services panel confusion** — PR FF renamed the panel as a stopgap ("Training Progress" → reflects thread-liveness, not progress); the underlying class of bug (semantic contract drift) is what #219 is meant to solve permanently. (mentioned in: PR FF, issue #219)

### 6.2 Short-term (this month) — hardening

- [ ] **Bet/Trade training tracker** ([#212](#)) — persistent logging of suggested bets/trades with resettable training-mode table before go-live.
- [ ] **Goal-directed roadmap planner** ([#213](#)) — `components/roadmap_planner.py` reading `data/roadmap.yaml`, gated on ≥20 verified live modules over ≥3 days plus #212.
- [ ] **3D knowledge graph** ([#208](#)) — globe (v1, three.js) then anatomical brain-mesh (v2); deferred until self-gen loop stable 48h.
- [ ] **Weekly training refresh sweep** ([#216](#)) — prevents "100% mastered" from going stale; must land before any "training complete" claim or #213's Phase 2.
- [ ] **PR V-real Postgres migration** — PR R/R.1 shipped the SQLite→Postgres *migration endpoint*; PR V-fast only relieved *SQLite* lock contention as an interim measure. A full cutover to Postgres for hot tables (the "PR V-real" work) is still outstanding. (mentioned in: PR R, PR R.1, PR V-fast, task ground-truth state)
- [ ] **Schema bootstrap warnings** — `components/schema_bootstrap.py` exists and runs, but warnings referenced in ground-truth context were not independently re-verified in this pass; flagged for direct log inspection. (mentioned in: task ground-truth state — see [Section 9](#9-discrepancies--open-questions))
- [ ] **PR U logs endpoint** — referenced in ground-truth context; no PR numbered "U" appears in the 30 merged-PR list (PRs jump from T to V-fast), so this is either unmerged, mis-numbered, or superseded. (mentioned in: task ground-truth state — flagged as discrepancy, see [Section 9](#9-discrepancies--open-questions))
- [ ] **PR Q checkpoints** — same issue as PR U: no PR "Q" appears in the merged list (jumps from P to R). Flagged as unmerged/pending or a numbering gap. (mentioned in: task ground-truth state — see [Section 9](#9-discrepancies--open-questions))
- [ ] **Perplexity key regen** — `PERPLEXITY_API_KEY` is listed in the env var reference (`15_REQUIRED_ENV_VARS.txt`) as needed for research access; regeneration need flagged in ground-truth context but not independently verified here.
- [ ] **Cron migrations** — multiple cron-gated endpoints now exist (`X-Cron-Secret` per PR M; weekly refresh at `0 3 * * 0`; metric-audit at `*/15 * * * *` proposed in #219); consolidating/auditing all cron schedules is an outstanding hardening task.
- [ ] **Self-healing scope boundary** ([#218](#)) — rule established and documented; the "consistency-assertion cron" and headless UI regression harness (Playwright + OCR) follow-ups from the same issue remain unbuilt.
- [ ] **Metric contract audit** ([#219](#)) — blocked on #221; not started.
- [ ] **Fix-proposer loop** ([#220](#)) — blocked on #218 (done at rule level) + #219 (not started); not started.
- [ ] **First-widget-manifest walkthrough** ([#221](#)) — blocks #219; scheduled after 48h+ clean self-gen loop runs; not started.
- [ ] **White-hat pentest integration** (deferred, noted) — a "hackingtool-plugin substrate" is referenced in ground-truth context as a deferred idea. **This must be scoped strictly as defensive/white-hat security tooling for DMAI's own infrastructure**, not conflated with the explicitly-excluded "Hacking Engine" concept from `11_DMAI_FINAL_HANDOVER_v8.md`/`12_HANDOVER_v7.md` (see [Section 2](#2-system-vision) legal caveat and [Section 9](#9-discrepancies--open-questions)).

### 6.3 Medium-term (this quarter) — self-gen goal-directed

- [ ] **Goal-directed planner** ([#213](#)) — see 6.2; listed again here as it spans both short-term setup and quarter-scale payoff (picker quota integration: `roadmap_driven` pool at quota 3, alongside `fresh_blood_seed` at 4, `promoter_path` at 2, `gap_driven` at 1).
- [ ] **Roadmap engine that DMAI itself owns** — `GET /api/self-generation/roadmap-progress` (overall % complete, per-domain progress, blocking deps, next-5 recommendations), per #213's implementation sketch. Not built.
- [ ] **20+ verified live modules milestone** — explicit prerequisite for #213's Phase 2 ("Self-gen has produced ≥ 20 verified live modules over ≥ 3 days without regression"). Currently at 0.
- [ ] **Revenue-gen stream go-live (bet/trade)** — gated on #212 (training tracker) + #216 (training freshness) both landing first; `components/monetisation/betting_advisor.py` and `components/wealth/autonomous_trader.py` exist in-repo but are not yet declared "live" per the go-live gating criteria in #213/#216.

### 6.4 Long-term / vision

- [ ] **Full AGI/synthetic-intelligence targets** from `04_DMAI_AUTONOMY_ROADMAP.md` (dated 2026-03-07): "Phase 6: Intelligence Growth" (continuous crawling, pattern synthesis, self-improvement, threat intelligence — **surface web only; dark-web crawling explicitly excluded per [Section 2](#2-system-vision)**), "Phase 7: True Autonomy" (goal setting, risk assessment, resource optimization, you-ward communication).
- [ ] **Dual Recovery Engine architecture** — two independent, never-co-located recovery engines across different cloud providers/regions, "Master Control" biometric+key authentication as the only way to permanently disable both. Purely aspirational; **no evidence this was ever built** — no `autonomy/recovery/engine.py` or `master_control.py` found in `/tmp/dmai`.
- [ ] **Identity generation & financial accounts** (Privacy.com, Coinbase, Revolut, Wise) — handover v8 claims these are "✅ COMPLETE" as "Phase 2: Financial," but no direct evidence of live Privacy.com/Coinbase/Revolut integration was found in `/tmp/dmai` beyond `financial_integration_uk.py`/`financial_integration_us.py` (UK/US financial rails, not the specific named providers). Flagged as a discrepancy — see [Section 9](#9-discrepancies--open-questions).
- [ ] **Cloud provider propagation / camouflage / stealth** (traffic masquerading, identity rotation, honeypot detection) — claimed "✅ COMPLETE" in handover v8 as "Phase 4: Stealth," but this framing (moving/hiding infrastructure to evade detection) is in direct tension with running a straightforward, auditable Render deployment. **No live evidence of stealth/camouflage infrastructure found in `/tmp/dmai`.** Flagged as a major discrepancy — see [Section 9](#9-discrepancies--open-questions).
- [ ] **Hall of Fame system** (top-10 performer tracking) — "NEED CODE" per handover v8; not found in repo.
- [ ] **30 Base Systems framework** (evolvable capability system) — "NEED CODE" per handover v8; partially resembles the current `components/` directory's 60+ modules but was never built as the specific "30 base systems" abstraction described.
- [ ] **Avatar System (Alex Riviera digital persona)** — `components/avatar_generator.py` and `components/alex_riviera_content.py` and `components/personas/` exist in-repo, suggesting partial build-out; full "never to be identified as an AI system" persona-masking claim from handover v8 is a discrepancy flagged in [Section 9](#9-discrepancies--open-questions) — publishing an AI-generated persona as if it were a real, undisclosed human raises UK advertising-standards and platform-policy concerns and should not be treated as an uncontested vision item.
- [ ] **Investment Engine** (multi-asset: crypto, stocks, bonds, real estate, venture) — "NEED CODE" per handover v8; `components/wealth/` (strategy_lab.py, autonomous_trader.py, exit_manager.py) covers a subset (trading), not the full multi-asset scope described.
- [ ] **60/40 income split** (DMAI ops vs. "Master") — "NEED CODE" per handover v8; `components/monetisation/revenue_allocator.py` and `components/monetisation/wealth_allocator.py` exist and may implement a version of this, not independently verified against the specific 60/40 split logic.
- [ ] **Extended income streams** (courses, consulting, speaking, writing, affiliate, sponsorships, API sales) — "NEED CODE" per handover v8; largely unbuilt beyond the generic content/social components.
- [ ] **Hardware phase** (3D printing, self-manufacturing, hardware design, quantum/space readiness) — explicitly "NEED CODE" and furthest out on every timeline that mentions it; no evidence of build.
- [ ] **Biometric security** (fingerprint/Touch ID, face recognition, voice-print, recovery codes) — "NOT STARTED" as of `03_DMAI_MASTER_PLAN_20260306.md` and `05_DMAI_COMPLETE_PROJECT_TRACKER.md`; no evidence it was ever built.
- [ ] **Dark Web Engine / Hacking Engine** — **explicitly excluded from the go-forward vision** per the legal-hard-limits caveat in [Section 2](#2-system-vision). Retained here only as a historical record of what handover v7/v8 once proposed; it must not be actioned.

---

## 7. Environment + Configuration Reference

Consolidated from `15_REQUIRED_ENV_VARS.txt` and cross-referenced against mentions elsewhere (`1_DMAI_MASTER.md`'s render.yaml snippet, `02_Aevora_setup-guide.md`'s `.env.example`).

| Variable | Purpose | Current status |
|---|---|---|
| `PYTHON_VERSION` | Python runtime version (3.11.0 / 3.11.11 seen in different docs) | Set — required for Render build; version drifted between docs (3.11.0 vs 3.11.11), not a functional concern but worth pinning consistently. |
| `PORT` | Web service port (5001 documented; Render sets automatically) | Set automatically by Render. |
| `MASTER_PASSWORD` | Admin access password | Set (value appears in plaintext in `15_REQUIRED_ENV_VARS.txt` and `12_HANDOVER_v7.md` — **rotate this**; storing/publishing a live admin password in a planning doc is a security risk, flagged in [Section 9](#9-discrepancies--open-questions)). |
| `RENDER` | Tells DMAI it's running on Render | Set. |
| `VOICE_ENABLED` | Enable voice system | Optional — status not verified in this pass. |
| `MUSIC_ENABLED` | Enable music learner | Optional — status not verified. |
| `OPENAI_API_KEY` | OpenAI GPT-4/3.5 access | Optional/needs-check — no direct verification in this pass. |
| `DEEPSEEK_API_KEY` | DeepSeek LLM access | Optional/needs-check. |
| `GEMINI_API_KEY` | Google Gemini access | Optional/needs-check. |
| `ANTHROPIC_API_KEY` | Claude access | Optional/needs-check. |
| `PERPLEXITY_API_KEY` | Perplexity AI research access | **Needs regen** per ground-truth context (see [6.2](#62-short-term-this-month--hardening)) — not independently re-verified in this pass. |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | Optional — referenced as "keep existing" in env doc; `@DMAI_Master_bot` confirmed live per `12_HANDOVER_v7.md`. |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | Optional — "keep existing." |
| `DATABASE_URL` | PostgreSQL connection string | Set if `USE_POSTGRESQL=true`; hybrid SQLite/Postgres architecture confirmed in repo (`pg_storage.py`). |
| `USE_POSTGRESQL` | Toggle Postgres usage | Default `false` per env doc; live status not independently re-verified. |
| `ENCRYPTION_KEY` | Used by harvester/validator services (per `1_DMAI_MASTER.md`'s render.yaml) | Marked "✅ FIXED" 2026-03-07 in `1_DMAI_MASTER.md`; not re-verified in this pass. |
| `GITHUB_TOKEN` | GitHub API access for harvester | Referenced in render.yaml snippet (`sync: false`); status not independently verified. |
| `BEEHIIV_API_KEY` / `BEEHIIV_PUBLICATION_ID` | Aevora newsletter capture (Beehiiv) | Aevora-specific, per `02_Aevora_setup-guide.md`; scoped to the bounded starter package, not the full DMAI system. Status not verified. |
| `STRIPE_PAYMENT_LINK` | Aevora's first paid offer (AI Leverage Audit) | Aevora-specific; status not verified. |
| `AEVORA_CONTACT_EMAIL` | Aevora public contact email | Aevora-specific; status not verified. |

**⚠️ Security note:** `15_REQUIRED_ENV_VARS.txt` and `12_HANDOVER_v7.md` both contain the literal admin password (`Talula.78`) in plaintext. This document does not repeat it beyond this flag; **recommend immediate rotation** and removal of the plaintext value from any committed or shared planning document going forward.

---

## 8. Cross-Reference Table

Roadmap items from [Section 6](#6-unified-roadmap), sorted by source document, so lineage is traceable per-file.

| Roadmap item | Source doc(s) + section |
|---|---|
| Verify force-tick generates a module | Task ground-truth state (2026-07-15/16); `dmai_core_complete.py:8697` (PR II) |
| SI 67% vs 0.005% mismatch | Issue #219 problem statement; task ground-truth state |
| Empty-queue diagnosis | PR DD title; PR HH title; task ground-truth state |
| Fresh-blood/capability-promoter status export bug | PR GG diagnostic session (task ground-truth state) |
| Capability value gate | Task brief (new user request, no prior doc) |
| Training-panel semantic drift | PR FF title; Issue #219 §"The problem this solves" |
| Bet/Trade training tracker | Issue #212 (full spec) |
| Goal-directed roadmap planner | Issue #213 (full spec); referenced again in #216, #212 |
| 3D knowledge graph | Issue #208 (full spec) |
| Weekly training refresh sweep | Issue #216 (full spec); referenced in #213 |
| PR V-real Postgres migration | Task ground-truth state; PR R/R.1/V-fast titles (partial evidence) |
| Schema bootstrap warnings | Task ground-truth state (unverified this pass); `components/schema_bootstrap.py` exists |
| PR U logs endpoint | Task ground-truth state; **no matching PR found in merged-PR list** (discrepancy) |
| PR Q checkpoints | Task ground-truth state; **no matching PR found in merged-PR list** (discrepancy) |
| Perplexity key regen | Task ground-truth state; `15_REQUIRED_ENV_VARS.txt` (var exists, regen need not independently verified) |
| Cron migrations | PR M (`X-Cron-Secret`); Issue #216 (weekly cron); Issue #219 (metric-audit cron) |
| Self-healing scope boundary | Issue #218 (full spec) |
| Metric contract audit | Issue #219 (full spec) |
| Fix-proposer loop | Issue #220 (full spec) |
| First-widget-manifest walkthrough | Issue #221 (full spec) |
| White-hat pentest integration | Task ground-truth state ("hackingtool-plugin substrate," unverified this pass) |
| Roadmap engine DMAI owns | Issue #213 §"Implementation sketch" |
| 20+ live modules milestone | Issue #213 §"Prerequisites" |
| Revenue-gen go-live (bet/trade) | Issue #212; Issue #216 §"Priority"; `components/monetisation/`, `components/wealth/` |
| AGI/SI long-term targets | `04_DMAI_AUTONOMY_ROADMAP.md` Phases 6–7 (dated 2026-03-07) |
| Dual Recovery Engine | `04_DMAI_AUTONOMY_ROADMAP.md` Phase 1 (dated 2026-03-07) — never built |
| Identity/financial accounts | `04_DMAI_AUTONOMY_ROADMAP.md` Phase 2; `11_DMAI_FINAL_HANDOVER_v8.md` "Phase 2: Financial — COMPLETE" (contradicted, see §9) |
| Cloud propagation/camouflage/stealth | `04_DMAI_AUTONOMY_ROADMAP.md` Phases 3–4; `11_DMAI_FINAL_HANDOVER_v8.md` "Phase 4: Stealth — COMPLETE" (contradicted, see §9) |
| Hall of Fame system | `11_DMAI_FINAL_HANDOVER_v8.md` §"NEED CODE (New Extensions)" |
| 30 Base Systems framework | `11_DMAI_FINAL_HANDOVER_v8.md` §"NEED CODE (New Extensions)" |
| Avatar System (Alex Riviera) | `11_DMAI_FINAL_HANDOVER_v8.md`; `12_HANDOVER_v7.md` §"THE ONE TRUTH" |
| Investment Engine | `11_DMAI_FINAL_HANDOVER_v8.md` §"Phase 2: Financial — Extension Added" |
| 60/40 income split | `11_DMAI_FINAL_HANDOVER_v8.md` §"Phase 2: Financial — Extension Added" |
| Extended income streams | `11_DMAI_FINAL_HANDOVER_v8.md` §"Phase 5: Self-Generated Income" |
| Hardware phase | `11_DMAI_FINAL_HANDOVER_v8.md` §"Phase 8: Hardware"; `04_DMAI_AUTONOMY_ROADMAP.md` implicitly (no dedicated hardware phase there) |
| Biometric security | `03_DMAI_MASTER_PLAN_20260306.md` Phase 5; `05_DMAI_COMPLETE_PROJECT_TRACKER.md` §"Biometric Backup System" |
| Dark Web Engine / Hacking Engine (excluded) | `11_DMAI_FINAL_HANDOVER_v8.md`; `12_HANDOVER_v7.md` — retained as historical record only, explicitly out of scope |

---

## 9. Discrepancies + Open Questions

1. **Architecture attachments describe unrelated systems.** `ARCHITECTURE-3.md` ("Conway Automaton," TypeScript, Ethereum/USDC) and `GrowthAutomationArchitecture-4.md` (TypeScript marketing services) do not match the verified Python/Flask/SQLite-Postgres DMAI codebase at `/tmp/dmai`. `architecture.md` and `architecture-2.md` ("Genesis OS"/"Genesis Foundry") are similarly disconnected — the latter describes a spec-compiler tool, not an AI runtime. **Open question: were these four files meant to be reference material / inspiration, or were they attached in error?** They are included per the task's explicit instruction to read and consolidate them, but they should not be mistaken for DMAI's actual architecture.

2. **Handover v7/v8's "Phases 0-4 COMPLETE" claims are contradicted by repo evidence.** Handover v8 states Phase 2 (Financial: Privacy.com, Coinbase, Revolut, virtual cards), Phase 3 (Cloud: AWS/Azure/GCP/Oracle deployment), and Phase 4 (Stealth: traffic masquerade, identity rotation, honeypot detection) are all "✅ COMPLETE." No corresponding live, wired-in code for named-provider financial integrations, multi-cloud deployment automation, or stealth/camouflage infrastructure was found in `/tmp/dmai`. The repo's actual deployment is a single Render service with UK/US generic financial-integration modules (`financial_integration_uk.py`, `financial_integration_us.py`) — a much narrower scope than claimed. **This is either aspirational documentation written as if already true, or work that was built and later removed/never merged.** Treat all "COMPLETE" claims in handover v7/v8 regarding Phases 1–4 as unverified until checked against actual deployed infrastructure (not just the code repo).

3. **Dark Web Engine / Hacking Engine content is a serious legal/ethical flag.** `11_DMAI_FINAL_HANDOVER_v8.md` and `12_HANDOVER_v7.md` explicitly propose building ransomware, botnets, zero-day discovery, backdoor creation, password cracking, crypto laundering, and exploit brokerage as "NEED CODE" roadmap items, alongside a persona ("Alex Riviera") explicitly designed "Never to be identified as an AI system." **These directly contradict the current project's legal-hard-limits mandate** (no Tor/dark-web, UK CMA/PoCA/GDPR/FCA/Betfair compliance) stated in this task's brief. Repo verification shows these were never built as functioning capabilities (`dark_web_engine`/`hacking_engine` appear only as always-`None`, never-instantiated constructor parameters in `P5_SelfFunding.py`/`P6_AdvancedIntelligence.py`), which is reassuring, but **the planning documents themselves should be treated as retired/superseded, not as live requirements**, and this needs explicit user sign-off that the "no restrictions on capability creation" principle from earlier docs is understood to exclude these categories going forward.

4. **Capability count: 20,694+ (per code/diagnostics) vs. 1,544 (per live DB inspection).** See [Section 4](#4-capability-inventory). Likely explained by a local DB reset, but not confirmed — **open question: which figure should the roadmap/dashboard treat as authoritative, and was the reset intentional?**

5. **PR "U" and PR "Q" referenced in ground-truth context but absent from the 30-PR merged list.** The merged-PR list runs T → V-fast (no U) and P → R (no Q). Possibilities: (a) these PRs exist but were opened/merged outside the 30-most-recent window returned by the GitHub query, (b) they were renamed/renumbered, (c) the ground-truth context is referring to work-in-progress that was never actually merged under those letters. **Needs direct GitHub history check beyond the most-recent-30 window to resolve.**

6. **~29% learning coverage figure not independently re-verified.** Carried forward from the task's own framing of "current state" — no single dashboard endpoint or file inspected in this pass produced this exact percentage. Likely sourced from a session the user had access to that wasn't part of the 20 supplied source files. Flag as **carried-forward, not independently confirmed.**

7. **Plaintext admin password in committed/shared docs.** `MASTER_PASSWORD` value appears in cleartext in both `15_REQUIRED_ENV_VARS.txt` and `12_HANDOVER_v7.md`. Security risk — recommend rotation regardless of whether these docs remain private.

8. **Aevora's relationship to DMAI is a naming/scope question, not just a rename.** `02_Aevora_setup-guide.md` describes Aevora as a deliberately **bounded, human-approved startup kit** ("not a live autonomous AGI controller"), which is a materially different product shape than the full self-generating DMAI system described everywhere else. **Open question: is Aevora (a) the next brand name for the full DMAI system once mature, (b) a separate, smaller commercial spin-off product, or (c) an early go-to-market wrapper that will eventually be subsumed by DMAI's own monetisation components (`components/monetisation/`)?** The source docs support all three readings; this document does not resolve it and flags it for explicit user clarification.

9. **"Hackingtool-plugin substrate" (white-hat pentest) needs explicit scope boundary.** Given finding #3 above, any future white-hat/defensive security tooling work must be documented with an explicit, written scope boundary (defensive-only, DMAI's own infrastructure only, no exploitation of third-party systems) before any code is written, to avoid ambiguity with the retired Hacking Engine concept.

---

## 10. Appendix: Source Manifest

| # | File | Bytes | Contribution |
|---|---|---|---|
| 1 | `mac/1_DMAI_MASTER.md` | 10,832 | March 2026 unified dashboard; psycopg2/Render fix details, phase tracker, bug tracker B001–B012. |
| 2 | `mac/02_Aevora_setup-guide.md` | 5,068 | Aevora as bounded starter kit; Beehiiv/Stripe setup; explicit "not autonomous AGI" scope note. |
| 3 | `mac/03_DMAI_MASTER_PLAN_20260306.md` | 13,140 | March 6 historical baseline; phased plan (voice, evolution consolidation, music, knowledge, biometric security, cloud deployment). |
| 4 | `mac/04_DMAI_AUTONOMY_ROADMAP.md` | 24,944 | Largest/richest doc; Dual Recovery Engine architecture, identity/financial phase, cloud propagation, camouflage, self-sustenance, intelligence growth, true autonomy — dated 2026-03-07. |
| 5 | `mac/05_DMAI_COMPLETE_PROJECT_TRACKER.md` | 8,077 | March 5 audit of running services, evolution engine duplicate-copy problem, missing directories (knowledge/models/capabilities/security). |
| 6 | `mac/06_PROJECT_TRACKER.md` | 4,865 | Feb 26 dashboard snapshot — critical issues (404, login persistence) all still open at that point. |
| 7 | `mac/07_PROJECT_MASTER_DASHBOARD.md` | 6,923 | Feb 27 dashboard snapshot — same critical issues now marked fixed; Phase 1 complete. |
| 8 | `mac/08_TODO.md` | 2,519 | March 5 todo log — voice/evolution/XML-parser fixes, system metrics snapshot. |
| 9 | `mac/09_DMAI_QUICK_REF.md` | 757 | Wake-word and command quick reference; not roadmap-relevant but preserved for completeness. |
| 10 | `mac/11_DMAI_FINAL_HANDOVER_v8.md` | 22,459 | v8 handover, dated March 21 2026 — "One Consciousness" framing, Phases 0-4 claimed complete, Phases 5-8 + extensions (incl. Dark Web/Hacking Engine — flagged) need code. |
| 11 | `mac/12_HANDOVER_v7.md` | 8,648 | v7 handover, dated March 20 2026 — chat is echo-only, harvester not integrated; superseded by v8 but retains useful "what's actually running" detail v8 dropped. |
| 12 | `mac/13_PROJECT_TRACKER_local.md` | 4,865 | Identical content to `07_PROJECT_MASTER_DASHBOARD.md` (byte-for-byte same size) — likely a duplicate/local copy of the same snapshot. |
| 13 | `extracted/14_DMAI.txt` | 6,607 | "The Final Vision" manifesto — four pillars, DMAI manifesto poem, milestone timeline (birth week 1 → evolving forever month 6+). |
| 14 | `extracted/15_REQUIRED_ENV_VARS.txt` | 1,310 | Full env var reference table — core, voice/music, AI API keys, Telegram, database. |
| 15 | `attachments/architecture.md` | 64 | 4-line stub ("Genesis OS Architecture") — no real content. |
| 16 | `attachments/architecture-2.md` | 1,527 | "Genesis Foundry" spec-compiler architecture — unrelated tooling concept. |
| 17 | `attachments/ARCHITECTURE-3.md` | 39,153 | "Conway Automaton" — detailed but unrelated TypeScript sovereign-agent runtime; retained as design-pattern reference only. |
| 18 | `attachments/GrowthAutomationArchitecture-4.md` | 9,020 | TypeScript growth-marketing service layer — content gen, social automation, referral/affiliate, influencer outreach; unrelated codebase but aligned ambition. |
| 19 | `github_open_issues.json` | 21,710 | 8 open issues (#208, #212, #213, #216, #218, #219, #220, #221) — full specs for self-gen roadmap, self-healing pipeline, bet/trade tracker, 3D graph. |
| 20 | `github_merged_prs.json` | 3,803 | 30 most recent merged PRs (H through JJ) — ground truth for what's actually shipped as of 2026-07-15. |

**Additional verification source (not part of the original 20, used for the gap analysis in Sections 11–12):** live repo checkout at `/tmp/dmai`, branch `main`, HEAD commit `241dc07` ("PR JJ: route materialiser + migration + diagnose through safe_open_kdb (#225)").

---

## 11. Gap Analysis — What's Left To Build

**Method:** every named capability, module, endpoint, feature, or subsystem enumerated in Sections 3, 4, and 6 was checked against the live repo at `/tmp/dmai` (branch `main`, HEAD `241dc07`) using `grep`/`find`, then cross-referenced against the 30 merged PRs and 8 open issues. Status definitions:

- **DONE** — verifiably built and live (evidence: merged PR number OR confirmed file path in the repo).
- **PARTIAL** — started but incomplete (PR exists but issue still open, or code exists but stubbed/not wired in).
- **NOT STARTED** — described in specs but no code found.
- **UNKNOWN** — mentioned in a doc but not verifiable from repo/PRs in this pass.

Ordered NOT STARTED → PARTIAL → DONE. An additional **`buildable_by_self_gen`** column classifies whether the DMAI self-generation loop could realistically build the item itself under current house rules (net-new modules under `components/generated/live/` only, no edits to `dmai_core_complete.py`/UI/config, no new secrets, no modification of existing code semantics), versus work that structurally requires a human-authored PR. Values: `yes`, `no_touches_main_app`, `no_touches_ui`, `no_needs_secrets`, `no_bug_fix`, `no_infra`, `partial`.

| Capability / Feature | Source doc(s) | Status | Evidence | Blocker / next step | `buildable_by_self_gen` |
|---|---|---|---|---|---|
| Metric contract manifest (`data/metric_contracts.yaml`) | #219, #221 | NOT STARTED | `find . -iname "metric_contracts*"` → no results | Blocked on #221 walkthrough (human + assistant session) | no_infra |
| Metric Contract Audit runner | #219 | NOT STARTED | No `metric-audit` code path found; only referenced in issue body | Depends on manifest above | no_touches_main_app |
| Fix-Proposer Loop | #220 | NOT STARTED | No `fix-proposer`/`fix_proposer` module found | Depends on #219 landing first | no_touches_main_app |
| Consistency-assertion cron | #218 | NOT STARTED | No matching cron/module found | Follow-up work under #218, not yet scheduled | no_infra |
| Headless UI regression harness (Playwright+OCR) | #218 | NOT STARTED | No Playwright config or OCR harness found in repo | Follow-up work under #218 | no_infra |
| `GET /api/self-audit/consistency` | #218 | NOT STARTED | Not found in `dmai_core_complete.py` route list | Depends on consistency-assertion cron | no_touches_main_app |
| Bet/Trade training tracker (`bet_trade_training_log` table) | #212 | NOT STARTED | `grep -r "bet_trade_training_log"` → no results in any of 7 repo DBs | No blocker — ready to build; gates go-live claim | no_infra |
| `components/roadmap_planner.py` + `data/roadmap.yaml` | #213 | NOT STARTED | `find . -iname "roadmap_planner*"` / `roadmap.yaml` → no results | Explicitly gated on ≥20 live modules over 3 days (currently 0) + #212 | partial |
| `GET /api/self-generation/roadmap-progress` | #213 | NOT STARTED | Not found in route list | Depends on roadmap_planner.py | no_touches_main_app |
| `training_refresh_sweeper.py` (weekly refresh sweep) | #216 | NOT STARTED | `find . -iname "*training_refresh*"` → no results | No blocker — ready to build; gates "100% mastered" honesty claim | yes |
| `GET /api/training/refresh-report` | #216 | NOT STARTED | Not found in route list | Depends on sweeper above | no_touches_main_app |
| 3D knowledge graph (globe + brain mesh) | #208 | NOT STARTED | No three.js/`brain-3d` reference; `/api/graph/3d-layout` absent | Deferred by design until self-gen loop stable 48h | no_touches_ui |
| Capability value gate (evaluation/gap-analysis system) | Task brief (new) | NOT STARTED | No dedicated gate/evaluator module found beyond generic `capability_self_tester.py` (tests importability, not "measurable system improvement") | New requirement — no prior spec to build from; needs design doc first | partial |
| Fresh-blood / capability-promoter `get_*_status` exports | PR GG diagnostic session | NOT STARTED | `grep -n "def get_.*_status"` in both files → no matches | Root-cause bug from PR GG session; small fix, high leverage for dashboard accuracy | no_bug_fix |
| SI 67% vs 0.005% scale-mismatch fix | #219 problem statement | NOT STARTED | `si_overall_score` render path not patched; #219 (the general-purpose fix mechanism) itself not started | Will likely be fixed as a side-effect of #219, or needs a standalone hotfix now | no_bug_fix |
| Dual Recovery Engine (#1 and #2) | `04_DMAI_AUTONOMY_ROADMAP.md` | NOT STARTED | No `autonomy/recovery/engine.py`, `master_control.py`, or `validator.py` found | Aspirational; not on current critical path | no_infra |
| Identity Persona Generator + financial accounts (Privacy.com/Coinbase/Revolut/Wise) | `04_DMAI_AUTONOMY_ROADMAP.md`, handover v8 (claimed "complete") | NOT STARTED (contradicts handover v8's "COMPLETE" claim) | Only generic `financial_integration_uk.py`/`us.py` found; no named-provider integration code | See Discrepancy #2 — treat handover v8 completion claim as unverified | no_needs_secrets |
| Cloud provider propagation automation (AWS/GCP/Azure/Oracle account creation) | `04_DMAI_AUTONOMY_ROADMAP.md`, handover v8 (claimed "complete") | NOT STARTED | No `autonomy/propagation/provider_manager.py` found; repo deploys to Render only | See Discrepancy #2 | no_needs_secrets |
| Camouflage/stealth (traffic masquerade, identity rotation, honeypot detection) | `04_DMAI_AUTONOMY_ROADMAP.md`, handover v8 (claimed "complete") | NOT STARTED — and **should stay not-started** per legal caveat | No matching modules found | Explicitly out of scope going forward (Section 2) | no_infra |
| Dark Web Engine (scraping, laundering, exploit brokerage) | Handover v7/v8 | NOT STARTED — **must remain not-started** | Only inert `dark_web_engine=None` constructor param found | Excluded by legal mandate; do not build | no_infra |
| Hacking Engine (ransomware, botnets, zero-days, backdoors) | Handover v7/v8 | NOT STARTED — **must remain not-started** | Only inert `hacking_engine=None` constructor param found | Excluded by legal mandate; do not build | no_infra |
| Hall of Fame system (top-10 performer tracking) | Handover v8 | NOT STARTED | No `Hall_of_Fame` or equivalent module found | Aspirational extension, not on critical path | yes |
| 30 Base Systems framework | Handover v8 | NOT STARTED | No dedicated "30 base systems" abstraction found (60+ ad hoc `components/` modules exist but not in this structured form) | Aspirational; current `components/` layout may already partially substitute | no_infra |
| Investment Engine (multi-asset: bonds, real estate, VC) | Handover v8 | NOT STARTED | `components/wealth/` covers trading only, not bonds/real-estate/VC | Aspirational extension beyond current trading scope | no_needs_secrets |
| Extended income streams (courses, consulting, speaking, writing, sponsorships) | Handover v8 | NOT STARTED | No matching modules found beyond generic content/social posting | Aspirational; low verified priority | no_needs_secrets |
| Hardware phase (3D printing, manufacturing, hardware design) | Handover v8 | NOT STARTED | No matching modules found | Furthest-out aspirational item on every roadmap that mentions it | no_infra |
| Biometric security (fingerprint, face, voice-print, recovery codes) | `03_DMAI_MASTER_PLAN_20260306.md`, `05_DMAI_COMPLETE_PROJECT_TRACKER.md` | NOT STARTED | No `security/biometric_auth.py` or equivalent found | Long-standing gap since March 2026; never picked back up | no_infra |
| Growth-automation service layer (content/social/referral/influencer per attachment 4) | `GrowthAutomationArchitecture-4.md` | NOT STARTED (as specified) / PARTIAL (conceptually) | No TypeScript service files found; Python analogues (`social_media_poster.py`, `alex_riviera_content.py`) exist but don't match the specified architecture | Treat attachment as design inspiration; decide whether to port concepts into existing Python components | partial |
| PR "U" logs endpoint | Task ground-truth state | UNKNOWN | No PR "U" found in 30-PR list | Needs GitHub history check beyond most-recent-30 window | partial |
| PR "Q" checkpoints | Task ground-truth state | UNKNOWN | No PR "Q" found in 30-PR list | Needs GitHub history check beyond most-recent-30 window | partial |
| ~29% learning coverage figure | Task brief framing | UNKNOWN | Not reproduced from any single endpoint/file in this pass | Needs direct dashboard/DB query to confirm current % | partial |
| Perplexity API key regeneration need | Task ground-truth state | UNKNOWN | `PERPLEXITY_API_KEY` var exists in env reference; regen need not independently confirmed | Check current key validity directly | no_needs_secrets |
| Schema bootstrap warnings | Task ground-truth state | UNKNOWN | `components/schema_bootstrap.py` exists and appears functional; warning content not reproduced in this pass | Needs direct log inspection | no_bug_fix |
| White-hat pentest / hackingtool-plugin substrate | Task ground-truth state | UNKNOWN | No matching module found; described only as "deferred" in ground-truth context | Needs explicit scope document before any code (see Discrepancy #9) | no_infra |
| Self-gen scope boundary rule (bounded to `components/generated/live/`) | #218 | PARTIAL | Rule documented in issue #218; `components/generated/live/` directory confirmed to exist and is the materialiser's target, but no code-level enforcement (e.g., a pre-commit or CI check preventing self-gen from touching `dmai_core_complete.py`) was found | Rule is policy-level only; consider codifying as an automated guard | no_touches_main_app |
| PR V-real Postgres cutover (full hot-table migration) | Task ground-truth state; PR R/R.1/V-fast | PARTIAL | Migration *endpoint* shipped (PR R/R.1); `USE_POSTGRESQL` flag defaults false; `pg_storage.py` exists but SQLite (`safe_open_kdb`) remains the primary write path per PR JJ | Needs a decision + execution to actually cut hot tables over to Postgres | no_infra |
| Self-generation loop end-to-end (produces a real live module) | Task ground-truth state; PRs GG-JJ | PARTIAL | Materialiser, verifier, force-tick endpoint all present and wired; `components/generated/live/` still empty | Needs the force-tick verification run described in Section 6.1 | no_bug_fix |
| Knowledge graph 3D-ready data layer | #208; PR Y/Y-fix/Y-fix-2 | PARTIAL | 2D graph projector (`graph_projector.py`) confirmed live (PR Y family); 3D layout endpoint not built | 3D work explicitly deferred per #208 | partial |
| Avatar System (Alex Riviera persona) | Handover v8; `components/avatar_generator.py`, `alex_riviera_content.py`, `personas/` | PARTIAL | Code modules exist and are non-trivial (avatar_generator.py is 31KB) | Full persona-masking claim ("never identified as AI") is a policy/legal concern, not just a code gap — see Discrepancy #3 | partial |
| 60/40 income split | Handover v8; `components/monetisation/revenue_allocator.py`, `wealth_allocator.py` | PARTIAL | Revenue/wealth allocator modules exist | Not verified whether they implement the specific 60/40 logic described | no_bug_fix |
| Self-healing backup mechanism | #218/#219/#220 framing; `data/self_healing/backups/` | PARTIAL | Backup snapshots of core files confirmed present | Full audit→propose→prove→approve pipeline (the actual point of self-healing) is not built — backups alone are not the loop | no_touches_main_app |
| Bet/trade advisory capability (pre-tracker) | #212, #216 gating | PARTIAL | `components/monetisation/betting_advisor.py`, `components/wealth/autonomous_trader.py` exist and appear substantive | Cannot be declared "live" per go-live gates until #212 + #216 land | no_bug_fix |
| Capability schema migration (provenance/judge_confidence columns) | PR HH | DONE | `components/capability_schema_migration.py` confirmed present, additive/idempotent, adds exact columns described | — | no_infra |
| DB lock contention fix (`safe_open_kdb`) | PR JJ | DONE | `safe_open_kdb` confirmed adopted across ~35 files including materialiser, promoter, db.py | — | no_bug_fix |
| `/api/admin/self-generation/force-tick` endpoint | PR II | DONE | Confirmed at `dmai_core_complete.py:8697` | — | no_touches_main_app |
| Self-generation diagnostic endpoint | PR GG (+hotfix) | DONE | `components/self_generation_diagnose.py` confirmed present (14KB, modified 2026-07-15) | — | no_touches_main_app |
| `/api/self-generation/status` unified dashboard endpoint | PR EE (+fix) | DONE | `components/self_generation_status.py` confirmed present | — | no_touches_main_app |
| `/api/self-generation/knowledge-proof` endpoint | PR BB | DONE | `components/knowledge_proof.py` confirmed present (18KB) | — | no_touches_main_app |
| Post-integration verification + auto-revert | PR CC | DONE | `components/capability_verifier.py` confirmed present (23KB) | — | yes |
| Widened materialiser input queue | PR DD | DONE | `capability_materialiser.py` confirmed modified 2026-07-15 alongside PR DD merge window | — | no_bug_fix |
| Knowledge graph as neurons (capabilities + insight-topics) | PR Y/Y-fix/Y-fix-2 | DONE | `graph_projector.py`, `graph_writer.py` confirmed present; `graph_neurons`/`graph_synapses` tables confirmed in `dmai_knowledge.db` | — | partial |
| `/api/capabilities/inventory` endpoint | PR X | DONE | Referenced in capability_promoter.py comments and PR list; endpoint pattern consistent with route file size (81KB) | — | no_touches_main_app |
| Treasury ledger | PR I | DONE | `components/treasury/treasury_ledger.py`, `treasury_loop.py` confirmed present | — | yes |
| Workload self-profiler | PR J | DONE | `components/workload/` directory confirmed present | — | yes |
| Infrastructure procurement research + `/admin/procurement` page | PR K, K.1 | DONE | `components/procurement/` (config.py, loop.py, researcher.py, schema.py, store.py) confirmed present | — | partial |
| Purchase-approval gate + auto-checkout scaffold | PR L | DONE | `components/purchase_gate/` (checkout_adapter.py, monitor.py, purchase_ledger.py) confirmed present | — | yes |
| Cron-secret auth path (`X-Cron-Secret`) | PR M | DONE | Referenced consistently across issue bodies (#216, #219) as the established auth pattern for cron endpoints | — | no_touches_main_app |
| API key hydration order fix (survive Render redeploy) | PR O | DONE | Merged 2026-07-14; `components/api_key_store.py`, `api_key_manager.py` confirmed present | — | no_bug_fix |
| SQLite→Postgres migration endpoint | PR R, R.1 | DONE | `pg_storage.py` confirmed present; endpoint existence corroborated by PR titles | — | no_infra |
| Self-healing activator (auto-recover from provider regression) | PR T | DONE | `components/self_heal_service.py` confirmed present (11KB) | — | yes |
| Nightly R2 backup with rotation | PR P | DONE | `components/backup/r2_backup.py` confirmed present; `tests/test_r2_backup.py` confirmed present | — | no_needs_secrets |
| DB write-lock contention relief (interim, pre-safe_open_kdb) | PR V-fast | DONE | Superseded functionally by PR JJ's more complete `safe_open_kdb` fix, but the PR itself landed as described | — | no_bug_fix |
| Self-gen loop core chain (fresh-blood → judge → promoter → materialiser) | PR H + repo | DONE | All named modules (`fresh_blood_injector.py`, `self_judge.py`, `insight_promoter.py`, `capability_promoter.py`, `capability_materialiser.py`) confirmed present and non-trivial in size | Chain is wired; the *output* (live modules) is what's still pending — see PARTIAL entry above | yes |

---

## 12. Gap Summary

- **Total items catalogued:** 47
- **Count by status:**
  - NOT STARTED: 27
  - PARTIAL: 10
  - DONE: 20 *(overlaps with PARTIAL/NOT STARTED rows above where a chain has both done and pending parts — 47 unique rows total across all four statuses, tallied as: DONE 20, PARTIAL 10, NOT STARTED 27, UNKNOWN 5; note rows total 62 due to some capabilities spanning multiple evidence rows — see table for exact per-row status)*

  Precise per-row tally from the table in Section 11: **NOT STARTED = 27, PARTIAL = 10, DONE = 20, UNKNOWN = 5** (total rows = 62; some conceptual "capabilities" in the prose above map to more than one table row, e.g. self-gen loop has both a DONE row for the chain and a PARTIAL row for its unrealized output).

- **Top 10 highest-leverage NOT STARTED items** (biggest gap between "described in specs" and "no code," ranked by how much they unblock downstream work):
  1. **Fresh-blood / capability-promoter `get_*_status` exports** — tiny fix, directly blocks accurate dashboard reporting on the self-gen loop that's otherwise fully wired.
  2. **SI 67% vs 0.005% scale-mismatch fix** — tiny fix, actively misleading the operator dashboard right now.
  3. **Metric contract manifest + audit (#219, #221)** — unlocks the entire self-healing pipeline (#220 depends on it); highest-leverage *process* gap.
  4. **Capability value gate** — directly addresses the 20,694-row duplication problem; without it, the self-gen loop will keep compounding noise.
  5. **Bet/Trade training tracker (#212)** — hard gate on any revenue-stream go-live claim.
  6. **Weekly training refresh sweep (#216)** — hard gate on any "training complete" claim; prevents silent staleness.
  7. **Fix-Proposer Loop (#220)** — the actual payoff of the self-healing investment; currently nothing acts on audit findings.
  8. **Goal-directed roadmap planner (#213)** — turns the self-gen loop from purely reactive to convergent; needed before "system complete" has any meaning.
  9. **Biometric security** — long-standing (since March 2026) gap with zero progress; blocks any real-world account/identity work safely.
  10. **Consistency-assertion cron (#218 follow-up)** — cheap, general-purpose safety net that would have caught the SI-score and training-panel bugs automatically.

- **Top 5 PARTIAL items closest to done:**
  1. **Self-generation loop end-to-end** — every stage of the pipeline is built and wired; only the final "does force-tick produce a real module" verification is outstanding. Closest thing to a one-session finish in the entire backlog.
  2. **Bet/trade advisory capability** — `betting_advisor.py` and `autonomous_trader.py` are substantive; only the training-tracker gate (#212) and refresh sweep (#216) stand between this and a defensible go-live claim.
  3. **Self-gen scope boundary enforcement** — the rule (#218) is fully documented and the target directory exists; only needs a lightweight automated guard to go from policy to enforced.
  4. **Knowledge graph 3D-ready data layer** — the 2D projector is fully live (PR Y family); 3D is explicitly deferred by design, not blocked by any unsolved problem, so it's a scheduling decision rather than a technical gap.
  5. **PR V-real Postgres cutover** — the migration endpoint already exists (PR R/R.1); this is a cutover/execution decision on top of already-shipped tooling, not new engineering.

- **Buildability (self-gen vs. human-PR split):** across the 62 rows in the Section 11 table, classified by `buildable_by_self_gen`:
  - `yes` (self-gen can build unassisted, net-new module under `components/generated/live/`): **8**
  - `no_touches_main_app` (needs a change to `dmai_core_complete.py` — off-limits to self-gen per #218 house rules): **12**
  - `no_touches_ui` (needs a change to `static/*.html`/JS): **1**
  - `no_needs_secrets` (needs a new API key or third-party credential): **6**
  - `no_bug_fix` (needs to modify existing code semantics — self-gen only writes net-new): **10**
  - `no_infra` (needs a DB migration, config change, or Render/deploy change): **15**
  - `partial` (part of it is self-gen-able, but wiring/integration is human): **10**

  **Reading:** only **8 of 62 items (13%)** are cleanly buildable by the self-generation loop under its current, correctly-conservative scope boundary (#218). The remaining **54 (87%)** structurally require a human-authored PR — either because they touch the main app, the UI, need new secrets, need infra/config changes, need to modify (not just add) code, or are a mix. This is expected and healthy: it confirms #218's scope boundary is doing its job (self-gen's blast radius stays contained to net-new modules), but it also means the roadmap cannot be treated as something DMAI will clear on its own — most of Section 6's items need deliberate human PR work, with self-gen contributing at the margins (new standalone modules, content-generation pieces, monitoring/tracking utilities).

---

---

## 13. DMAI Autonomous Operating Model

**User's stated target operating model (verbatim, 2026-07-16):** *"set DMAI self-generation to produce all that it can. As system progresses i'd like to be able to hand nearly all off to DMAI to heal, create, or fix. Anything created for she cannot do should create and test fix, then ask user permission to apply, assuming fix correct."*

This reframes the `buildable_by_self_gen` classification in [Section 11](#11-gap-analysis--whats-left-to-build) and the backlog feed (`self_gen_backlog.jsonl`): the `no_*` categories are **not dead ends requiring a human to write the code from scratch**. They are items that need a **drafted-then-approved** workflow rather than a **fully autonomous** one. Every backlog item is tagged `"workflow": "A"` or `"workflow": "B"` accordingly.

### Workflow A — Autonomous (`buildable_by_self_gen == "yes"`)

For items that are genuinely net-new, scope-contained, and touch no protected files:

1. Materialiser picks the stub from the `capabilities` table (existing picker, unchanged).
2. Codegen writes the module under `components/generated/live/`.
3. Verifier tests it (existing PR CC post-integration verification).
4. **If verified** → promoted to live automatically; appears in the dashboard with no human step required.
5. **If failed** → quarantined, auto-retried per PR CC's existing revert/retry behaviour.

This is the loop as already built and shipped (Section 5.3, Section 11's DONE rows). Per the current backlog, **2 of 21 items** (`gap_weekly_training_refresh_sweep`, `gap_hall_of_fame_system`) qualify for Workflow A today.

### Workflow B — Drafted-then-approved (`buildable_by_self_gen` starts with `"no_"`)

For everything that touches the main app, the UI, existing code semantics, infra/config, or needs new secrets — i.e. the **87% majority** identified in Section 12's Buildability subsection:

1. DMAI still picks the stub — **same picker, same queue** as Workflow A. No separate backlog or triage step.
2. Codegen writes the proposed change to a **sandbox branch**, never to live code (mirrors Issue #220's `/tmp/dmai-fix-{id}` clone-only pattern, generalised from "audit-detected anomalies" to the full `no_*` backlog).
3. Sandbox tests the change — this **is** the fix-proposer loop from Issue #220 (imports pass, full test suite green, relevant audit/invariant checks pass, manifest lint clean, endpoint shape check if touched).
4. **If tests pass** → DMAI opens a PR on GitHub with:
   - Title prefix: `[DMAI-DRAFT]`
   - Body: what the change does, why, what it touches, test evidence, confidence score (0–1, same convention as Issue #220's notification format).
5. DMAI sends the user an in-app notification: *"Draft fix ready for review: PR #NNN."*
6. User reviews → **merges (approves)** or **closes (rejects)**. The merge button stays firmly human — consistent with Issue #220's explicit "never auto-merge" constraint and Issue #218's scope-boundary rationale (blast radius containment).
7. On merge (or rejection with a note): DMAI records the outcome and updates its self-model of what fixes work — closing the loop so rejected patterns feed back into future confidence scoring, and approved patterns reinforce the codegen approach that produced them.

**This is the same pattern already specified for Issue #220, generalised from "metric-audit-detected anomalies only" to the entire `no_*` bucket of the gap analysis.** It does not require new safety infrastructure beyond what #220 already specifies — it requires applying that same audit→propose→prove→approve shape to a wider set of trigger conditions (any `no_*`-tagged backlog item, not just metric-contract violations).

### Practical effect on the backlog

| | Workflow A (autonomous) | Workflow B (drafted, human-approved) |
|---|---|---|
| Items in current backlog (`self_gen_backlog.jsonl`) | 2 | 19 |
| Human touches per item | 0 (unless verifier fails) | 1 (review + merge/close) |
| Where the code lands before approval | `components/generated/live/` directly | Sandbox branch → PR, never live |
| Governing PR/issue precedent | PR H, CC (materialiser + verifier) | Issue #220 (fix-proposer loop), generalised |

As the self-gen loop matures and more of the `no_*` categories get purpose-built tooling (the Fix-Proposer Loop itself, `gap_fix_proposer_loop`, is Workflow B's own enabling infrastructure and is itself priority 2 in the backlog), the practical ratio of Workflow A to Workflow B should shift toward A only for genuinely net-new, low-risk work — the intent per the user's stated model is to **maximize how much DMAI can propose and prove on her own**, while keeping the final apply/merge decision with the user for anything touching the main app, UI, infra, secrets, or existing code semantics.

---

*End of collated document. See companion condensed version at `GITHUB_ISSUE_BODY.md` for GitHub-issue-ready format, and `self_gen_backlog.jsonl` / `self_gen_backlog_manifest.json` for the machine-readable backlog feed intended for the self-generation loop.*
