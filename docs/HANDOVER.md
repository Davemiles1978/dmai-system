# DMAI — Master Handover Document

**Last updated:** 2026-06-26 19:00 BST
**Maintainer:** David Miles (milesd040@gmail.com)
**Purpose:** Single source of truth for resuming work on DMAI in any new chat or by any contributor. Paste the "Quick Start" section as your first message in a fresh thread.

---

## 1. Quick Start (paste this into a new chat)

> You are continuing the **DMAI** build for David Miles. DMAI is an autonomous, self-evolving revenue agent (Python/Flask, deployed on Render). Repo: `Davemiles1978/dmai-system`. Live URL: `https://dmai-web.onrender.com`. Master password header: `X-Master-Password: Talula.78`. **Before any work, read this file (`docs/HANDOVER.md`) in full and run the verification checks in §9.**

---

## 2. What DMAI Is

DMAI is a **production Flask app** (`dmai_core_complete:app`) that:

1. **Runs 8 background training loops** in-process — web learner, autonomous researcher, stage learner, graph evolution, vocab ingest, etc. (See `dmai_core_complete.py:7584-7611` for the PID-aware spawn guard.)
2. **Generates greyhound betting tips** — Timeform Master Ratings → softmax implied odds → settled against GBGB results. UK greyhounds only (never horse racing — explicit user rule).
3. **Federates 14+ AI providers** behind one router — Groq, Cerebras, Google AI Studio, OpenAI, Anthropic, DeepSeek, Mistral, Cohere, HuggingFace, Tavily, OpenRouter, Cloudflare AI, Perplexity, GitHub Models.
4. **Self-repairs** via a Kaizen auto-repair engine (two-pass: unbounded dead-letter sweep + capped AI repair).
5. **Tracks stage progression** Baby → Toddler → Child → Teen → Adult → Expert → Master by accumulating insights / capabilities / vocab.

**Naming rule:** `DMAI` = internal name (admin UI, code, operator chat). `Alex Riviera` = public-facing persona (social media, content publishing, books).

**Sports model rule:** Microfish is for stocks/FOREX only. Sports markets MUST go through `StatisticalGreyhoundModel` (still to be built — see §6 P0-2).

---

## 3. Production Environment

| Item | Value |
|---|---|
| Service URL | `https://dmai-web.onrender.com` |
| Render service ID | `srv-d6sd3chj16oc73emdj6g` (Frankfurt, standard 2GB) |
| Render owner ID | `tea-d6ldghvkijhs73b3ca7g` |
| Repo | `https://github.com/Davemiles1978/dmai-system` |
| Master password | `Talula.78` → header `X-Master-Password: Talula.78` |
| JWT secret env | `JWT_SECRET=f474445f826de33de01a238111d2d76b33e056489053a1c6334cec91aa77dcc5` |
| Start command (canonical) | `gunicorn dmai_core_complete:app --config gunicorn_config.py` |
| Render API auth | `api_credentials=["custom-cred:api.render.com"]` |
| GitHub auth | `api_credentials=["github"]` |
| Slack alert channel | `slack_direct` → `C0BCKABKVDG` |
| User Mac device ID | `1A0AABD1-4B6B-5278-9DE0-1D9753219D04` |

### ⚠️ Known Production-Critical Config Conflict

`render.yaml` currently specifies:

```
startCommand: gunicorn dmai_core_complete:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2 --max-requests 100 --max-requests-jitter 50
```

This **contradicts** `Procfile` + `gunicorn_config.py` (which correctly use 8 threads, timeout 300, no recycling). The `--max-requests 100` setting **recycles workers every 100 requests, killing all background training threads**. This is likely the root cause of the recurring "background loops stopped" issue.

**Fix (when ready):** edit `render.yaml` to either remove `startCommand` entirely (Render will use `Procfile`) or align it with `gunicorn_config.py` settings.

---

## 4. Architecture Map

```
dmai-system/
├── dmai_core_complete.py        # Main Flask app, all routes, app factory
├── dmai_api_routes.py           # Additional API route blueprints
├── dmai_complete_syllabus.py    # Learning curriculum content
├── dmai_syllabus_data.py        # Syllabus structured data
├── dmai_syllabus_knowledge.py   # Syllabus knowledge graph
├── gunicorn_config.py           # Canonical gunicorn config (USE THIS)
├── Procfile                     # web: gunicorn dmai_core_complete:app --config gunicorn_config.py
├── render.yaml                  # Render deploy spec (HAS CONFLICTING startCommand — see §3)
├── requirements.txt             # Python deps
│
├── components/
│   ├── microfish/               # PredictionEngine — stocks/FOREX only
│   ├── monetisation/            # betting_advisor.py, greyhound_runner.py, trader
│   ├── kaizen_auto_repair.py    # Self-repair engine (two-pass)
│   ├── meta_learner_fixed.py    # Current MetaLearner (older meta_learner.py removed)
│   ├── si_core.py               # KPI store + Self-Improvement core
│   ├── phase0/ phase1/ phase2/  # Bootstrap, foundation, scaling
│   ├── phase6/ phase7/          # Active phases (autonomy, master control)
│   ├── phase10/ phase11/        # AI Integration Hub, advanced consciousness
│   ├── alex_riviera/            # Public persona engine
│   ├── consciousness/           # Global Workspace / working memory
│   ├── content/, media/, music/ # Content generation pipelines
│   ├── funding/                 # SelfFundingExecutionReal
│   ├── trading/                 # (stub — needs rebuild, see §6 P0-3)
│   └── ...
│
├── static/
│   └── admin.html               # Single-page admin UI (3000+ lines)
│
├── data/
│   ├── dmai_knowledge.db        # Main SQLite DB (committed; 6.3 MB)
│   ├── avatars/                 # Alex Riviera reference + canonical specs
│   ├── art/, learning/          # Generated assets, training data
│   └── trading_mastery.db
│
├── docs/
│   ├── HANDOVER.md              # THIS FILE
│   └── recovered/               # Older spec docs (v1, v2, v2.1, roadmap)
│
├── scripts/                     # Operational scripts (check_system.py etc.)
├── tests/                       # Test suite
└── ops/                         # DevOps tooling
```

### Key File Locations

| Path | What it does |
|---|---|
| `dmai_core_complete.py:456-466` | KnowledgeGraph singleton init |
| `dmai_core_complete.py:669-677` | Microfish PredictionEngine wiring |
| `dmai_core_complete.py:1011-1015` | BettingAdvisor wiring (route greyhounds here) |
| `dmai_core_complete.py:1025-1039` | GreyhoundRunner start |
| `dmai_core_complete.py:7003-7065` | `/api/stage/analytics` route |
| `dmai_core_complete.py:7584-7611` | `_BG_SERVICES_STARTED` PID-aware guard |
| `dmai_core_complete.py:8202-8330` | Admin endpoints (force-start, db-query, db-bootstrap) |
| `components/kaizen_auto_repair.py` | Two-pass dead-letter + AI repair |
| `components/monetisation/betting_advisor.py` | Tip generator |
| `components/monetisation/greyhound_runner.py` | Timeform racecard + GBGB settler |
| `components/microfish/prediction_engine.py` | Microfish (KEEP for stocks/FOREX) |
| `components/si_core.py` | KPI store (note KPI-auth issue, §6 P1-7) |
| `static/admin.html` | Single-page admin UI |

### API Endpoint Reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | Healthcheck |
| `GET /api/training/status` | 8 background service status |
| `POST /api/admin/start-services?force=true` | Force-restart BG loops |
| `GET /api/admin/keys` | List provider keys (masked) |
| `POST /api/admin/db-rebuild` | Quarantine + recreate DB |
| `POST /api/admin/db-bootstrap` | Recreate `mf_*` tables |
| `POST /api/admin/db-query` | Read-only SELECT/PRAGMA/EXPLAIN |
| `GET /api/admin/db-integrity` | SQLite integrity check |
| `GET /api/stage/analytics` | Stage forecast + velocity |
| `GET /api/kaizen/status` | Pending/failed/executed counts |
| `POST /api/kaizen/auto-repair` | Trigger one repair cycle |
| `GET /api/jobs/<id>` | Job status |
| `POST /api/monetisation/tips/digest` | Daily tips JSON |
| `POST /api/monetisation/trader/digest` | Daily trader JSON |
| `GET /api/harvester/status` | AI provider key health check |

---

## 5. Recent Fixes (verified live, June 2026)

1. **Kaizen auto-repair drain** — was skipping 9,818 errors/cycle. Two-pass design: unbounded dead-letter sweep + capped AI repair (100/cycle). First run drained 7,018 of 10,458 proposals. (`components/kaizen_auto_repair.py`, commit `e2d924041`)
2. **Training loop PID-aware guard** — gunicorn fork was preventing background loops from spawning. Now resets `_BG_SERVICES_STARTED` when PID changes. (`dmai_core_complete.py:7584-7611`, commit `66e4bd139`)
3. **KnowledgeGraph None crashes in StageLearner** — singleton now instantiated and passed to all 9 consumer sites. (`dmai_core_complete.py:456-466`, commit `66e4bd139`)
4. **Stage Analytics frontend degraded-mode banner** — clicking the tab no longer silently fails when API returns `degraded: true`. (`static/admin.html:2933-2991`, commit `74b71e4af`)
5. **Admin endpoints added** — `/api/admin/db-query`, `/api/admin/db-bootstrap`, `/api/admin/start-services?force=true`. (`dmai_core_complete.py:8202-8330`, commit `66e4bd139`)
6. **Tipster digest cron** — fixed literal `$MASTER_PASSWORD` bug in single-quoted curl. Cron `6713f869`, schedule `0 7 * * *` UTC.
7. **Gemini endpoint 400 + capabilities fallback** — patch applied 2026-06-26 (see chat handoff doc).

---

## 6. Outstanding Work — Priority Order

### P0 — User-blocking (do FIRST)

1. **Build betting prediction tracking table + dashboard.** User decision (verbatim):
   > "Greyhound runner won't be placing any bets, I will. We are to track your predictions against the upcoming races for 2-7 days, to see how you do then, if results are good, I will start to put live bets down."

   Existing schema: `mon_tips` in `data/dmai_knowledge.db` (`betting_advisor.py:29-54`). Settles automatically from GBGB (`greyhound_runner.py:439-508`).

   **To build:**
   - Admin tab "Tip Tracking": pending predictions (next 7 days, sortable by race/EV/confidence), settled history (hit rate, ROI %, EV vs realised, calibration plot), Paper vs Live filter (via `notes` field)
   - New `mon_user_bets` table:
     ```sql
     CREATE TABLE mon_user_bets (
         id TEXT PRIMARY KEY,
         tip_id TEXT REFERENCES mon_tips(id),
         placed_at REAL NOT NULL,
         event_name TEXT,
         selection TEXT,
         actual_odds REAL,
         actual_stake REAL,
         bookmaker TEXT,
         status TEXT,            -- pending|won|lost|void|cashed_out
         settled_at REAL,
         actual_return REAL,
         profit_loss REAL,
         notes TEXT
     );
     ```
   - APIs: `GET /api/monetisation/tips/upcoming?days=7`, `GET /api/monetisation/tips/history`, `POST /api/monetisation/bets`, `GET /api/monetisation/bets/performance`

2. **StatisticalGreyhoundModel** — drop-in replacement for `prediction_engine.predict(...)` returning `{probability, confidence, rationale, id}`.
   Algorithm: weighted logistic on (Timeform z-score, trap bias by track, recent form, trainer 3-week strike rate). All fields already extracted by `greyhound_runner.py:135-168`.
   Routing in `BettingAdvisor.analyse_candidate()`: if `market.startswith('trap_')` or `market == 'greyhound_winner'` → `StatisticalGreyhoundModel`, else → Microfish.

3. **Trading dashboard rebuild** — currently a 38-line stub user has flagged 3 times. Target layout: tickers list left, chart centre, positions/orders/history tabs bottom, news + analysis right rail. Live positions from `aggressive_trader`, candlestick + MA20/50 + volume, news×price cross-check, P/L history, Microfish analysis right rail, auto-trade panel gated behind toggle (initially manual). Reference spec repos: `Davemiles1978/Trader` (canonical) and `Davemiles1978/algo-nexus-ai` (sister project).

### P1 — Cleanups user has flagged

4. **Stage Analytics tab still doesn't open** for user despite the frontend fix. Likely JWT auth flow on `/api/stage/analytics`. Open DevTools, watch Network tab. Also `/api/heartbeat` and `/api/knowledge/*` show "no metrics for last 7 days".

5. **Trader cron `$MASTER_PASSWORD` bug** — cron `5870d8a5` schedule `10 21 * * 1-5` UTC still uses literal `'$MASTER_PASSWORD'` in single quotes. Same fix as tipster.

6. **Security tab in Admin** — password change, attack monitoring, login attempt log, optional TOTP 2FA, IP allowlist toggle.

7. **SI KPIs reading 0** — KPIs that had figures yesterday now read 0. Logs show "Rejected KPI update for ... invalid/missing token" for `skill_acquisition_rate`, `transfer_learning_rate`, `zero_shot_success_count`, `agentic_capability_score`, `recursive_self_improvement_rate`, etc. The KPI write API requires an internal token the loops aren't passing. Fix in `components/si_core.py`.

8. **Orphaned neurons** (19 warnings + 11 info) — neurons like "DMAI Core", "Meta Controller", "SICore Engine" lack supporting insights. Self-referential. Extend kaizen/suggestion_executor with an `orphaned_neuron` auto-repair handler that generates an insight from the neuron name + docstring/comments.

9. **AI provider key failures** — Groq 401 (regenerate at console.groq.com/keys), Google AI Studio 400, OpenAI 429, DeepSeek 402, Anthropic 400. Verify each key; many likely rolled or hit free-tier limits. Hourly Provider Health Check cron already alerting (run #83 saw 2 core failures on 2026-06-26).

10. **AI-native image/video generation** — user note:
    > "when I ask for image I get a text prompt but no image. DMAI needs to create images and videos from within her own coding, then display it on the screen to view output (initially all created will need to pass via me for approval. Once she is consistently creating high quality, I'll then switch over for DMAI to automatically upload"

    Wire up Stability AI / fal.ai / Replicate / HF Inference SD3 → persist to `static/generated/` → admin approval queue.

### P2 — Hygiene

11. Clean up 7 quarantined `dmai_knowledge.db.malformed_*` files on disk.
12. Resolve the `render.yaml` startCommand conflict (see §3 — likely cause of recurring background-thread loss).
13. Betfair `USERNAME`/`PASSWORD` for Tier 2 live staking — **DEFER** until user flips from manual to auto-staking.

---

## 7. Active Scheduled Crons

| ID | Schedule (UTC) | Purpose | Status |
|---|---|---|---|
| `19763951` | `19 * * * *` | DMAI Production Health Check (Slack alerts) | ✅ Working |
| `6713f869` | `0 7 * * *` | DMAI tipster digest | ✅ Fixed |
| `5870d8a5` | `10 21 * * 1-5` | DMAI trader daily digest | ❌ Still has `$MASTER_PASSWORD` bug (P1-5) |
| Hourly Provider Health Check | every hour | Alerts on 401/402 core provider failures | ✅ Working (alerted run #83) |
| DMAI Friday Graph Evolution | Fridays | Graph evolution job | ✅ Working |

---

## 8. Hard Rules (User-Stated)

- **Never re-request a credential the user has already submitted.** Run `list_credentials` first.
- **Microfish is NOT for any sports market.** Sports → `StatisticalGreyhoundModel`. Microfish is stocks/FOREX/ETF only.
- **Greyhounds only.** Never horse racing.
- **All bets MANUAL** until user explicitly flips to auto-staking. Don't ask for Betfair username/password.
- **Use master password `Talula.78` directly** in cron curl commands (not env-var substitution inside single quotes).
- **When something works, verify with a live curl.** Don't claim success based on deploy status alone.
- **Notify prominently on +EV/hot tips** (push or in-app, not just logs).
- **Render deployment is temporary** — user plans to migrate to a self-hosted home hub once funded. Keep architecture portable.

---

## 9. Verification Checklist for a New Session

Run these immediately after picking up the project:

```bash
# 1. Health
curl -s https://dmai-web.onrender.com/health

# 2. Background loops alive?
curl -s -H "X-Master-Password: Talula.78" \
  https://dmai-web.onrender.com/api/training/status | jq '.active_count'
# Expect: 8

# 3. Provider health
curl -s -H "X-Master-Password: Talula.78" \
  https://dmai-web.onrender.com/api/harvester/status | jq '.summary'

# 4. KPI auth issue check (P1-7)
curl -s -H "X-Master-Password: Talula.78" \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT kpi_name, MAX(value), MAX(recorded_at) FROM kpi_history GROUP BY kpi_name ORDER BY MAX(recorded_at) DESC LIMIT 20"}' \
  https://dmai-web.onrender.com/api/admin/db-query

# 5. Tip pipeline alive?
curl -s -H "X-Master-Password: Talula.78" \
  -X POST https://dmai-web.onrender.com/api/monetisation/tips/digest | jq '.'
```

---

## 10. Phase 12 Strategic Roadmap (Revenue × Stability Prioritised)

Phases 0–11 built the foundation (bootstrap → autonomy → AI integration hub). **Phase 12 is the monetisation activation phase.** Everything is scored on two axes: **Revenue Impact (R)** and **System Stability Impact (S)**, each 1–5.

### Phase 12 — "Revenue Activation" (Weeks 1–4)

Focus: prove the model with paper bets, then unlock live revenue without further code work.

| # | Feature | R | S | Why now |
|---|---|---|---|---|
| 12.1 | Betting prediction tracking + paper-bet dashboard (§6 P0-1) | 5 | 4 | Cannot prove model → cannot place real money. Direct revenue gate. |
| 12.2 | `StatisticalGreyhoundModel` replacing Microfish for sports (§6 P0-2) | 5 | 5 | Microfish-on-sports is producing the bad predictions blocking 12.1. |
| 12.3 | Resolve `render.yaml` startCommand conflict (§3) | 0 | 5 | Currently silently killing background loops every 100 requests. Cheapest highest-impact fix in the repo. |
| 12.4 | KPI auth fix (§6 P1-7) | 1 | 5 | Without KPI data, stage progression is blind. Self-improvement loop is degraded. |
| 12.5 | Trader cron `$MASTER_PASSWORD` fix (§6 P1-5) | 2 | 3 | Trivial 5-min fix unblocks daily trader digest. |

**Phase 12 exit criteria:** 2–7 days of paper-bet tracking with > 50% hit-rate at +EV, system stable for 7 consecutive days with all 8 loops alive, KPIs populating.

### Phase 13 — "Trading Revenue Unlock" (Weeks 5–8)

Focus: rebuild trading from stub to production. Activate Alpaca paper trades, then live with conservative caps.

| # | Feature | R | S | Why |
|---|---|---|---|---|
| 13.1 | Trading dashboard rebuild (§6 P0-3) | 4 | 3 | Stub blocking visibility; canonical spec exists in `Davemiles1978/Trader`. |
| 13.2 | Microfish paper-trade evaluation (30-day backtest report) | 4 | 3 | Mirror of 12.1 for stocks. Same prove-then-deploy pattern. |
| 13.3 | News × price cross-check signal layer | 3 | 2 | Differentiator vs simple TA bots. Free LLM inference via Groq/Cerebras already wired. |
| 13.4 | Auto-trade gating + risk caps | 4 | 4 | Required before any live cash; copy risk rules from `Davemiles1978/algo-nexus-ai`. |

### Phase 14 — "Alex Riviera Public Persona" (Weeks 9–14)

Focus: switch on the public-facing revenue channels (content, affiliate, modelling-adjacent).

| # | Feature | R | S | Why |
|---|---|---|---|---|
| 14.1 | AI-native image/video generation w/ approval queue (§6 P1-10) | 4 | 2 | Without this, Alex Riviera content pipeline is gated. |
| 14.2 | Avatar consistency engine production-grade | 3 | 2 | Already partially built (`components/media/AvatarConsistencyEngine.py`). Finish + ship. |
| 14.3 | Affiliate link router + click-through tracking | 4 | 2 | Direct monetisation. Low complexity. |
| 14.4 | Social posting automation (X, IG, TikTok) behind approval queue | 4 | 2 | Existing content engine + approval queue = ship. |
| 14.5 | Book/KDP publishing pipeline | 3 | 1 | Env vars already in `render.yaml` (`KDP_EMAIL`, `KDP_PASSWORD`); needs orchestrator. |

### Phase 15 — "Property + Real-World Revenue" (Weeks 15–22)

Focus: connect DMAI to the user's stated interests in property and sports management.

| # | Feature | R | S | Why |
|---|---|---|---|---|
| 15.1 | Property scanner: Rightmove/Zoopla deal-flow agent | 4 | 2 | User has stated property interest. Below-market-value detector. |
| 15.2 | Boxing/sports management deal-discovery agent | 3 | 1 | User's domain. Niche, defensible. |
| 15.3 | Multi-revenue stream KPI consolidator | 3 | 3 | Single dashboard for betting + trading + content + affiliate. |

### Phase 16+ — "Autonomy and Migration" (Weeks 22+)

| # | Feature | R | S | Why |
|---|---|---|---|---|
| 16.1 | Local home-hub migration plan (off Render) | 0 | 5 | User intent. Avoid lock-in. Phase 11 architecture already portable. |
| 16.2 | Security tab + 2FA + IP allowlist (§6 P1-6) | 1 | 5 | Required before exposing publicly. |
| 16.3 | Orphaned-neuron auto-repair handler (§6 P1-8) | 0 | 3 | Cleans up the self-referential warnings. |
| 16.4 | Stage-progression-driven feature unlocks | 2 | 2 | Use existing Baby→Master ladder as feature gates. |
| 16.5 | Self-evolving codegen (write→test→PR→merge) | 5 | 1 | Highest revenue potential, highest risk. Defer until everything else stable. |

### Ordering Rationale

The hard ordering principle is: **stability fixes that gate revenue come before revenue features that depend on them.** That's why 12.3 (render.yaml fix, R=0) outranks 13.x and 14.x: nothing else stays alive without it. 12.1+12.2 are the only pair where revenue and stability both score 5/5 — they are the highest-leverage work in the entire roadmap.

---

## 11. Top 3 Technical Risks + Mitigations

### Risk 1 — Single-process gunicorn worker is a single point of failure

**What.** All 8 background training loops run as daemon threads inside one gunicorn worker. The current `render.yaml` recycles the worker every 100 requests (max-requests=100), killing every thread. Even the canonical config (`max_requests=0`) keeps everything in one process — one OOM, one segfault, one runaway thread, and the entire learning system stops. Render's standard 2GB tier offers limited headroom for the DB + 14 provider clients + knowledge graph.

**Symptoms already seen.** Background loops "stopped" requiring `/api/admin/start-services?force=true`. The PID-aware guard was added because of exactly this class of failure.

**Mitigation.**
1. **Immediate:** Remove the `--max-requests 100` setting from `render.yaml` (resolves §3 conflict). Confirms canonical 8-thread, no-recycle config.
2. **Short-term (Phase 12):** Add a watchdog cron — every 5 minutes hit `/api/training/status`; if `active_count < 8`, POST to `/api/admin/start-services?force=true` and alert Slack. Already partially in place via the hourly health check — drop to 5-minute cadence and add auto-restart.
3. **Medium-term (Phase 13):** Externalise the heaviest loops (graph evolution, vocab ingest) into separate Render background workers. They communicate with the main app via the existing SQLite DB + a job queue table.
4. **Long-term (Phase 16):** When migrating to home-hub, run loops as `systemd` services with auto-restart and per-service resource caps.

### Risk 2 — Concentrated DB risk (single committed SQLite file)

**What.** `data/dmai_knowledge.db` (6.3 MB committed to git) holds the knowledge graph, KPIs, tip history, learning state, and stage progression. It is the source of truth for everything DMAI "knows". A single corruption event (already happened — see the 7 quarantined `*.malformed_*` files) erases learning progress. SQLite + threaded gunicorn + concurrent writers is a known sharp edge, and the existing thread-safety fix (commit `1da660c6b`) suggests the issue has bitten before.

**Symptoms already seen.** 7 quarantined malformed DB files on disk. The `db-rebuild` / `db-bootstrap` endpoints exist precisely because corruption recovery is a routine operational concern.

**Mitigation.**
1. **Immediate:** Enable SQLite WAL mode at startup (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`) — vastly reduces corruption risk under concurrent reads/writes. One-line patch at DB-init time.
2. **Short-term (Phase 12):** Schedule a 6-hourly online backup using SQLite's `.backup` API (not file copy) into `data/backups/dmai_knowledge.db.YYYYMMDD_HHMM` with a 14-day rolling retention. Add `POST /api/admin/db-restore-latest`.
3. **Medium-term (Phase 13):** Split write-heavy tables (KPI history, kaizen proposals, tip history) into separate DB files. Keep `dmai_knowledge.db` for the knowledge graph alone. Reduces blast radius — a kaizen-loop corruption can no longer take out the knowledge graph.
4. **Long-term (Phase 16):** Migrate to PostgreSQL (Render add-on or self-hosted). The DATABASE_URL env var already exists in `render.yaml`, so the code path is anticipated. Use `pg_storage.py` (already in `components/`) as the abstraction layer.

### Risk 3 — Multi-provider AI dependency with cascading auth failures

**What.** DMAI relies on 14+ external AI providers. Many are free-tier (Groq, Cerebras, Google AI Studio, HF, Cohere) and rotate keys, hit rate limits, or change endpoints without notice (the 2026-06-26 Gemini endpoint 400 is a recent example). Currently, 4 of 14 providers are returning 401/402 simultaneously (Groq, Google AI Studio, OpenAI, DeepSeek). If the wrong 2 providers fail at once, chat goes offline, the autonomous researcher stops, and the kaizen AI repair stage stalls.

**Symptoms already seen.** Hourly Provider Health Check run #83 fired Condition A (core provider failure). Repeated Groq 401s. Quarterly key rotations and free-tier quota exhaustion.

**Mitigation.**
1. **Immediate:** Define a 3-tier provider priority in `AIIntegrationHub`:
   - **Tier A (paid, must-stay-alive):** Anthropic, OpenAI (when funded)
   - **Tier B (free, primary):** Groq, Cerebras, Google AI Studio
   - **Tier C (free, fallback):** HF, Cohere, OpenRouter, Cloudflare, GitHub Models
   Router must always route to the highest healthy tier. Already partially structured — formalise it.
2. **Short-term (Phase 12):** Per-provider circuit breaker (`circuit_breaker.py` already exists at root — wire it into the hub). After N consecutive failures, mark provider as `cooldown` for 1 hour; don't retry until cooldown expires. Reduces alert noise and protects free-tier quotas.
3. **Medium-term (Phase 13):** Automate API key rotation detection — when a 401 fires, attempt to call the provider's "create new key" endpoint (where available, e.g. Groq's API) and update Render env via the Render API. Where not automatable, fire a high-priority Slack alert with a one-click renewal URL.
4. **Long-term (Phase 14+):** Self-host one open-weights model (Llama 3.1 70B or Qwen 2.5 72B) on the home-hub as a permanent fallback. Removes external dependency for core reasoning. Bandwidth: ~40 GB model + 64 GB RAM.

---

## 12. Recent Codebase Cleanup

See PR `cleanup/purge-redundant-modules` (June 2026). Removed:

- Historical backup dirs (`components/backup_*` — 800 KB)
- Superseded `meta_learner.py` (kept `meta_learner_fixed.py`)
- 11 unreferenced root scripts (`dmai_smart_endpoint_stable.py`, `dmai_core_patch.py`, etc.)
- 5 unreferenced phase task dirs (`phase3/4/5/8/9` — 364 KB)

Net repo size: 47 MB → 46 MB. No functional change — the running system is unaffected. All key modules verified parseable post-cleanup.

---

*End of HANDOVER.md*
