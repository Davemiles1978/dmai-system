-- ============================================================================
-- DMAI knowledge DB schema  --  scripts/schema.sql
-- ============================================================================
-- Source: dumped from live prod via in-code DDL recovery on 2026-06-30.
--   Recovered by scanning the application source for
--   `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` statements
--   (components/*.py + dmai_core_complete.py + root helpers). String literals
--   are reconstructed with Python's tokenizer (handling implicit
--   adjacent-literal concatenation and triple-quoted executescript blocks),
--   then each statement is isolated with a balanced-paren walker -- the same
--   approach components/schema_bootstrap.py uses at boot.
--   Table set validated against the live-prod salvage manifest
--   (POST /api/admin/db-salvage, captured 2026-06-30T09:35Z): all 76 tables
--   present, 0 unresolved.
--
-- Branch base SHA: f6fba6094dee83b175c662fcc9de3b1726a9e002 (main)
--
-- This file is the SOURCE OF TRUTH for the schema. It is consumed by:
--   - scripts/db_health.py   (schema-drift check)
--   - scripts/db_migrate.py  (idempotent CREATE-missing-tables step)
-- Every statement is IF NOT EXISTS so the migrator is fully idempotent.
-- ============================================================================

PRAGMA foreign_keys = ON;


-- admin_api_keys  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS admin_api_keys (
    provider_id TEXT PRIMARY KEY,
    api_key TEXT NOT NULL,
    updated_at TEXT
);

-- ai_systems  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS ai_systems (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    organization TEXT NOT NULL,
    first_release_date TEXT,
    status TEXT DEFAULT 'tracked',
    tracking_since TEXT DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    website TEXT,
    category TEXT DEFAULT 'llm'
);

-- api_keys  (recovered from components/api_key_store.py)
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    source TEXT DEFAULT 'manual',
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_validated TEXT,
    last_used TEXT,
    health_status TEXT DEFAULT 'unknown',
    call_count INTEGER DEFAULT 0,
    notes TEXT,
    is_active INTEGER DEFAULT 1
);

-- at_exits  (recovered from components/wealth/exit_manager.py)
CREATE TABLE IF NOT EXISTS at_exits (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')), symbol TEXT NOT NULL, qty REAL, entry_avg REAL, exit_price REAL, pnl_usd REAL, pnl_pct REAL, hold_days REAL, reason TEXT NOT NULL, live INTEGER, result_json TEXT);

-- at_pending  (recovered from components/wealth/autonomous_trader.py)
CREATE TABLE IF NOT EXISTS at_pending (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT (datetime('now')), symbol TEXT NOT NULL, confidence REAL, ev REAL, tier TEXT, status TEXT NOT NULL DEFAULT 'pending', resolved_ts TEXT, result_json TEXT);

-- at_position_high  (recovered from components/wealth/exit_manager.py)
CREATE TABLE IF NOT EXISTS at_position_high (symbol TEXT PRIMARY KEY, session_high REAL, updated_at TEXT DEFAULT (datetime('now')));

-- at_state  (recovered from components/wealth/autonomous_trader.py)
CREATE TABLE IF NOT EXISTS at_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    enabled         INTEGER NOT NULL DEFAULT 0,
    tier            TEXT    NOT NULL DEFAULT 'conservative',
    last_tick_ts    TEXT,
    last_tick_note  TEXT,
    today_date      TEXT,
    today_deployed_pct REAL NOT NULL DEFAULT 0,
    today_trades    INTEGER NOT NULL DEFAULT 0,
    today_open_eq   REAL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- at_ticks  (recovered from components/wealth/autonomous_trader.py)
CREATE TABLE IF NOT EXISTS at_ticks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now')),
    market_open     INTEGER NOT NULL,
    tier            TEXT    NOT NULL,
    live            INTEGER NOT NULL,
    signals_seen    INTEGER NOT NULL DEFAULT 0,
    signals_passed  INTEGER NOT NULL DEFAULT 0,
    trades_placed   INTEGER NOT NULL DEFAULT 0,
    note            TEXT
);

-- at_tier_changes  (recovered from components/wealth/autonomous_trader.py)
CREATE TABLE IF NOT EXISTS at_tier_changes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now')),
    from_tier       TEXT    NOT NULL,
    to_tier         TEXT    NOT NULL,
    reason          TEXT    NOT NULL
);

-- at_trades  (recovered from components/wealth/autonomous_trader.py)
CREATE TABLE IF NOT EXISTS at_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now')),
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    qty             REAL,
    confidence      REAL,
    ev              REAL,
    tier            TEXT    NOT NULL,
    live            INTEGER NOT NULL,
    result_json     TEXT
);

-- brain_entries  (recovered from components/brain/brain_loader.py)
CREATE TABLE IF NOT EXISTS brain_entries (id TEXT PRIMARY KEY, domain TEXT NOT NULL, domain_label TEXT, topic TEXT NOT NULL, content TEXT NOT NULL, source_url TEXT NOT NULL, tier TEXT DEFAULT 'canonical', version TEXT, loaded_at TEXT DEFAULT (datetime('now')));

-- brain_load_log  (recovered from components/brain/brain_loader.py)
CREATE TABLE IF NOT EXISTS brain_load_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')), seed_version TEXT, entries_loaded INTEGER, entries_skipped INTEGER, notes TEXT);

-- capabilities  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS capabilities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,               -- 'class', 'function', etc.
    capability_type TEXT NOT NULL,    -- 'funding', 'replication', etc.
    description TEXT,
    source_url TEXT,
    source_repo TEXT,
    file_path TEXT,
    runtime_mode TEXT,                -- 'autonomous', 'ondemand'
    language TEXT,
    methods TEXT,                     -- JSON array
    is_async INTEGER DEFAULT 0,
    args TEXT,                        -- JSON array
    integrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- conv_messages  (recovered from components/conversation_memory.py)
CREATE TABLE IF NOT EXISTS conv_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT
);

-- conv_sessions  (recovered from components/conversation_memory.py)
CREATE TABLE IF NOT EXISTS conv_sessions (
    session_id TEXT PRIMARY KEY,
    started_ts TEXT NOT NULL,
    last_ts TEXT NOT NULL,
    msg_count INTEGER NOT NULL DEFAULT 0,
    title TEXT
);

-- conversations  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    user TEXT,
    message TEXT,
    response TEXT,
    is_task INTEGER DEFAULT 0
);

-- core_knowledge  (recovered from components/knowledge_manager.py)
CREATE TABLE IF NOT EXISTS core_knowledge (
    id TEXT PRIMARY KEY,
    topic TEXT UNIQUE,
    category TEXT,
    content TEXT,
    mastery_level REAL DEFAULT 1.0,
    last_reviewed TIMESTAMP,
    created_at TIMESTAMP,
    required_for_system BOOLEAN DEFAULT 1,
    metadata TEXT
);

-- encyclopaedia  (recovered from components/knowledge/vocabulary_ingester.py)
CREATE TABLE IF NOT EXISTS encyclopaedia (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    categories TEXT,
    url TEXT,
    domain TEXT DEFAULT 'general',
    word_count INTEGER DEFAULT 0,
    source TEXT DEFAULT 'unknown',
    created_at TEXT NOT NULL
);

-- evolution_cycles  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS evolution_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_number INTEGER,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    insights_created INTEGER DEFAULT 0,
    synapses_created INTEGER DEFAULT 0,
    consciousness_level REAL
);

-- evolution_state  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS evolution_state (
    id TEXT PRIMARY KEY DEFAULT 'core',
    consciousness REAL DEFAULT 0,
    neurons INTEGER DEFAULT 0,
    synapses INTEGER DEFAULT 0,
    evolution_cycles INTEGER DEFAULT 0,
    evolution_count INTEGER DEFAULT 0,
    last_update TEXT
);

-- experiments  (recovered from components/self_optimizer.py)
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    hypothesis TEXT,
    change_code TEXT,
    expected_outcome TEXT,
    test_results TEXT,
    success BOOLEAN,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- funding_avenues  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS funding_avenues (
    id TEXT PRIMARY KEY,
    name TEXT,
    progress REAL DEFAULT 0,
    completed INTEGER DEFAULT 0,
    updated_at TEXT
);

-- funding_concepts  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS funding_concepts (
    id TEXT PRIMARY KEY,
    learned_at TEXT
);

-- funding_state  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS funding_state (
    id TEXT PRIMARY KEY DEFAULT 'core',
    completed_avenues TEXT DEFAULT '[]',
    concepts_learned INTEGER DEFAULT 0,
    concepts_total INTEGER DEFAULT 0,
    learning_active INTEGER DEFAULT 0,
    training_complete INTEGER DEFAULT 0,
    progress REAL DEFAULT 0,
    updated_at TEXT
);

-- genealogy_predictions  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS genealogy_predictions (
    id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL,
    predicted_version TEXT NOT NULL,
    predicted_date TEXT,
    predicted_capabilities TEXT,
    predicted_architecture TEXT,
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending',
    actual_version TEXT,
    actual_date TEXT,
    lead_time_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(system_id) REFERENCES ai_systems(id) ON DELETE CASCADE
);

-- improvement_patterns  (recovered from components/self_optimizer.py)
CREATE TABLE IF NOT EXISTS improvement_patterns (
    pattern_id TEXT PRIMARY KEY,
    component TEXT,
    pattern_type TEXT,
    pattern_data TEXT,
    success_rate REAL,
    times_tested INTEGER,
    created_at TIMESTAMP
);

-- insight_topics  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS insight_topics (
    insight_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    PRIMARY KEY(insight_id, topic_name),
    FOREIGN KEY(insight_id) REFERENCES insights(id) ON DELETE CASCADE,
    FOREIGN KEY(topic_name) REFERENCES topics(name) ON DELETE CASCADE
);

-- insights  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    insight_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entities TEXT NOT NULL,           -- JSON array
    relationship TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_topic TEXT NOT NULL,
    target_topic TEXT NOT NULL,
    source_url TEXT,
    source_title TEXT,
    source_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- integration_queue  (recovered from components/integration/repo_integration_engine.py)
CREATE TABLE IF NOT EXISTS integration_queue (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    name TEXT,
    priority INTEGER DEFAULT 2,
    level INTEGER DEFAULT 1,
    category TEXT,
    replaces TEXT,
    augments TEXT,
    safety_required INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',
    added_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    approved_at TEXT,
    error TEXT,
    classification TEXT
);

-- integration_registry  (recovered from components/integration/repo_integration_engine.py)
CREATE TABLE IF NOT EXISTS integration_registry (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    name TEXT,
    level INTEGER DEFAULT 1,
    category TEXT,
    status TEXT,
    completed_at TEXT,
    data TEXT
);

-- integrity_flags  (recovered from components/knowledge/integrity_checker.py)
CREATE TABLE IF NOT EXISTS integrity_flags (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    flag_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    entity_id TEXT,
    entity_type TEXT,
    title TEXT NOT NULL,
    detail TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    resolved INTEGER DEFAULT 0,
    resolved_at TEXT,
    resolution_note TEXT,
    created_at TEXT NOT NULL
);

-- integrity_reports  (recovered from components/knowledge/integrity_checker.py)
CREATE TABLE IF NOT EXISTS integrity_reports (
    id TEXT PRIMARY KEY,
    run_at TEXT NOT NULL,
    total_checked INTEGER DEFAULT 0,
    total_flags INTEGER DEFAULT 0,
    critical INTEGER DEFAULT 0,
    warning INTEGER DEFAULT 0,
    info INTEGER DEFAULT 0,
    resolved INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    report_json TEXT NOT NULL,
    summary TEXT
);

-- learning_outcomes  (recovered from components/meta_learner.py)
CREATE TABLE IF NOT EXISTS learning_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    strategy_used TEXT,
    weight_before INTEGER,
    weight_after INTEGER,
    response_quality REAL,
    time_spent REAL,
    timestamp TIMESTAMP
);

-- learning_patterns  (recovered from components/meta_learner.py)
CREATE TABLE IF NOT EXISTS learning_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT,
    pattern_data TEXT,
    success_rate REAL,
    times_used INTEGER,
    created_at TIMESTAMP
);

-- learning_progress  (recovered from components/trading/mastery_system.py)
CREATE TABLE IF NOT EXISTS learning_progress (
    trading_type TEXT PRIMARY KEY,
    mastery_level REAL,
    papers_studied INTEGER,
    strategies_implemented INTEGER,
    backtests_run INTEGER,
    last_update REAL
);

-- mf_actions  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mf_actions (
    prediction_id TEXT NOT NULL,
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    content TEXT,
    target_id TEXT,
    round_num INTEGER NOT NULL,
    ts REAL NOT NULL
);

-- mf_agents  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mf_agents (
    prediction_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    persona_json TEXT NOT NULL,
    platform TEXT,
    PRIMARY KEY (prediction_id, agent_id)
);

-- mf_entities  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mf_entities (
    prediction_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT,
    PRIMARY KEY (prediction_id, entity_id)
);

-- mf_predictions  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mf_predictions (
    id TEXT PRIMARY KEY,
    requirement TEXT NOT NULL,
    seed_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    verdict_json TEXT,
    created_at REAL NOT NULL DEFAULT 0,
    completed_at REAL
);

-- mf_relations  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mf_relations (
    prediction_id TEXT NOT NULL,
    rel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT
);

-- mon_alerts  (recovered from components/monetisation/notifier.py)
CREATE TABLE IF NOT EXISTS mon_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (datetime('now')),
    category    TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    body        TEXT,
    meta_json   TEXT,
    delivered   INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);

-- mon_bill_payments  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mon_bill_payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bill_id TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    ts REAL NOT NULL,
    notes TEXT
);

-- mon_bills  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mon_bills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    cadence TEXT NOT NULL DEFAULT 'monthly',
    next_due REAL,
    auto_pay INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0
);

-- mon_income_events  (recovered from components/monetisation/revenue_allocator.py)
CREATE TABLE IF NOT EXISTS mon_income_events (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    ts REAL NOT NULL,
    raw_json TEXT
);

-- mon_tips  (recovered from components/monetisation/betting_advisor.py)
CREATE TABLE IF NOT EXISTS mon_tips (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    market TEXT,
    selection TEXT NOT NULL,
    bookmaker TEXT,
    decimal_odds REAL NOT NULL,
    model_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_value REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    recommended_stake REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    rationale TEXT,
    prediction_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|placed|skipped|won|lost|void
    placed_at REAL,
    settled_at REAL,
    actual_stake REAL,
    profit_loss REAL,
    notes TEXT,
    created_at REAL NOT NULL
);

-- mon_tracking_picks  (recovered from components/monetisation/betting_advisor.py)
CREATE TABLE IF NOT EXISTS mon_tracking_picks (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    market TEXT NOT NULL DEFAULT 'trap_winner',
    selection TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    model_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_value REAL NOT NULL,
    rationale TEXT,
    prediction_id TEXT,
    -- outcome: pending | won | lost | void  (settled by runner against GBGB)
    outcome TEXT NOT NULL DEFAULT 'pending',
    settled_at REAL,
    -- paper P/L if you had staked 1 unit at decimal_odds (informational only)
    paper_pl REAL,
    notes TEXT,
    created_at REAL NOT NULL,
    UNIQUE(event_name, market)
);

-- mon_user_bets  (recovered from components/monetisation/betting_advisor.py)
CREATE TABLE IF NOT EXISTS mon_user_bets (
    id TEXT PRIMARY KEY,
    tip_id TEXT,
    placed_at REAL NOT NULL,
    event_name TEXT NOT NULL,
    market TEXT,
    selection TEXT NOT NULL,
    actual_odds REAL NOT NULL,
    actual_stake REAL NOT NULL,
    bookmaker TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    settled_at REAL,
    actual_return REAL,
    profit_loss REAL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    notes TEXT,
    created_at REAL NOT NULL
);

-- mon_wallet_ledger  (recovered from components/monetisation/revenue_allocator.py)
CREATE TABLE IF NOT EXISTS mon_wallet_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet TEXT NOT NULL,
    delta REAL NOT NULL,
    balance_after REAL NOT NULL,
    event_id TEXT,
    reason TEXT NOT NULL,
    ts REAL NOT NULL
);

-- mon_wallets  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mon_wallets (
    name TEXT PRIMARY KEY,
    balance REAL NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'GBP',
    updated_at REAL NOT NULL DEFAULT 0
);

-- mon_wealth_deployments  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS mon_wealth_deployments (
    id TEXT PRIMARY KEY,
    total_amount REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    basket_name TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    status TEXT NOT NULL,
    ts REAL NOT NULL,
    notes TEXT
);

-- optimal_strategies  (recovered from components/meta_learner.py)
CREATE TABLE IF NOT EXISTS optimal_strategies (
    topic_category TEXT,
    strategy TEXT,
    effectiveness REAL,
    sample_size INTEGER,
    last_updated TIMESTAMP,
    PRIMARY KEY (topic_category, strategy)
);

-- performance  (recovered from components/trading/mastery_system.py)
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trading_type TEXT,
    algorithm TEXT,
    date TEXT,
    daily_pnl REAL,
    win_rate REAL,
    sharpe REAL,
    max_drawdown REAL,
    trades_count INTEGER
);

-- performance_baselines  (recovered from components/self_optimizer.py)
CREATE TABLE IF NOT EXISTS performance_baselines (
    component TEXT PRIMARY KEY,
    metric_name TEXT,
    baseline_value REAL,
    current_value REAL,
    target_value REAL,
    last_updated TIMESTAMP
);

-- persona  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS persona (
    id TEXT PRIMARY KEY DEFAULT 'dmai',
    traits TEXT DEFAULT '{}',
    speaking_style TEXT DEFAULT 'emerging',
    emotional_state TEXT DEFAULT 'neutral',
    consciousness_level REAL DEFAULT 0,
    last_update TEXT
);

-- persona_usage  (recovered from components/personas/persona_registry.py)
CREATE TABLE IF NOT EXISTS persona_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')), persona TEXT, component TEXT, task TEXT);

-- personas  (recovered from components/personas/persona_registry.py)
CREATE TABLE IF NOT EXISTS personas (name TEXT PRIMARY KEY, label TEXT, scope TEXT, used_by_json TEXT, brain_domains_json TEXT, model_pref_json TEXT, system_prompt TEXT, decision_rules_json TEXT, version TEXT, updated_at TEXT DEFAULT (datetime('now')));

-- processed_repos  (recovered from components/integration/repo_processor.py)
CREATE TABLE IF NOT EXISTS processed_repos (
    repo_name TEXT PRIMARY KEY,
    processed_at TIMESTAMP,
    status TEXT
);

-- se_edits  (recovered from components/self_edit_queue.py)
CREATE TABLE IF NOT EXISTS se_edits (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    capability TEXT NOT NULL,
    target_file TEXT NOT NULL,
    bytes_proposed INTEGER NOT NULL,
    bytes_existing INTEGER NOT NULL,
    rationale TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    decided_ts TEXT,
    decided_by TEXT,
    commit_sha TEXT
);

-- sg_autonomy_log  (recovered from components/self_gen_autonomy_tracker.py)
CREATE TABLE IF NOT EXISTS sg_autonomy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    gap_name TEXT,
    edit_id TEXT,
    meta TEXT
);

-- skill_assessments  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS skill_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT (datetime('now')),
    submission_id TEXT NOT NULL,
    work_type TEXT NOT NULL,
    scores_json TEXT,
    overall REAL,
    passed INTEGER,
    notes TEXT,
    assessor TEXT DEFAULT 'auto'
);

-- skill_graduation  (recovered from components/review/skill_assessor.py)
CREATE TABLE IF NOT EXISTS skill_graduation (work_type TEXT PRIMARY KEY, graduated INTEGER DEFAULT 0, graduated_at TEXT, graduated_by TEXT, notes TEXT);

-- sources  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS sources (
    url TEXT PRIMARY KEY,
    repo_name TEXT,
    source_type TEXT,
    processed_at TIMESTAMP,
    capabilities_found INTEGER DEFAULT 0,
    capabilities_integrated INTEGER DEFAULT 0
);

-- stage_history  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS stage_history (id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL, prev_stage TEXT, insights INTEGER, capabilities INTEGER, vocab INTEGER, avg_kpi REAL, within_pct REAL, recorded_at TEXT NOT NULL DEFAULT (datetime('now')));

-- strategy_runs  (recovered from components/wealth/strategy_lab.py)
CREATE TABLE IF NOT EXISTS strategy_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')), variant TEXT NOT NULL, trades_considered INTEGER, trades_accepted INTEGER, total_pnl_usd REAL, win_rate REAL, avg_pnl_pct REAL, score REAL, notes TEXT);

-- suggestions  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS suggestions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'user',
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    complexity TEXT DEFAULT NULL,
    plan TEXT DEFAULT NULL,
    result TEXT DEFAULT NULL,
    pr_url TEXT DEFAULT NULL,
    branch TEXT DEFAULT NULL,
    files_changed TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT DEFAULT NULL
);

-- syllabus_content  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS syllabus_content (topic TEXT PRIMARY KEY, name TEXT, stage TEXT, category TEXT, content TEXT, mastery REAL DEFAULT 0.0, topic_type TEXT DEFAULT 'general', last_trained TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')));

-- synapses  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS synapses (
    id TEXT PRIMARY KEY,
    from_insight TEXT NOT NULL,
    to_insight TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    occurrences INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(from_insight) REFERENCES insights(id) ON DELETE CASCADE,
    FOREIGN KEY(to_insight) REFERENCES insights(id) ON DELETE CASCADE
);

-- system_optimizations  (recovered from components/self_optimizer.py)
CREATE TABLE IF NOT EXISTS system_optimizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT,
    change_type TEXT,
    old_version TEXT,
    new_version TEXT,
    expected_improvement REAL,
    actual_improvement REAL,
    status TEXT,
    tested_at TIMESTAMP,
    deployed_at TIMESTAMP,
    rollback_at TIMESTAMP
);

-- system_state  (recovered from dmai_core_complete.py)
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY, value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- system_versions  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS system_versions (
    id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL,
    version_name TEXT NOT NULL,
    release_date TEXT,
    architecture TEXT,
    context_window INTEGER,
    modalities TEXT,
    key_additions TEXT,
    benchmarks TEXT,
    training_data TEXT,
    safety_changes TEXT,
    source_urls TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(system_id) REFERENCES ai_systems(id) ON DELETE CASCADE
);

-- tasks  (recovered from components/sqlite_storage.py)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    description TEXT,
    status TEXT DEFAULT 'pending',
    created TEXT,
    updated TEXT,
    user TEXT DEFAULT 'master',
    priority TEXT DEFAULT 'normal'
);

-- topics  (recovered from components/sqlite_persistence.py)
CREATE TABLE IF NOT EXISTS topics (
    name TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- trades  (recovered from components/trading/mastery_system.py)
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    symbol TEXT,
    trading_type TEXT,
    algorithm TEXT,
    action TEXT,
    quantity REAL,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_percent REAL,
    confidence REAL,
    reasoning TEXT
);

-- vocabulary  (recovered from components/knowledge/vocabulary_ingester.py)
CREATE TABLE IF NOT EXISTS vocabulary (
    id TEXT PRIMARY KEY,
    word TEXT NOT NULL UNIQUE,
    part_of_speech TEXT,
    definition TEXT NOT NULL,
    etymology TEXT,
    example TEXT,
    pronunciation TEXT,
    domain TEXT DEFAULT 'general',
    source TEXT DEFAULT 'wiktionary',
    confidence REAL DEFAULT 0.9,
    created_at TEXT NOT NULL,
    last_reviewed TEXT
);

-- weighted_knowledge  (recovered from components/knowledge_manager.py)
CREATE TABLE IF NOT EXISTS weighted_knowledge (
    id TEXT PRIMARY KEY,
    topic TEXT UNIQUE,
    normalized_topic TEXT,
    content TEXT,
    weight REAL DEFAULT 0.1,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP,
    source TEXT,
    confidence REAL DEFAULT 0.5,
    can_promote BOOLEAN DEFAULT 0,
    metadata TEXT
);

-- work_review_queue  (recovered from components/review/work_review_queue.py)
CREATE TABLE IF NOT EXISTS work_review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_uid TEXT UNIQUE,
    work_type TEXT NOT NULL,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    scores_json TEXT,
    overall_score REAL,
    passed INTEGER,
    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT,
    decided_by TEXT,
    decision_notes TEXT,
    source_component TEXT,
    persona TEXT
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_provider ON api_keys(provider);
CREATE INDEX IF NOT EXISTS idx_brain_domain ON brain_entries(domain);
CREATE INDEX IF NOT EXISTS idx_capabilities_runtime ON capabilities(runtime_mode);
CREATE INDEX IF NOT EXISTS idx_capabilities_type ON capabilities(capability_type);
CREATE INDEX IF NOT EXISTS idx_core_topic ON core_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_encyc_title ON encyclopaedia(title);
CREATE INDEX IF NOT EXISTS idx_flags_report ON integrity_flags(report_id);
CREATE INDEX IF NOT EXISTS idx_flags_resolved ON integrity_flags(resolved);
CREATE INDEX IF NOT EXISTS idx_flags_type ON integrity_flags(flag_type);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at);
CREATE INDEX IF NOT EXISTS idx_insights_entity_type ON insights(entity_type);
CREATE INDEX IF NOT EXISTS idx_insights_source_topic ON insights(source_topic);
CREATE INDEX IF NOT EXISTS idx_insights_source_url ON insights(source_url);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_timestamp ON learning_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_topic ON learning_outcomes(topic);
CREATE INDEX IF NOT EXISTS idx_mf_actions_pred ON mf_actions(prediction_id, round_num);
CREATE INDEX IF NOT EXISTS idx_mf_entities_pred ON mf_entities(prediction_id);
CREATE INDEX IF NOT EXISTS idx_mf_relations_pred ON mf_relations(prediction_id);
CREATE INDEX IF NOT EXISTS idx_mon_bills_active ON mon_bills(active, next_due);
CREATE INDEX IF NOT EXISTS idx_mon_ledger_wallet ON mon_wallet_ledger(wallet, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mon_tips_status ON mon_tips(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mon_wealth_ts ON mon_wealth_deployments(ts DESC);
CREATE INDEX IF NOT EXISTS idx_skill_work_type ON skill_assessments(work_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_runs_variant ON strategy_runs(variant, ts DESC);
CREATE INDEX IF NOT EXISTS idx_synapses_from ON synapses(from_insight);
CREATE INDEX IF NOT EXISTS idx_synapses_to ON synapses(to_insight);
CREATE INDEX IF NOT EXISTS idx_tracking_outcome ON mon_tracking_picks(outcome, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_persona ON persona_usage(persona, ts DESC);
CREATE INDEX IF NOT EXISTS idx_user_bets_status ON mon_user_bets(status, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_bets_tip ON mon_user_bets(tip_id);
CREATE INDEX IF NOT EXISTS idx_vocab_domain ON vocabulary(domain);
CREATE INDEX IF NOT EXISTS idx_vocab_word ON vocabulary(word);
CREATE INDEX IF NOT EXISTS idx_weighted_topic ON weighted_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_weighted_weight ON weighted_knowledge(weight DESC);
CREATE INDEX IF NOT EXISTS idx_wrq_status ON work_review_queue(status);
CREATE INDEX IF NOT EXISTS idx_wrq_type ON work_review_queue(work_type);
CREATE INDEX IF NOT EXISTS ix_conv_messages_session
ON conv_messages(session_id, ts);
CREATE INDEX IF NOT EXISTS ix_se_edits_status ON se_edits(status, ts);
CREATE INDEX IF NOT EXISTS ix_sg_autonomy_log_event
ON sg_autonomy_log(event, ts);
CREATE INDEX IF NOT EXISTS ix_sg_autonomy_log_ts
ON sg_autonomy_log(ts);
CREATE INDEX IF NOT EXISTS mon_alerts_cat_ts ON mon_alerts(category, ts DESC);
