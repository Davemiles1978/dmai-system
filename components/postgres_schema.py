"""PostgreSQL schema for DMAI — auto-generated from data/postgres_schema.sql"""

POSTGRES_SCHEMA_SQL = """\
-- ============================================================================
-- DMAI PostgreSQL Schema — Complete Production Schema
-- Generated 2026-08-08 — replaces all SQLite CREATE TABLE statements
-- ============================================================================

-- Column migrations for existing tables (MUST run before CREATE TABLE)
ALTER TABLE insights ADD COLUMN IF NOT EXISTS concept TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS insight_text TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 0.5;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS domain TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS source TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS source_topic TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS target_topic TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS occurrence_count INTEGER DEFAULT 1;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS last_used TIMESTAMPTZ;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS neuron_level TEXT DEFAULT 'micro';
ALTER TABLE insights ADD COLUMN IF NOT EXISTS parent_macro_id TEXT;
ALTER TABLE insights ADD COLUMN IF NOT EXISTS provenance TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS source_repo TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS file_path TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS runtime_mode TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS methods TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS is_async INTEGER DEFAULT 0;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS args TEXT;
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS integrated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS provenance TEXT;
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'stage_history') THEN
        ALTER TABLE stage_history ALTER COLUMN required_for_system TYPE INTEGER USING COALESCE(required_for_system::INTEGER, 0);
        ALTER TABLE stage_history ALTER COLUMN required_for_system SET DEFAULT 0;
    END IF;
END $$;
ALTER TABLE mon_wallets ADD COLUMN IF NOT EXISTS currency TEXT NOT NULL DEFAULT 'GBP';
ALTER TABLE mon_wallets ADD COLUMN IF NOT EXISTS updated_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE system_state ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Core system tables
CREATE TABLE IF NOT EXISTS capabilities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'function',
    capability_type TEXT NOT NULL DEFAULT 'general',
    description TEXT,
    source_url TEXT,
    source_repo TEXT,
    file_path TEXT,
    runtime_mode TEXT,
    language TEXT,
    methods TEXT,
    is_async INTEGER DEFAULT 0,
    args TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    integrated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    concept TEXT,
    insight_text TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.5,
    domain TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    source_topic TEXT,
    target_topic TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_used TIMESTAMPTZ,
    neuron_level TEXT DEFAULT 'micro',
    parent_macro_id TEXT,
    provenance TEXT
);
CREATE INDEX IF NOT EXISTS idx_insights_concept ON insights(concept);
CREATE INDEX IF NOT EXISTS idx_insights_domain ON insights(domain);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_insights_source_topic ON insights(source_topic);
CREATE INDEX IF NOT EXISTS idx_insights_provenance ON insights(provenance);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Monetisation tables
CREATE TABLE IF NOT EXISTS mon_wallets (
    name TEXT PRIMARY KEY,
    balance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'GBP',
    updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mon_wallet_ledger (
    id SERIAL PRIMARY KEY,
    wallet_name TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    category TEXT,
    description TEXT,
    ts DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS mon_income_events (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    ts DOUBLE PRECISION NOT NULL,
    split_json TEXT
);

CREATE TABLE IF NOT EXISTS mon_tips (
    id TEXT PRIMARY KEY,
    event_name TEXT,
    market TEXT,
    selection TEXT,
    bookmaker TEXT,
    decimal_odds DOUBLE PRECISION,
    model_probability DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    kelly_fraction DOUBLE PRECISION,
    recommended_stake DOUBLE PRECISION,
    currency TEXT NOT NULL DEFAULT 'GBP',
    rationale TEXT,
    prediction_id TEXT,
    status TEXT DEFAULT 'pending',
    placed_at DOUBLE PRECISION,
    settled_at DOUBLE PRECISION,
    actual_stake DOUBLE PRECISION DEFAULT 0,
    profit_loss DOUBLE PRECISION DEFAULT 0,
    notes TEXT,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mon_tips_status ON mon_tips(status, created_at DESC);

CREATE TABLE IF NOT EXISTS mon_tracking_picks (
    id TEXT PRIMARY KEY,
    event_name TEXT NOT NULL,
    market TEXT,
    selection TEXT NOT NULL,
    decimal_odds DOUBLE PRECISION,
    model_probability DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    rationale TEXT,
    prediction_id TEXT,
    outcome TEXT,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS mon_user_bets (
    id SERIAL PRIMARY KEY,
    tip_id TEXT,
    event_name TEXT,
    selection TEXT,
    odds DOUBLE PRECISION,
    stake DOUBLE PRECISION,
    currency TEXT DEFAULT 'GBP',
    status TEXT DEFAULT 'pending',
    profit_loss DOUBLE PRECISION,
    placed_at DOUBLE PRECISION,
    settled_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS mon_bills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    cadence TEXT NOT NULL DEFAULT 'monthly',
    next_due DOUBLE PRECISION,
    auto_pay INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mon_bill_payments (
    id SERIAL PRIMARY KEY,
    bill_id TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS mon_wealth_deployments (
    id TEXT PRIMARY KEY,
    total_amount DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'GBP',
    basket_name TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    status TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS mon_alerts (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    meta_json TEXT,
    delivered INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_mon_alerts_cat_ts ON mon_alerts(category, ts DESC);

-- Autonomous trader tables
CREATE TABLE IF NOT EXISTS at_state (
    id INTEGER PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'conservative',
    mode TEXT NOT NULL DEFAULT 'paper',
    last_tick_ts TEXT,
    last_tick_note TEXT,
    today_date TEXT,
    today_deployed_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    today_trades INTEGER NOT NULL DEFAULT 0,
    today_open_eq DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS at_trades (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    ev DOUBLE PRECISION,
    tier TEXT NOT NULL,
    live INTEGER NOT NULL,
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS at_ticks (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    market_open INTEGER NOT NULL,
    tier TEXT NOT NULL,
    live INTEGER NOT NULL,
    signals_seen INTEGER NOT NULL DEFAULT 0,
    signals_passed INTEGER NOT NULL DEFAULT 0,
    trades_placed INTEGER NOT NULL DEFAULT 0,
    note TEXT
);

CREATE TABLE IF NOT EXISTS at_exits (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER,
    symbol TEXT,
    exit_price DOUBLE PRECISION,
    profit_loss DOUBLE PRECISION,
    reason TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS at_pending (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty DOUBLE PRECISION,
    limit_price DOUBLE PRECISION,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS at_position_high (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    high_price DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS at_tier_changes (
    id SERIAL PRIMARY KEY,
    old_tier TEXT,
    new_tier TEXT,
    reason TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);

-- Training tables
CREATE TABLE IF NOT EXISTS training_paper_tips (
    id SERIAL PRIMARY KEY,
    event_name TEXT,
    selection TEXT,
    odds DOUBLE PRECISION,
    stake DOUBLE PRECISION,
    outcome TEXT,
    profit_loss DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_paper_trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    qty DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    profit_loss DOUBLE PRECISION,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

-- Knowledge/learning tables
CREATE TABLE IF NOT EXISTS syllabus_content (
    topic TEXT PRIMARY KEY,
    mastery DOUBLE PRECISION,
    last_trained TEXT,
    stage TEXT
);

CREATE TABLE IF NOT EXISTS vocabulary (
    id SERIAL PRIMARY KEY,
    word TEXT NOT NULL,
    part_of_speech TEXT,
    definition TEXT,
    domain TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vocabulary_word ON vocabulary(word);
CREATE INDEX IF NOT EXISTS idx_vocabulary_domain ON vocabulary(domain);

CREATE TABLE IF NOT EXISTS encyclopaedia (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    domain TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    source_type TEXT,
    description TEXT,
    domain TEXT,
    priority DOUBLE PRECISION DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    domain TEXT,
    mastery DOUBLE PRECISION DEFAULT 0
);

CREATE TABLE IF NOT EXISTS learning_queue (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    priority DOUBLE PRECISION DEFAULT 0.5,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_progress (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    stage TEXT NOT NULL,
    mastery DOUBLE PRECISION DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_outcomes (
    id SERIAL PRIMARY KEY,
    domain TEXT,
    stage TEXT,
    outcome TEXT,
    passed INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learning_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type TEXT,
    pattern_data TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS skill_assessments (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    submission_id TEXT NOT NULL,
    work_type TEXT NOT NULL,
    scores_json TEXT,
    overall DOUBLE PRECISION,
    passed INTEGER,
    notes TEXT,
    assessor TEXT DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS skill_graduation (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    from_stage TEXT,
    to_stage TEXT,
    score DOUBLE PRECISION,
    graduated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS coding_curriculum_mastery (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    language TEXT,
    mastery DOUBLE PRECISION DEFAULT 0,
    last_exercise TIMESTAMPTZ
);

-- Work review
CREATE TABLE IF NOT EXISTS work_review_queue (
    id SERIAL PRIMARY KEY,
    submission_uid TEXT UNIQUE,
    work_type TEXT NOT NULL,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    scores_json TEXT,
    overall_score DOUBLE PRECISION,
    passed INTEGER,
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    decided_at TIMESTAMPTZ,
    decided_by TEXT,
    decision_notes TEXT,
    source_component TEXT,
    persona TEXT
);
CREATE INDEX IF NOT EXISTS idx_wrq_status ON work_review_queue(status);
CREATE INDEX IF NOT EXISTS idx_wrq_type ON work_review_queue(work_type);

-- Microfish prediction engine
CREATE TABLE IF NOT EXISTS mf_predictions (
    id TEXT PRIMARY KEY,
    requirement TEXT NOT NULL,
    seed_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    verdict_json TEXT,
    created_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    completed_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS mf_entities (
    prediction_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT,
    PRIMARY KEY (prediction_id, entity_id)
);

CREATE TABLE IF NOT EXISTS mf_relations (
    prediction_id TEXT NOT NULL,
    rel_id SERIAL,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    type TEXT NOT NULL,
    attrs_json TEXT,
    PRIMARY KEY (prediction_id, rel_id)
);

CREATE TABLE IF NOT EXISTS mf_agents (
    prediction_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    persona_json TEXT NOT NULL,
    platform TEXT,
    PRIMARY KEY (prediction_id, agent_id)
);

CREATE TABLE IF NOT EXISTS mf_actions (
    prediction_id TEXT NOT NULL,
    action_id SERIAL,
    agent_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    content TEXT,
    target_id TEXT,
    round_num INTEGER NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (prediction_id, action_id)
);

-- Evolution / self-improvement
CREATE TABLE IF NOT EXISTS evolution_state (
    id SERIAL PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evolution_cycles (
    id SERIAL PRIMARY KEY,
    cycle_number INTEGER,
    gap_count INTEGER,
    fixes_applied INTEGER,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS se_edits (
    id SERIAL PRIMARY KEY,
    file_path TEXT,
    edit_type TEXT,
    description TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sg_autonomy_log (
    id SERIAL PRIMARY KEY,
    action TEXT,
    result TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS improvement_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name TEXT,
    pattern_data TEXT,
    success_rate DOUBLE PRECISION DEFAULT 0
);

CREATE TABLE IF NOT EXISTS system_optimizations (
    id SERIAL PRIMARY KEY,
    optimization_type TEXT,
    description TEXT,
    impact_score DOUBLE PRECISION,
    applied INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_versions (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL,
    changes TEXT,
    deployed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge graph
CREATE TABLE IF NOT EXISTS knowledge_graph (
    id SERIAL PRIMARY KEY,
    entity_type TEXT,
    entity_id TEXT,
    data_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS graph_neurons (
    id SERIAL PRIMARY KEY,
    label TEXT,
    neuron_type TEXT,
    data_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS graph_synapses (
    id SERIAL PRIMARY KEY,
    source_id INTEGER,
    target_id INTEGER,
    weight DOUBLE PRECISION DEFAULT 1.0,
    synapse_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS synapses (
    id SERIAL PRIMARY KEY,
    from_node TEXT,
    to_node TEXT,
    relationship TEXT,
    strength DOUBLE PRECISION DEFAULT 1.0
);

-- Brain
CREATE TABLE IF NOT EXISTS brain_entries (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    content TEXT,
    source TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS brain_load_log (
    id SERIAL PRIMARY KEY,
    source_file TEXT,
    entries_loaded INTEGER,
    loaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Personas
CREATE TABLE IF NOT EXISTS personas (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS persona_usage (
    id SERIAL PRIMARY KEY,
    persona_id INTEGER,
    action TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user_msg TEXT,
    message TEXT,
    response TEXT,
    context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conv_sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conv_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Config / migrations
CREATE TABLE IF NOT EXISTS config_kv (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS migration_log (
    id SERIAL PRIMARY KEY,
    migration_name TEXT,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- Funding
CREATE TABLE IF NOT EXISTS funding_state (
    id SERIAL PRIMARY KEY,
    revenue_avenues TEXT,
    learned_concepts TEXT,
    performance_data TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS funding_avenues (
    id SERIAL PRIMARY KEY,
    avenue_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS funding_concepts (
    concept TEXT PRIMARY KEY,
    added_at TIMESTAMPTZ DEFAULT NOW()
);

-- Experiments
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    hypothesis TEXT,
    result TEXT,
    status TEXT DEFAULT 'running',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Strategy
CREATE TABLE IF NOT EXISTS strategy_runs (
    id SERIAL PRIMARY KEY,
    strategy_name TEXT,
    parameters_json TEXT,
    result_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS optimal_strategies (
    id SERIAL PRIMARY KEY,
    strategy_type TEXT,
    config_json TEXT,
    score DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance / benchmarking
CREATE TABLE IF NOT EXISTS performance (
    id SERIAL PRIMARY KEY,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_baselines (
    id SERIAL PRIMARY KEY,
    metric TEXT NOT NULL UNIQUE,
    baseline_value DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Integrity
CREATE TABLE IF NOT EXISTS integrity_flags (
    id SERIAL PRIMARY KEY,
    check_name TEXT,
    status TEXT,
    details TEXT,
    flagged_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS integrity_reports (
    id SERIAL PRIMARY KEY,
    report_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Verification
CREATE TABLE IF NOT EXISTS verification_log (
    id SERIAL PRIMARY KEY,
    check_type TEXT,
    passed INTEGER,
    details TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Integration
CREATE TABLE IF NOT EXISTS integration_queue (
    id SERIAL PRIMARY KEY,
    source TEXT,
    payload TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS integration_registry (
    id SERIAL PRIMARY KEY,
    integration_name TEXT UNIQUE,
    config_json TEXT,
    active INTEGER DEFAULT 1
);

-- Vectors
CREATE TABLE IF NOT EXISTS vectors (
    id SERIAL PRIMARY KEY,
    entity_type TEXT,
    entity_id TEXT,
    embedding_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Weighted knowledge
CREATE TABLE IF NOT EXISTS weighted_knowledge (
    id SERIAL PRIMARY KEY,
    topic TEXT,
    weight DOUBLE PRECISION DEFAULT 1.0,
    source TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Core knowledge
CREATE TABLE IF NOT EXISTS core_knowledge (
    id SERIAL PRIMARY KEY,
    domain TEXT,
    concept TEXT,
    content TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.5
);

-- AI systems
CREATE TABLE IF NOT EXISTS ai_systems (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    capabilities_json TEXT
);

-- Insights topics
CREATE TABLE IF NOT EXISTS insight_topics (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    weight DOUBLE PRECISION DEFAULT 1.0
);

-- Suggestions
CREATE TABLE IF NOT EXISTS suggestions (
    id SERIAL PRIMARY KEY,
    suggestion_type TEXT,
    title TEXT,
    description TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deferred seeds
CREATE TABLE IF NOT EXISTS deferred_seeds (
    id SERIAL PRIMARY KEY,
    seed_type TEXT,
    payload_json TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- API keys
CREATE TABLE IF NOT EXISTS api_keys (
    key TEXT PRIMARY KEY,
    service TEXT,
    source TEXT,
    validated INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    key_hash TEXT,
    scope TEXT DEFAULT '',
    rate_limit_per_min INTEGER DEFAULT 60,
    revoked INTEGER DEFAULT 0,
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_service ON api_keys(service);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS admin_api_keys (
    id SERIAL PRIMARY KEY,
    key_hash TEXT NOT NULL,
    label TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- External API calls
CREATE TABLE IF NOT EXISTS external_api_calls (
    id BIGSERIAL PRIMARY KEY,
    key_hash TEXT NOT NULL,
    service TEXT,
    endpoint TEXT NOT NULL,
    status_code INTEGER,
    ts TIMESTAMPTZ DEFAULT NOW(),
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ext_calls_key_ts ON external_api_calls(key_hash, ts DESC);

-- Exam history (from our ExamSystem)
CREATE TABLE IF NOT EXISTS exam_history (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    stage TEXT NOT NULL,
    exam_type TEXT NOT NULL,
    overall_score DOUBLE PRECISION,
    passed INTEGER DEFAULT 0,
    pass_threshold DOUBLE PRECISION,
    total_skills INTEGER,
    passed_skills INTEGER,
    failed_skills TEXT,
    grade_summary TEXT,
    gap_analysis TEXT,
    retry_count INTEGER DEFAULT 0,
    syllabus_modified INTEGER DEFAULT 0,
    completed_at TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_exam_history_domain_stage ON exam_history(domain, stage);
CREATE INDEX IF NOT EXISTS idx_exam_history_completed ON exam_history(completed_at);

-- Treasury
CREATE TABLE IF NOT EXISTS treasury_state (
    id SERIAL PRIMARY KEY,
    balance DOUBLE PRECISION DEFAULT 0,
    currency TEXT DEFAULT 'GBP',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS treasury_ledger (
    id SERIAL PRIMARY KEY,
    amount DOUBLE PRECISION,
    category TEXT,
    description TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);

-- Trades / ledger
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    qty DOUBLE PRECISION,
    price DOUBLE PRECISION,
    total DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trades_ledger (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    profit_loss DOUBLE PRECISION,
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS bets_ledger (
    id SERIAL PRIMARY KEY,
    event_name TEXT,
    selection TEXT,
    stake DOUBLE PRECISION,
    odds DOUBLE PRECISION,
    result TEXT,
    profit_loss DOUBLE PRECISION,
    settled_at TIMESTAMPTZ
);

-- Procurement
CREATE TABLE IF NOT EXISTS procurement_state (
    id SERIAL PRIMARY KEY,
    budget DOUBLE PRECISION DEFAULT 0,
    currency TEXT DEFAULT 'GBP',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS procurement_shortlist (
    id SERIAL PRIMARY KEY,
    item_name TEXT,
    category TEXT,
    estimated_cost DOUBLE PRECISION,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS purchase_proposals (
    id SERIAL PRIMARY KEY,
    item_name TEXT,
    cost DOUBLE PRECISION,
    justification TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hardware catalog
CREATE TABLE IF NOT EXISTS hardware_catalog (
    id SERIAL PRIMARY KEY,
    component_type TEXT,
    model TEXT,
    specs_json TEXT,
    price DOUBLE PRECISION,
    url TEXT
);

-- Workload
CREATE TABLE IF NOT EXISTS workload_state (
    id SERIAL PRIMARY KEY,
    cpu_pct DOUBLE PRECISION,
    memory_mb DOUBLE PRECISION,
    disk_pct DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workload_samples (
    id SERIAL PRIMARY KEY,
    metric TEXT,
    value DOUBLE PRECISION,
    ts TIMESTAMPTZ DEFAULT NOW()
);

-- Genealogy
CREATE TABLE IF NOT EXISTS genealogy_predictions (
    id SERIAL PRIMARY KEY,
    prediction_type TEXT,
    input_data TEXT,
    result_json TEXT,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Processed repos
CREATE TABLE IF NOT EXISTS processed_repos (
    id SERIAL PRIMARY KEY,
    repo_url TEXT UNIQUE,
    status TEXT DEFAULT 'processed',
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tasks
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT DEFAULT 'pending',
    priority TEXT DEFAULT 'medium',
    data_json TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Materialisation log
CREATE TABLE IF NOT EXISTS materialisation_log (
    id SERIAL PRIMARY KEY,
    entity_type TEXT,
    entity_count INTEGER,
    ts TIMESTAMPTZ DEFAULT NOW()
);



"""
