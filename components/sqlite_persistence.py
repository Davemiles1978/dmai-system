"""
SQLite Persistence Layer for DMAI
Guarantees ALL knowledge survives deployments, crashes, and restarts.
"""

import sqlite3
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)


class SQLitePersistence:
    """
    Guaranteed persistence for DMAI knowledge.
    
    Features:
    - ACID compliant (survives crashes)
    - Single file (easy backup/migration)
    - Zero external dependencies
    - Thread-safe with connection pooling
    - Automatic recovery from corruption
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        # Honour DATA_PATH env var (matches Render's persistent-disk mount and
        # every other DB-touching component). Explicit data_dir still wins so
        # tests can point at a tmpdir.
        if data_dir is None:
            data_dir = os.environ.get("DATA_PATH", "data").rstrip("/").rstrip("\\")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "dmai_knowledge.db"
        self._lock = threading.Lock()
        self._local = threading.local()
        
        # Enable WAL mode for better concurrency and crash recovery
        self._init_db()
        
        logger.info(f"✅ SQLite persistence initialized at {self.db_path}")

    def _seed_genealogy_data(self, conn):
        """Seed initial AI system genealogy data with 8 major AI systems and 38 versions"""
        import json
        systems = [
            ("openai_gpt", "GPT / ChatGPT", "OpenAI", "2018-06-11", "GPT series from GPT-1 through GPT-5, o1, o3 reasoning models", "https://openai.com", "llm"),
            ("anthropic_claude", "Claude", "Anthropic", "2023-03-14", "Claude series with Constitutional AI, safety-focused design", "https://anthropic.com", "llm"),
            ("google_gemini", "Gemini", "Google DeepMind", "2023-12-06", "Gemini series, natively multimodal from inception", "https://deepmind.google", "multimodal"),
            ("deepseek", "DeepSeek", "DeepSeek (High-Flyer)", "2023-11-02", "Open-weight Chinese LLM with MoE architecture and strong coding/math", "https://deepseek.com", "llm"),
            ("xai_grok", "Grok", "xAI (Elon Musk)", "2023-11-04", "Grok series with real-time X/Twitter access", "https://x.ai", "llm"),
            ("meta_llama", "Llama", "Meta", "2023-02-24", "Open-weight Llama series, most widely adopted open model ecosystem", "https://llama.meta.com", "llm"),
            ("mistral", "Mistral", "Mistral AI", "2023-09-27", "European open-weight LLMs, efficient architecture pioneer", "https://mistral.ai", "llm"),
            ("cohere", "Command R", "Cohere", "2023-09-01", "Enterprise-focused LLMs with RAG optimization", "https://cohere.com", "llm"),
        ]
        for s in systems:
            conn.execute(
                "INSERT OR REPLACE INTO ai_systems (id, name, organization, first_release_date, description, website, category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", s
            )
        
        versions = [
            ("openai_gpt_gpt_1", "openai_gpt", "GPT-1", "2018-06-11", "Transformer decoder, 117M params", 512, json.dumps(["text"]), json.dumps(["First GPT model", "Pretraining + fine-tuning paradigm"])),
            ("openai_gpt_gpt_2", "openai_gpt", "GPT-2", "2019-02-14", "Transformer decoder, 1.5B params", 1024, json.dumps(["text"]), json.dumps(["Zero-shot task transfer", "Larger scale (10x)"])),
            ("openai_gpt_gpt_3", "openai_gpt", "GPT-3", "2020-06-11", "Transformer decoder, 175B params", 2048, json.dumps(["text"]), json.dumps(["Few-shot learning", "In-context learning", "175B scale"])),
            ("openai_gpt_gpt_3_5_chatgpt", "openai_gpt", "GPT-3.5 / ChatGPT", "2022-11-30", "GPT-3.5-turbo, RLHF fine-tuned", 4096, json.dumps(["text"]), json.dumps(["RLHF alignment", "Chat interface", "Instruction following"])),
            ("openai_gpt_gpt_4", "openai_gpt", "GPT-4", "2023-03-14", "Mixture of Experts, ~1.7T params", 8192, json.dumps(["text", "image"]), json.dumps(["Multimodal input", "Improved reasoning", "Steerability"])),
            ("openai_gpt_gpt_4_turbo", "openai_gpt", "GPT-4 Turbo", "2023-11-06", "Optimized GPT-4", 128000, json.dumps(["text", "image"]), json.dumps(["128K context", "JSON mode", "Function calling improvements"])),
            ("openai_gpt_gpt_4o", "openai_gpt", "GPT-4o", "2024-05-13", "Omni-modal, natively multimodal", 128000, json.dumps(["text", "image", "audio"]), json.dumps(["Real-time audio", "Vision improvements", "Omni-modal"])),
            ("openai_gpt_o1_preview", "openai_gpt", "o1-preview", "2024-09-12", "Chain-of-thought reasoning model", 128000, json.dumps(["text"]), json.dumps(["Test-time compute scaling", "Chain-of-thought reasoning", "STEM focus"])),
            ("openai_gpt_o3", "openai_gpt", "o3", "2025-12-20", "Advanced reasoning, program search", 200000, json.dumps(["text", "image"]), json.dumps(["Program synthesis", "Deliberative alignment", "200K context"])),
            ("openai_gpt_gpt_5", "openai_gpt", "GPT-5", "2025-08-05", "Unified GPT + o-series architecture", 256000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Unified model", "All modalities", "Agentic capabilities"])),
            ("anthropic_claude_claude_1", "anthropic_claude", "Claude 1", "2023-03-14", "Constitutional AI, transformer-based", 9000, json.dumps(["text"]), json.dumps(["Constitutional AI", "Safety-first design"])),
            ("anthropic_claude_claude_2", "anthropic_claude", "Claude 2", "2023-07-11", "Improved transformer", 100000, json.dumps(["text"]), json.dumps(["100K context", "Improved reasoning", "Code generation"])),
            ("anthropic_claude_claude_3", "anthropic_claude", "Claude 3 (Haiku/Sonnet/Opus)", "2024-03-04", "Multi-scale architecture (3 sizes)", 200000, json.dumps(["text", "image"]), json.dumps(["Multimodal", "200K context", "Tool use"])),
            ("anthropic_claude_claude_3_5_sonnet", "anthropic_claude", "Claude 3.5 Sonnet", "2024-06-20", "Enhanced Claude 3", 200000, json.dumps(["text", "image"]), json.dumps(["Artifacts", "Agentic computer use", "Improved coding"])),
            ("anthropic_claude_claude_4_opus", "anthropic_claude", "Claude 4 Opus", "2025-05-22", "Next-gen architecture", 200000, json.dumps(["text", "image", "audio"]), json.dumps(["Audio input", "Extended thinking", "Agent SDK"])),
            ("google_gemini_gemini_1_0", "google_gemini", "Gemini 1.0", "2023-12-06", "Native multimodal, MoE", 32768, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["Natively multimodal", "Three sizes", "Code generation"])),
            ("google_gemini_gemini_1_5_pro", "google_gemini", "Gemini 1.5 Pro", "2024-02-15", "MoE with long-context", 1000000, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["1M context window", "Mixture of Experts", "Improved reasoning"])),
            ("google_gemini_gemini_2_0_flash", "google_gemini", "Gemini 2.0 Flash", "2024-12-11", "Next-gen efficient architecture", 1000000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Agentic era", "Multimodal reasoning", "Spatial understanding"])),
            ("google_gemini_gemini_2_5_pro", "google_gemini", "Gemini 2.5 Pro", "2025-03-25", "Advanced thinking model", 1000000, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["Thinking mode", "Code execution", "Enhanced reasoning"])),
            ("deepseek_deepseek_v1", "deepseek", "DeepSeek V1", "2023-11-02", "Transformer, 67B params", 4096, json.dumps(["text"]), json.dumps(["Open-weight", "Strong code generation"])),
            ("deepseek_deepseek_v2", "deepseek", "DeepSeek V2", "2024-05-06", "MoE, 236B total (21B active)", 128000, json.dumps(["text"]), json.dumps(["Multi-head Latent Attention", "MoE efficiency", "128K context"])),
            ("deepseek_deepseek_v3", "deepseek", "DeepSeek V3", "2024-12-26", "MoE, 671B total (37B active)", 128000, json.dumps(["text"]), json.dumps(["FP8 training", "Auxiliary-loss-free load balancing", "Multi-token prediction"])),
            ("deepseek_deepseek_r1", "deepseek", "DeepSeek R1", "2025-01-20", "Reasoning model based on V3", 128000, json.dumps(["text"]), json.dumps(["Chain-of-thought reasoning", "Open-weight reasoning", "Distillation to smaller models"])),
            ("xai_grok_grok_1", "xai_grok", "Grok-1", "2023-11-04", "Transformer, 314B params (MoE)", 8192, json.dumps(["text"]), json.dumps(["Real-time X access", "Open-weight release"])),
            ("xai_grok_grok_2", "xai_grok", "Grok-2", "2024-08-13", "Improved MoE, on X infrastructure", 128000, json.dumps(["text", "image"]), json.dumps(["Image generation (Flux)", "Real-time web search", "X integration"])),
            ("xai_grok_grok_3", "xai_grok", "Grok-3", "2025-02-17", "Massive training cluster (Colossus)", 1000000, json.dumps(["text", "image", "code"]), json.dumps(["DeepSearch", "Think mode", "1M context"])),
            ("meta_llama_llama_1", "meta_llama", "Llama 1", "2023-02-24", "Transformer decoder, 7B-65B", 2048, json.dumps(["text"]), json.dumps(["First major open-weight LLM"])),
            ("meta_llama_llama_2", "meta_llama", "Llama 2", "2023-07-18", "Transformer, 7B-70B, RLHF", 4096, json.dumps(["text"]), json.dumps(["Commercial license", "RLHF alignment", "Chat variants"])),
            ("meta_llama_llama_3", "meta_llama", "Llama 3", "2024-04-18", "Transformer, 8B-70B", 8192, json.dumps(["text"]), json.dumps(["Improved tokenizer", "Grouped query attention"])),
            ("meta_llama_llama_3_1", "meta_llama", "Llama 3.1", "2024-07-23", "Transformer, 8B-405B", 128000, json.dumps(["text"]), json.dumps(["405B open model", "Multilingual", "Tool calling"])),
            ("meta_llama_llama_4", "meta_llama", "Llama 4", "2025-04-05", "MoE, multimodal native", 10000000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Natively multimodal", "10M context", "MoE", "Scout/Maverick"])),
            ("mistral_mistral_7b", "mistral", "Mistral 7B", "2023-09-27", "Sliding window attention, GQA", 8192, json.dumps(["text"]), json.dumps(["Sliding window attention", "Grouped query attention"])),
            ("mistral_mixtral_8x7b", "mistral", "Mixtral 8x7B", "2023-12-10", "Sparse MoE, 46.7B total (12.9B active)", 32768, json.dumps(["text"]), json.dumps(["Sparse MoE", "32K context"])),
            ("mistral_mistral_large", "mistral", "Mistral Large", "2024-02-26", "Dense transformer, proprietary", 32768, json.dumps(["text"]), json.dumps(["Top-tier reasoning", "Multilingual", "Function calling"])),
            ("mistral_mistral_large_2", "mistral", "Mistral Large 2", "2024-07-24", "Dense, 123B params", 128000, json.dumps(["text"]), json.dumps(["123B dense", "128K context", "Code generation"])),
            ("cohere_command", "cohere", "Command", "2023-09-01", "Transformer, proprietary", 4096, json.dumps(["text"]), json.dumps(["Enterprise focus", "Summarization"])),
            ("cohere_command_r", "cohere", "Command R", "2024-03-11", "Transformer optimized for RAG", 128000, json.dumps(["text"]), json.dumps(["RAG optimization", "128K context", "Tool use"])),
            ("cohere_command_r_plus", "cohere", "Command R+", "2024-04-04", "Enhanced RAG architecture", 128000, json.dumps(["text"]), json.dumps(["Improved RAG", "Multi-step tool use", "10 languages"])),
        ]
        for v in versions:
            conn.execute(
                "INSERT OR REPLACE INTO system_versions (id, system_id, version_name, release_date, architecture, context_window, modalities, key_additions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", v
            )
        conn.commit()
        logger.info(f"Seeded genealogy data: {len(systems)} systems, {len(versions)} versions")

    
    def _get_connection(self) -> sqlite3.Connection:
        """Thread-safe connection with WAL mode"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = safe_open_kdb(str(self.db_path), timeout=30.0)
        return self._local.conn
    
    def _init_db(self):
        """Create all tables if they don't exist"""
        conn = safe_open_kdb(str(self.db_path))
        
        # ============================================================
        # INSIGHTS TABLE (matches InsightNeuron exactly)
        # ============================================================
        conn.execute('''
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
            )
        ''')
        
        # ============================================================
        # SYNAPSES TABLE (edges between insights)
        # ============================================================
        conn.execute('''
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
            )
        ''')
        
        # ============================================================
        # TOPICS TABLE
        # ============================================================
        conn.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS insight_topics (
                insight_id TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                PRIMARY KEY(insight_id, topic_name),
                FOREIGN KEY(insight_id) REFERENCES insights(id) ON DELETE CASCADE,
                FOREIGN KEY(topic_name) REFERENCES topics(name) ON DELETE CASCADE
            )
        ''')
        
        # ============================================================
        # CAPABILITIES TABLE (from CapabilityIntegrator registry)
        # ============================================================
        conn.execute('''
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
            )
        ''')
        
        # ============================================================
        # EVOLUTION TRACKING
        # ============================================================
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evolution_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_number INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                insights_created INTEGER DEFAULT 0,
                synapses_created INTEGER DEFAULT 0,
                consciousness_level REAL
            )
        ''')
        
        # ============================================================
        # SOURCES TRACKING
        # ============================================================
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                url TEXT PRIMARY KEY,
                repo_name TEXT,
                source_type TEXT,
                processed_at TIMESTAMP,
                capabilities_found INTEGER DEFAULT 0,
                capabilities_integrated INTEGER DEFAULT 0
            )
        ''')
        
        # ============================================================

        # ============================================================
        # AI GENEALOGY TABLES - Track AI system versions and predictions
        # ============================================================
        conn.execute('''
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
            )
        ''')
        
        conn.execute('''
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
            )
        ''')
        
        conn.execute('''
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
            )
        ''')
        
        # Seed initial data if tables are empty
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ai_systems")
        if cursor.fetchone()[0] == 0:
            self._seed_genealogy_data(conn)
        
        # INDEXES for fast queries
        # ============================================================
        conn.execute('CREATE INDEX IF NOT EXISTS idx_insights_entity_type ON insights(entity_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_insights_source_url ON insights(source_url)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_insights_source_topic ON insights(source_topic)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_synapses_from ON synapses(from_insight)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_synapses_to ON synapses(to_insight)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_capabilities_type ON capabilities(capability_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_capabilities_runtime ON capabilities(runtime_mode)')
        
        conn.commit()
        conn.close()
        
        logger.info("📊 SQLite schema initialized")
    
    # ============================================================
    # INSIGHT METHODS (match InsightNeuron exactly)
    # ============================================================
    
    def save_insight(self, insight: Any) -> bool:
        """Save an InsightNeuron to SQLite"""
        try:
            conn = self._get_connection()
            with self._lock:
                conn.execute('''
                    INSERT OR REPLACE INTO insights 
                    (id, insight_text, entity_type, entities, relationship, confidence,
                     source_topic, target_topic, source_url, source_title, source_type,
                     created_at, occurrence_count, last_used, neuron_level, parent_macro_id,
                     cluster_id, is_visible_at_top_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    insight.id,
                    insight.insight_text,
                    insight.entity_type,
                    json.dumps(insight.entities),
                    insight.relationship,
                    insight.confidence,
                    insight.source_topic,
                    insight.target_topic,
                    insight.source_url,
                    insight.source_title,
                    insight.source_type,
                    insight.created_at,
                    insight.occurrence_count,
                    insight.last_used,
                    getattr(insight, 'neuron_level', 'micro'),
                    getattr(insight, 'parent_macro_id', None),
                    getattr(insight, 'cluster_id', None),
                    1 if getattr(insight, 'is_visible_at_top_level', False) else 0
                ))
                
                # Save topics
                for topic in [insight.source_topic, insight.target_topic]:
                    if topic:
                        conn.execute('INSERT OR IGNORE INTO topics (name) VALUES (?)', (topic,))
                        conn.execute('INSERT OR IGNORE INTO insight_topics (insight_id, topic_name) VALUES (?, ?)',
                                    (insight.id, topic))
                
                conn.commit()
            logger.debug(f"💾 Saved insight to SQLite: {insight.id[:16]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to save insight to SQLite: {e}")
            return False
    
    def load_all_insights(self) -> Dict[str, Any]:
        """Load all insights from SQLite as InsightNeuron objects"""
        from dmai_core_complete import InsightNeuron
        
        insights = {}
        try:
            conn = self._get_connection()
            cursor = conn.execute('SELECT * FROM insights')
            
            for row in cursor:
                data = {
                    'id': row[0],
                    'insight_text': row[1],
                    'entity_type': row[2],
                    'entities': json.loads(row[3]),
                    'relationship': row[4],
                    'confidence': row[5],
                    'source_topic': row[6],
                    'target_topic': row[7],
                    'source_url': row[8],
                    'source_title': row[9],
                    'source_type': row[10],
                    'created_at': row[11],
                    'occurrence_count': row[12],
                    'last_used': row[13],
                    'neuron_level': row[14] if len(row) > 14 else 'micro',
                    'parent_macro_id': row[15] if len(row) > 15 else None,
                    'cluster_id': row[16] if len(row) > 16 else None,
                    'is_visible_at_top_level': bool(row[17]) if len(row) > 17 else False
                }
                
                neuron = InsightNeuron.from_dict(data)
                insights[neuron.id] = neuron
            
            logger.info(f"📂 Loaded {len(insights)} insights from SQLite")
        except Exception as e:
            logger.error(f"Failed to load insights from SQLite: {e}")
        
        return insights
    
    def load_all_synapses(self) -> List[Dict]:
        """Load all synapses from SQLite"""
        synapses = []
        try:
            conn = self._get_connection()
            cursor = conn.execute('SELECT id, from_insight, to_insight, relationship, weight, occurrences FROM synapses')
            
            for row in cursor:
                synapses.append({
                    'id': row[0],
                    'from': row[1],
                    'to': row[2],
                    'relationship': row[3],
                    'weight': row[4],
                    'occurrences': row[5]
                })
        except Exception as e:
            logger.error(f"Failed to load synapses: {e}")
        
        return synapses
    
    def load_all_topics(self) -> Dict[str, List[str]]:
        """Load topics and their insight IDs"""
        topics = {}
        try:
            conn = self._get_connection()
            cursor = conn.execute('''
                SELECT t.name, it.insight_id 
                FROM topics t 
                JOIN insight_topics it ON t.name = it.topic_name
            ''')
            
            for row in cursor:
                topic_name = row[0]
                insight_id = row[1]
                if topic_name not in topics:
                    topics[topic_name] = []
                topics[topic_name].append(insight_id)
        except Exception as e:
            logger.error(f"Failed to load topics: {e}")
        
        return topics
    
    def save_synapse(self, synapse: Dict) -> bool:
        """Save a synapse to SQLite"""
        try:
            conn = self._get_connection()
            with self._lock:
                conn.execute('''
                    INSERT OR REPLACE INTO synapses 
                    (id, from_insight, to_insight, relationship, weight, occurrences)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    synapse['id'],
                    synapse['from'],
                    synapse['to'],
                    synapse.get('relationship', 'related_to'),
                    synapse.get('weight', 1.0),
                    synapse.get('occurrences', 1)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save synapse: {e}")
            return False
    
    # ============================================================
    # CAPABILITY METHODS
    # ============================================================
    
    def save_capability(self, cap_id: str, cap_data: Dict) -> bool:
        """Save a capability to SQLite"""
        try:
            conn = self._get_connection()
            with self._lock:
                conn.execute('''
                    INSERT OR REPLACE INTO capabilities 
                    (id, name, type, capability_type, description, source_url, source_repo,
                     file_path, runtime_mode, language, methods, is_async, args, integrated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cap_id,
                    cap_data.get('name'),
                    cap_data.get('type'),
                    cap_data.get('capability_type'),
                    cap_data.get('description'),
                    cap_data.get('source_url'),
                    cap_data.get('source_repo'),
                    cap_data.get('file_path'),
                    cap_data.get('runtime_mode'),
                    cap_data.get('language'),
                    json.dumps(cap_data.get('methods', [])),
                    1 if cap_data.get('is_async') else 0,
                    json.dumps(cap_data.get('args', [])),
                    cap_data.get('integrated_at', datetime.now().isoformat())
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save capability: {e}")
            return False
    
    def load_all_capabilities(self) -> Dict[str, Dict]:
        """Load all capabilities from SQLite"""
        capabilities = {}
        try:
            conn = self._get_connection()
            cursor = conn.execute('SELECT * FROM capabilities')
            columns = [description[0] for description in cursor.description]
            
            for row in cursor:
                cap = dict(zip(columns, row))
                if cap.get('methods'):
                    cap['methods'] = json.loads(cap['methods'])
                if cap.get('args'):
                    cap['args'] = json.loads(cap['args'])
                cap['is_async'] = bool(cap.get('is_async', 0))
                capabilities[cap['id']] = cap
            
            logger.info(f"📂 Loaded {len(capabilities)} capabilities from SQLite")
        except Exception as e:
            logger.error(f"Failed to load capabilities: {e}")
        
        return capabilities
    
    def save_source(self, url: str, data: Dict) -> bool:
        """Save a processed source"""
        try:
            conn = self._get_connection()
            with self._lock:
                conn.execute('''
                    INSERT OR REPLACE INTO sources 
                    (url, repo_name, source_type, processed_at, capabilities_found, capabilities_integrated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    url,
                    data.get('repo_name'),
                    data.get('source_type'),
                    data.get('processed_at'),
                    data.get('capabilities_found', 0),
                    data.get('capabilities_integrated', 0)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save source: {e}")
            return False
    
    def save_evolution_cycle(self, cycle_data: Dict) -> bool:
        """Record an evolution cycle"""
        try:
            conn = self._get_connection()
            with self._lock:
                conn.execute('''
                    INSERT INTO evolution_cycles 
                    (cycle_number, completed_at, insights_created, synapses_created, consciousness_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    cycle_data.get('cycle_number'),
                    datetime.now().isoformat(),
                    cycle_data.get('insights_created', 0),
                    cycle_data.get('synapses_created', 0),
                    cycle_data.get('consciousness_level', 0.0)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save evolution cycle: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            conn = self._get_connection()
            
            insights = conn.execute('SELECT COUNT(*) FROM insights').fetchone()[0]
            capabilities = conn.execute('SELECT COUNT(*) FROM capabilities').fetchone()[0]
            synapses = conn.execute('SELECT COUNT(*) FROM synapses').fetchone()[0]
            sources = conn.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
            
            # Autonomous vs on-demand
            autonomous = conn.execute(
                "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'autonomous'"
            ).fetchone()[0]
            ondemand = conn.execute(
                "SELECT COUNT(*) FROM capabilities WHERE runtime_mode = 'ondemand'"
            ).fetchone()[0]
            
            # By capability type
            cap_by_type = {}
            cursor = conn.execute(
                'SELECT capability_type, COUNT(*) FROM capabilities GROUP BY capability_type'
            )
            for row in cursor:
                cap_by_type[row[0]] = row[1]
            
            return {
                'insights': insights,
                'capabilities': capabilities,
                'synapses': synapses,
                'sources_processed': sources,
                'autonomous_count': autonomous,
                'ondemand_count': ondemand,
                'capabilities_by_type': cap_by_type,
                'total_nodes': insights + capabilities
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def vacuum(self):
        """Optimize database"""
        try:
            conn = self._get_connection()
            conn.execute('VACUUM')
            logger.info("🗜️ SQLite database optimized")
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")
    
    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database"""
        import shutil
        
        if backup_path is None:
            backup_dir = self.data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"dmai_knowledge_{timestamp}.db"
        
        try:
            # Use SQLite's backup API for safe copy
            source = safe_open_kdb(str(self.db_path))
            dest = sqlite3.connect(str(backup_path))
            source.backup(dest)
            source.close()
            dest.close()
            
            logger.info(f"💾 Database backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Fallback to file copy
            shutil.copy2(self.db_path, backup_path)
            return backup_path
    
    def close(self):
        """Close all connections"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn
