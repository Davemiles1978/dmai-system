#!/usr/bin/env python3
"""Setup AI Genealogy tables and seed initial data for Phase A."""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/dmai_knowledge.db")
conn = sqlite3.connect(str(DB_PATH))

# ============================================================
# AI SYSTEMS TABLE
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

# ============================================================
# SYSTEM VERSIONS TABLE
# ============================================================
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

# ============================================================
# PREDICTIONS TABLE (Phase C)
# ============================================================
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

conn.commit()

# ============================================================
# SEED DATA
# ============================================================
systems = [
    {
        "id": "openai_gpt", "name": "GPT / ChatGPT", "organization": "OpenAI",
        "first_release_date": "2018-06-11", "description": "GPT series from GPT-1 through GPT-5, o1, o3 reasoning models",
        "website": "https://openai.com", "category": "llm",
        "versions": [
            {"version_name": "GPT-1", "release_date": "2018-06-11", "architecture": "Transformer decoder, 117M params", "context_window": 512, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["First GPT model", "Pretraining + fine-tuning paradigm"])},
            {"version_name": "GPT-2", "release_date": "2019-02-14", "architecture": "Transformer decoder, 1.5B params", "context_window": 1024, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Zero-shot task transfer", "Larger scale (10x)"])},
            {"version_name": "GPT-3", "release_date": "2020-06-11", "architecture": "Transformer decoder, 175B params", "context_window": 2048, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Few-shot learning", "In-context learning", "175B scale"])},
            {"version_name": "GPT-3.5 / ChatGPT", "release_date": "2022-11-30", "architecture": "GPT-3.5-turbo, RLHF fine-tuned", "context_window": 4096, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["RLHF alignment", "Chat interface", "Instruction following"])},
            {"version_name": "GPT-4", "release_date": "2023-03-14", "architecture": "Mixture of Experts, ~1.7T params", "context_window": 8192, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["Multimodal input", "Improved reasoning", "Steerability"])},
            {"version_name": "GPT-4 Turbo", "release_date": "2023-11-06", "architecture": "Optimized GPT-4", "context_window": 128000, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["128K context", "JSON mode", "Function calling improvements"])},
            {"version_name": "GPT-4o", "release_date": "2024-05-13", "architecture": "Omni-modal, natively multimodal", "context_window": 128000, "modalities": json.dumps(["text", "image", "audio"]), "key_additions": json.dumps(["Real-time audio", "Vision improvements", "Omni-modal"])},
            {"version_name": "o1-preview", "release_date": "2024-09-12", "architecture": "Chain-of-thought reasoning model", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Test-time compute scaling", "Chain-of-thought reasoning", "STEM focus"])},
            {"version_name": "o3", "release_date": "2025-12-20", "architecture": "Advanced reasoning, program search", "context_window": 200000, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["Program synthesis", "Deliberative alignment", "200K context"])},
            {"version_name": "GPT-5", "release_date": "2025-08-05", "architecture": "Unified GPT + o-series architecture", "context_window": 256000, "modalities": json.dumps(["text", "image", "audio", "video"]), "key_additions": json.dumps(["Unified model", "All modalities", "Agentic capabilities"])},
        ]
    },
    {
        "id": "anthropic_claude", "name": "Claude", "organization": "Anthropic",
        "first_release_date": "2023-03-14", "description": "Claude series with Constitutional AI, safety-focused design",
        "website": "https://anthropic.com", "category": "llm",
        "versions": [
            {"version_name": "Claude 1", "release_date": "2023-03-14", "architecture": "Constitutional AI, transformer-based", "context_window": 9000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Constitutional AI", "Safety-first design"])},
            {"version_name": "Claude 2", "release_date": "2023-07-11", "architecture": "Improved transformer", "context_window": 100000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["100K context", "Improved reasoning", "Code generation"])},
            {"version_name": "Claude 3 (Haiku/Sonnet/Opus)", "release_date": "2024-03-04", "architecture": "Multi-scale architecture (3 sizes)", "context_window": 200000, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["Multimodal", "200K context", "Tool use"])},
            {"version_name": "Claude 3.5 Sonnet", "release_date": "2024-06-20", "architecture": "Enhanced Claude 3", "context_window": 200000, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["Artifacts", "Agentic computer use", "Improved coding"])},
            {"version_name": "Claude 4 Opus", "release_date": "2025-05-22", "architecture": "Next-gen architecture", "context_window": 200000, "modalities": json.dumps(["text", "image", "audio"]), "key_additions": json.dumps(["Audio input", "Extended thinking", "Agent SDK"])},
        ]
    },
    {
        "id": "google_gemini", "name": "Gemini", "organization": "Google DeepMind",
        "first_release_date": "2023-12-06", "description": "Gemini series, natively multimodal from inception",
        "website": "https://deepmind.google", "category": "multimodal",
        "versions": [
            {"version_name": "Gemini 1.0", "release_date": "2023-12-06", "architecture": "Native multimodal, MoE", "context_window": 32768, "modalities": json.dumps(["text", "image", "audio", "video", "code"]), "key_additions": json.dumps(["Natively multimodal", "Three sizes", "Code generation"])},
            {"version_name": "Gemini 1.5 Pro", "release_date": "2024-02-15", "architecture": "MoE with long-context", "context_window": 1000000, "modalities": json.dumps(["text", "image", "audio", "video", "code"]), "key_additions": json.dumps(["1M context window", "Mixture of Experts", "Improved reasoning"])},
            {"version_name": "Gemini 2.0 Flash", "release_date": "2024-12-11", "architecture": "Next-gen efficient architecture", "context_window": 1000000, "modalities": json.dumps(["text", "image", "audio", "video"]), "key_additions": json.dumps(["Agentic era", "Multimodal reasoning", "Spatial understanding"])},
            {"version_name": "Gemini 2.5 Pro", "release_date": "2025-03-25", "architecture": "Advanced thinking model", "context_window": 1000000, "modalities": json.dumps(["text", "image", "audio", "video", "code"]), "key_additions": json.dumps(["Thinking mode", "Code execution", "Enhanced reasoning"])},
        ]
    },
    {
        "id": "deepseek", "name": "DeepSeek", "organization": "DeepSeek (High-Flyer)",
        "first_release_date": "2023-11-02", "description": "Open-weight Chinese LLM with MoE architecture and strong coding/math",
        "website": "https://deepseek.com", "category": "llm",
        "versions": [
            {"version_name": "DeepSeek V1", "release_date": "2023-11-02", "architecture": "Transformer, 67B params", "context_window": 4096, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Open-weight", "Strong code generation"])},
            {"version_name": "DeepSeek V2", "release_date": "2024-05-06", "architecture": "MoE, 236B total (21B active)", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Multi-head Latent Attention", "MoE efficiency", "128K context"])},
            {"version_name": "DeepSeek V3", "release_date": "2024-12-26", "architecture": "MoE, 671B total (37B active)", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["FP8 training", "Auxiliary-loss-free load balancing", "Multi-token prediction"])},
            {"version_name": "DeepSeek R1", "release_date": "2025-01-20", "architecture": "Reasoning model based on V3", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Chain-of-thought reasoning", "Open-weight reasoning", "Distillation to smaller models"])},
        ]
    },
    {
        "id": "xai_grok", "name": "Grok", "organization": "xAI (Elon Musk)",
        "first_release_date": "2023-11-04", "description": "Grok series with real-time X/Twitter access",
        "website": "https://x.ai", "category": "llm",
        "versions": [
            {"version_name": "Grok-1", "release_date": "2023-11-04", "architecture": "Transformer, 314B params (MoE)", "context_window": 8192, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Real-time X access", "Open-weight release"])},
            {"version_name": "Grok-2", "release_date": "2024-08-13", "architecture": "Improved MoE, on X infrastructure", "context_window": 128000, "modalities": json.dumps(["text", "image"]), "key_additions": json.dumps(["Image generation (Flux)", "Real-time web search", "X integration"])},
            {"version_name": "Grok-3", "release_date": "2025-02-17", "architecture": "Massive training cluster (Colossus)", "context_window": 1000000, "modalities": json.dumps(["text", "image", "code"]), "key_additions": json.dumps(["DeepSearch", "Think mode", "1M context"])},
        ]
    },
    {
        "id": "meta_llama", "name": "Llama", "organization": "Meta",
        "first_release_date": "2023-02-24", "description": "Open-weight Llama series, most widely adopted open model ecosystem",
        "website": "https://llama.meta.com", "category": "llm",
        "versions": [
            {"version_name": "Llama 1", "release_date": "2023-02-24", "architecture": "Transformer decoder, 7B-65B", "context_window": 2048, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["First major open-weight LLM"])},
            {"version_name": "Llama 2", "release_date": "2023-07-18", "architecture": "Transformer, 7B-70B, RLHF", "context_window": 4096, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Commercial license", "RLHF alignment", "Chat variants"])},
            {"version_name": "Llama 3", "release_date": "2024-04-18", "architecture": "Transformer, 8B-70B", "context_window": 8192, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Improved tokenizer", "Grouped query attention"])},
            {"version_name": "Llama 3.1", "release_date": "2024-07-23", "architecture": "Transformer, 8B-405B", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["405B open model", "Multilingual", "Tool calling"])},
            {"version_name": "Llama 4", "release_date": "2025-04-05", "architecture": "MoE, multimodal native", "context_window": 10000000, "modalities": json.dumps(["text", "image", "audio", "video"]), "key_additions": json.dumps(["Natively multimodal", "10M context", "MoE", "Scout/Maverick"])},
        ]
    },
    {
        "id": "mistral", "name": "Mistral", "organization": "Mistral AI",
        "first_release_date": "2023-09-27", "description": "European open-weight LLMs, efficient architecture pioneer",
        "website": "https://mistral.ai", "category": "llm",
        "versions": [
            {"version_name": "Mistral 7B", "release_date": "2023-09-27", "architecture": "Sliding window attention, GQA", "context_window": 8192, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Sliding window attention", "Grouped query attention"])},
            {"version_name": "Mixtral 8x7B", "release_date": "2023-12-10", "architecture": "Sparse MoE, 46.7B total (12.9B active)", "context_window": 32768, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Sparse MoE", "32K context"])},
            {"version_name": "Mistral Large", "release_date": "2024-02-26", "architecture": "Dense transformer, proprietary", "context_window": 32768, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Top-tier reasoning", "Multilingual", "Function calling"])},
            {"version_name": "Mistral Large 2", "release_date": "2024-07-24", "architecture": "Dense, 123B params", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["123B dense", "128K context", "Code generation"])},
        ]
    },
    {
        "id": "cohere", "name": "Command R", "organization": "Cohere",
        "first_release_date": "2023-09-01", "description": "Enterprise-focused LLMs with RAG optimization",
        "website": "https://cohere.com", "category": "llm",
        "versions": [
            {"version_name": "Command", "release_date": "2023-09-01", "architecture": "Transformer, proprietary", "context_window": 4096, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Enterprise focus", "Summarization"])},
            {"version_name": "Command R", "release_date": "2024-03-11", "architecture": "Transformer optimized for RAG", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["RAG optimization", "128K context", "Tool use"])},
            {"version_name": "Command R+", "release_date": "2024-04-04", "architecture": "Enhanced RAG architecture", "context_window": 128000, "modalities": json.dumps(["text"]), "key_additions": json.dumps(["Improved RAG", "Multi-step tool use", "10 languages"])},
        ]
    },
]

for system in systems:
    versions = system.pop('versions')
    conn.execute('''
        INSERT OR REPLACE INTO ai_systems (id, name, organization, first_release_date, description, website, category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (system['id'], system['name'], system['organization'], system['first_release_date'],
          system['description'], system['website'], system['category']))
    
    for v in versions:
        version_id = f"{system['id']}_{v['version_name'].lower().replace(' ', '_').replace('/', '_').replace('.', '_')}"
        conn.execute('''
            INSERT OR REPLACE INTO system_versions 
            (id, system_id, version_name, release_date, architecture, context_window, modalities, key_additions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (version_id, system['id'], v['version_name'], v['release_date'],
              v.get('architecture', ''), v.get('context_window', 0),
              v.get('modalities', '[]'), v.get('key_additions', '[]')))

conn.commit()

count = conn.execute("SELECT COUNT(*) FROM ai_systems").fetchone()[0]
ver_count = conn.execute("SELECT COUNT(*) FROM system_versions").fetchone()[0]
print(f"✅ Genealogy tables created: {count} AI systems, {ver_count} versions tracked\n")

for row in conn.execute("SELECT name, organization FROM ai_systems ORDER BY name").fetchall():
    ver_count_sys = conn.execute("SELECT COUNT(*) FROM system_versions WHERE system_id = (SELECT id FROM ai_systems WHERE name = ?)", (row[0],)).fetchone()[0]
    print(f"   {row[0]} ({row[1]}): {ver_count_sys} versions")

conn.close()
print("\nDone. Database ready for genealogy operations.")
