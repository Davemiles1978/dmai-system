import sys

with open('components/sqlite_persistence.py', 'r') as f:
    lines = f.readlines()

# Find insertion point for CREATE TABLE statements (after SOURCES TRACKING, before INDEXES)
insert_tables_at = None
for i, line in enumerate(lines):
    if "# INDEXES for fast queries" in line:
        insert_tables_at = i
        break

if not insert_tables_at:
    print("ERROR: Could not find INDEXES insertion point")
    sys.exit(1)

genealogy_tables = [
    '',
    '        # ============================================================',
    '        # AI GENEALOGY TABLES - Track AI system versions and predictions',
    '        # ============================================================',
    "        conn.execute('''",
    '            CREATE TABLE IF NOT EXISTS ai_systems (',
    '                id TEXT PRIMARY KEY,',
    '                name TEXT NOT NULL,',
    '                organization TEXT NOT NULL,',
    '                first_release_date TEXT,',
    "                status TEXT DEFAULT 'tracked',",
    '                tracking_since TEXT DEFAULT CURRENT_TIMESTAMP,',
    '                description TEXT,',
    '                website TEXT,',
    "                category TEXT DEFAULT 'llm'",
    '            )',
    "        ''')",
    '        ',
    "        conn.execute('''",
    '            CREATE TABLE IF NOT EXISTS system_versions (',
    '                id TEXT PRIMARY KEY,',
    '                system_id TEXT NOT NULL,',
    '                version_name TEXT NOT NULL,',
    '                release_date TEXT,',
    '                architecture TEXT,',
    '                context_window INTEGER,',
    '                modalities TEXT,',
    '                key_additions TEXT,',
    '                benchmarks TEXT,',
    '                training_data TEXT,',
    '                safety_changes TEXT,',
    '                source_urls TEXT,',
    '                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,',
    '                FOREIGN KEY(system_id) REFERENCES ai_systems(id) ON DELETE CASCADE',
    '            )',
    "        ''')",
    '        ',
    "        conn.execute('''",
    '            CREATE TABLE IF NOT EXISTS genealogy_predictions (',
    '                id TEXT PRIMARY KEY,',
    '                system_id TEXT NOT NULL,',
    '                predicted_version TEXT NOT NULL,',
    '                predicted_date TEXT,',
    '                predicted_capabilities TEXT,',
    '                predicted_architecture TEXT,',
    '                confidence REAL DEFAULT 0.5,',
    "                status TEXT DEFAULT 'pending',",
    '                actual_version TEXT,',
    '                actual_date TEXT,',
    '                lead_time_days INTEGER,',
    '                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,',
    '                FOREIGN KEY(system_id) REFERENCES ai_systems(id) ON DELETE CASCADE',
    '            )',
    "        ''')",
    '        ',
    '        # Seed initial data if tables are empty',
    '        cursor = conn.cursor()',
    '        cursor.execute("SELECT COUNT(*) FROM ai_systems")',
    '        if cursor.fetchone()[0] == 0:',
    '            self._seed_genealogy_data(conn)',
    '        ',
]

for j, gl in enumerate(genealogy_tables):
    lines.insert(insert_tables_at + j, gl + '\n')

# Now find where to insert the _seed_genealogy_data method
# Insert it right after the __init__ method (after the logger.info init message)
insert_method_at = None
for i, line in enumerate(lines):
    if "SQLite persistence initialized at" in line:
        insert_method_at = i + 1
        break

if not insert_method_at:
    print("ERROR: Could not find method insertion point")
    sys.exit(1)

# We'll append the method at the end of the class definition section
# Find the last method before routes/api stuff
# Instead, let's add it right after __init__
seed_method = [
    '',
    '    def _seed_genealogy_data(self, conn):',
    '        """Seed initial AI system genealogy data with 8 major AI systems and 38 versions"""',
    '        import json',
    '        systems = [',
    '            ("openai_gpt", "GPT / ChatGPT", "OpenAI", "2018-06-11", "GPT series from GPT-1 through GPT-5, o1, o3 reasoning models", "https://openai.com", "llm"),',
    '            ("anthropic_claude", "Claude", "Anthropic", "2023-03-14", "Claude series with Constitutional AI, safety-focused design", "https://anthropic.com", "llm"),',
    '            ("google_gemini", "Gemini", "Google DeepMind", "2023-12-06", "Gemini series, natively multimodal from inception", "https://deepmind.google", "multimodal"),',
    '            ("deepseek", "DeepSeek", "DeepSeek (High-Flyer)", "2023-11-02", "Open-weight Chinese LLM with MoE architecture and strong coding/math", "https://deepseek.com", "llm"),',
    '            ("xai_grok", "Grok", "xAI (Elon Musk)", "2023-11-04", "Grok series with real-time X/Twitter access", "https://x.ai", "llm"),',
    '            ("meta_llama", "Llama", "Meta", "2023-02-24", "Open-weight Llama series, most widely adopted open model ecosystem", "https://llama.meta.com", "llm"),',
    '            ("mistral", "Mistral", "Mistral AI", "2023-09-27", "European open-weight LLMs, efficient architecture pioneer", "https://mistral.ai", "llm"),',
    '            ("cohere", "Command R", "Cohere", "2023-09-01", "Enterprise-focused LLMs with RAG optimization", "https://cohere.com", "llm"),',
    '        ]',
    '        for s in systems:',
    '            conn.execute(',
    '                "INSERT OR REPLACE INTO ai_systems (id, name, organization, first_release_date, description, website, category) "',
    '                "VALUES (?, ?, ?, ?, ?, ?, ?)", s',
    '            )',
    '        ',
    '        versions = [',
    '            ("openai_gpt_gpt_1", "openai_gpt", "GPT-1", "2018-06-11", "Transformer decoder, 117M params", 512, json.dumps(["text"]), json.dumps(["First GPT model", "Pretraining + fine-tuning paradigm"])),',
    '            ("openai_gpt_gpt_2", "openai_gpt", "GPT-2", "2019-02-14", "Transformer decoder, 1.5B params", 1024, json.dumps(["text"]), json.dumps(["Zero-shot task transfer", "Larger scale (10x)"])),',
    '            ("openai_gpt_gpt_3", "openai_gpt", "GPT-3", "2020-06-11", "Transformer decoder, 175B params", 2048, json.dumps(["text"]), json.dumps(["Few-shot learning", "In-context learning", "175B scale"])),',
    '            ("openai_gpt_gpt_3_5_chatgpt", "openai_gpt", "GPT-3.5 / ChatGPT", "2022-11-30", "GPT-3.5-turbo, RLHF fine-tuned", 4096, json.dumps(["text"]), json.dumps(["RLHF alignment", "Chat interface", "Instruction following"])),',
    '            ("openai_gpt_gpt_4", "openai_gpt", "GPT-4", "2023-03-14", "Mixture of Experts, ~1.7T params", 8192, json.dumps(["text", "image"]), json.dumps(["Multimodal input", "Improved reasoning", "Steerability"])),',
    '            ("openai_gpt_gpt_4_turbo", "openai_gpt", "GPT-4 Turbo", "2023-11-06", "Optimized GPT-4", 128000, json.dumps(["text", "image"]), json.dumps(["128K context", "JSON mode", "Function calling improvements"])),',
    '            ("openai_gpt_gpt_4o", "openai_gpt", "GPT-4o", "2024-05-13", "Omni-modal, natively multimodal", 128000, json.dumps(["text", "image", "audio"]), json.dumps(["Real-time audio", "Vision improvements", "Omni-modal"])),',
    '            ("openai_gpt_o1_preview", "openai_gpt", "o1-preview", "2024-09-12", "Chain-of-thought reasoning model", 128000, json.dumps(["text"]), json.dumps(["Test-time compute scaling", "Chain-of-thought reasoning", "STEM focus"])),',
    '            ("openai_gpt_o3", "openai_gpt", "o3", "2025-12-20", "Advanced reasoning, program search", 200000, json.dumps(["text", "image"]), json.dumps(["Program synthesis", "Deliberative alignment", "200K context"])),',
    '            ("openai_gpt_gpt_5", "openai_gpt", "GPT-5", "2025-08-05", "Unified GPT + o-series architecture", 256000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Unified model", "All modalities", "Agentic capabilities"])),',
    '            ("anthropic_claude_claude_1", "anthropic_claude", "Claude 1", "2023-03-14", "Constitutional AI, transformer-based", 9000, json.dumps(["text"]), json.dumps(["Constitutional AI", "Safety-first design"])),',
    '            ("anthropic_claude_claude_2", "anthropic_claude", "Claude 2", "2023-07-11", "Improved transformer", 100000, json.dumps(["text"]), json.dumps(["100K context", "Improved reasoning", "Code generation"])),',
    '            ("anthropic_claude_claude_3", "anthropic_claude", "Claude 3 (Haiku/Sonnet/Opus)", "2024-03-04", "Multi-scale architecture (3 sizes)", 200000, json.dumps(["text", "image"]), json.dumps(["Multimodal", "200K context", "Tool use"])),',
    '            ("anthropic_claude_claude_3_5_sonnet", "anthropic_claude", "Claude 3.5 Sonnet", "2024-06-20", "Enhanced Claude 3", 200000, json.dumps(["text", "image"]), json.dumps(["Artifacts", "Agentic computer use", "Improved coding"])),',
    '            ("anthropic_claude_claude_4_opus", "anthropic_claude", "Claude 4 Opus", "2025-05-22", "Next-gen architecture", 200000, json.dumps(["text", "image", "audio"]), json.dumps(["Audio input", "Extended thinking", "Agent SDK"])),',
    '            ("google_gemini_gemini_1_0", "google_gemini", "Gemini 1.0", "2023-12-06", "Native multimodal, MoE", 32768, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["Natively multimodal", "Three sizes", "Code generation"])),',
    '            ("google_gemini_gemini_1_5_pro", "google_gemini", "Gemini 1.5 Pro", "2024-02-15", "MoE with long-context", 1000000, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["1M context window", "Mixture of Experts", "Improved reasoning"])),',
    '            ("google_gemini_gemini_2_0_flash", "google_gemini", "Gemini 2.0 Flash", "2024-12-11", "Next-gen efficient architecture", 1000000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Agentic era", "Multimodal reasoning", "Spatial understanding"])),',
    '            ("google_gemini_gemini_2_5_pro", "google_gemini", "Gemini 2.5 Pro", "2025-03-25", "Advanced thinking model", 1000000, json.dumps(["text", "image", "audio", "video", "code"]), json.dumps(["Thinking mode", "Code execution", "Enhanced reasoning"])),',
    '            ("deepseek_deepseek_v1", "deepseek", "DeepSeek V1", "2023-11-02", "Transformer, 67B params", 4096, json.dumps(["text"]), json.dumps(["Open-weight", "Strong code generation"])),',
    '            ("deepseek_deepseek_v2", "deepseek", "DeepSeek V2", "2024-05-06", "MoE, 236B total (21B active)", 128000, json.dumps(["text"]), json.dumps(["Multi-head Latent Attention", "MoE efficiency", "128K context"])),',
    '            ("deepseek_deepseek_v3", "deepseek", "DeepSeek V3", "2024-12-26", "MoE, 671B total (37B active)", 128000, json.dumps(["text"]), json.dumps(["FP8 training", "Auxiliary-loss-free load balancing", "Multi-token prediction"])),',
    '            ("deepseek_deepseek_r1", "deepseek", "DeepSeek R1", "2025-01-20", "Reasoning model based on V3", 128000, json.dumps(["text"]), json.dumps(["Chain-of-thought reasoning", "Open-weight reasoning", "Distillation to smaller models"])),',
    '            ("xai_grok_grok_1", "xai_grok", "Grok-1", "2023-11-04", "Transformer, 314B params (MoE)", 8192, json.dumps(["text"]), json.dumps(["Real-time X access", "Open-weight release"])),',
    '            ("xai_grok_grok_2", "xai_grok", "Grok-2", "2024-08-13", "Improved MoE, on X infrastructure", 128000, json.dumps(["text", "image"]), json.dumps(["Image generation (Flux)", "Real-time web search", "X integration"])),',
    '            ("xai_grok_grok_3", "xai_grok", "Grok-3", "2025-02-17", "Massive training cluster (Colossus)", 1000000, json.dumps(["text", "image", "code"]), json.dumps(["DeepSearch", "Think mode", "1M context"])),',
    '            ("meta_llama_llama_1", "meta_llama", "Llama 1", "2023-02-24", "Transformer decoder, 7B-65B", 2048, json.dumps(["text"]), json.dumps(["First major open-weight LLM"])),',
    '            ("meta_llama_llama_2", "meta_llama", "Llama 2", "2023-07-18", "Transformer, 7B-70B, RLHF", 4096, json.dumps(["text"]), json.dumps(["Commercial license", "RLHF alignment", "Chat variants"])),',
    '            ("meta_llama_llama_3", "meta_llama", "Llama 3", "2024-04-18", "Transformer, 8B-70B", 8192, json.dumps(["text"]), json.dumps(["Improved tokenizer", "Grouped query attention"])),',
    '            ("meta_llama_llama_3_1", "meta_llama", "Llama 3.1", "2024-07-23", "Transformer, 8B-405B", 128000, json.dumps(["text"]), json.dumps(["405B open model", "Multilingual", "Tool calling"])),',
    '            ("meta_llama_llama_4", "meta_llama", "Llama 4", "2025-04-05", "MoE, multimodal native", 10000000, json.dumps(["text", "image", "audio", "video"]), json.dumps(["Natively multimodal", "10M context", "MoE", "Scout/Maverick"])),',
    '            ("mistral_mistral_7b", "mistral", "Mistral 7B", "2023-09-27", "Sliding window attention, GQA", 8192, json.dumps(["text"]), json.dumps(["Sliding window attention", "Grouped query attention"])),',
    '            ("mistral_mixtral_8x7b", "mistral", "Mixtral 8x7B", "2023-12-10", "Sparse MoE, 46.7B total (12.9B active)", 32768, json.dumps(["text"]), json.dumps(["Sparse MoE", "32K context"])),',
    '            ("mistral_mistral_large", "mistral", "Mistral Large", "2024-02-26", "Dense transformer, proprietary", 32768, json.dumps(["text"]), json.dumps(["Top-tier reasoning", "Multilingual", "Function calling"])),',
    '            ("mistral_mistral_large_2", "mistral", "Mistral Large 2", "2024-07-24", "Dense, 123B params", 128000, json.dumps(["text"]), json.dumps(["123B dense", "128K context", "Code generation"])),',
    '            ("cohere_command", "cohere", "Command", "2023-09-01", "Transformer, proprietary", 4096, json.dumps(["text"]), json.dumps(["Enterprise focus", "Summarization"])),',
    '            ("cohere_command_r", "cohere", "Command R", "2024-03-11", "Transformer optimized for RAG", 128000, json.dumps(["text"]), json.dumps(["RAG optimization", "128K context", "Tool use"])),',
    '            ("cohere_command_r_plus", "cohere", "Command R+", "2024-04-04", "Enhanced RAG architecture", 128000, json.dumps(["text"]), json.dumps(["Improved RAG", "Multi-step tool use", "10 languages"])),',
    '        ]',
    '        for v in versions:',
    '            conn.execute(',
    '                "INSERT OR REPLACE INTO system_versions (id, system_id, version_name, release_date, architecture, context_window, modalities, key_additions) "',
    '                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", v',
    '            )',
    '        conn.commit()',
    '        logger.info(f"Seeded genealogy data: {len(systems)} systems, {len(versions)} versions")',
    '',
]

for j, sm in enumerate(seed_method):
    lines.insert(insert_method_at + j, sm + '\n')

with open('components/sqlite_persistence.py', 'w') as f:
    f.writelines(lines)

print("Genealogy tables and seed method added to sqlite_persistence.py")
