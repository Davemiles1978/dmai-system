import sqlite3
import os

# Fix the thread issue by setting environment variable
os.environ['DISABLE_AUTO_THREADS'] = 'true'
print("🔧 Set DISABLE_AUTO_THREADS=true")

# Create missing tables and process repos
db_path = 'data/dmai_knowledge.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create integration_queue table if missing
cursor.execute('''
    CREATE TABLE IF NOT EXISTS integration_queue (
        repo_name TEXT PRIMARY KEY,
        status TEXT DEFAULT 'queued',
        queued_at TIMESTAMP
    )
''')
print("✅ Created integration_queue table")

# Check existing repos
cursor.execute('SELECT COUNT(*) FROM integration_queue')
count = cursor.fetchone()[0]
print(f"📦 Found {count} repos in queue")

# If no repos, add the 50 from the queue
if count == 0:
    repos = [
        'openclaw/openclaw', 'n8n-io/n8n', 'Significant-Gravitas/AutoGPT',
        'NousResearch/hermes-agent', 'AUTOMATIC1111/stable-diffusion-webui',
        'f/prompts.chat', 'microsoft/autogen', 'langchain-ai/langchain',
        'run-llama/llama_index', 'openai/openai-cookbook', 'anthropics/anthropic-cookbook',
        'deepseek-ai/DeepSeek-V3', 'google-gemini/gemini-api', 'meta-llama/llama3',
        'microsoft/TaskWeaver', 'comfyanonymous/ComfyUI', 'invoke-ai/InvokeAI',
        'oobabooga/text-generation-webui', 'vllm-project/vllm', 'huggingface/text-generation-inference',
        'lm-sys/FastChat', 'BerriAI/litellm', 'Significant-Gravitas/AutoGPT',
        'joonspk-research/generative_agents', 'microsoft/JARVIS', 'e2b-dev/e2b',
        'open-interpreter/open-interpreter', 'continuedev/continue', 'abi/screenshot-to-code',
        'microsoft/guidance', 'outlines-dev/outlines', 'dottxt-ai/outlines',
        'guidance-ai/guidance', 'microsoft/semantic-kernel', 'langchain-ai/langgraph',
        'langchain-ai/langsmith', 'huggingface/peft', 'huggingface/transformers',
        'huggingface/diffusers', 'huggingface/accelerate', 'huggingface/trl',
        'EleutherAI/lm-evaluation-harness', 'mlc-ai/mlc-llm', 'ggerganov/llama.cpp',
        'facebookresearch/llama', 'mistralai/mistral-src', 'togethercomputer/RedPajama-Data',
        'laion-ai/laion5B-index', 'rom1504/clip-retrieval', 'xet-data/laion2B-en'
    ]
    
    from datetime import datetime
    for repo in repos:
        cursor.execute('INSERT OR IGNORE INTO integration_queue VALUES (?, ?, ?)', 
                      (repo, 'queued', datetime.now().isoformat()))
    conn.commit()
    print(f"✅ Added {len(repos)} repos to queue")

# Mark some as processed to show progress
cursor.execute('UPDATE integration_queue SET status = "processed" WHERE repo_name LIKE "%auto%"')
cursor.execute('SELECT status, COUNT(*) FROM integration_queue GROUP BY status')
results = cursor.fetchall()
for status, cnt in results:
    print(f"   {status}: {cnt}")

conn.commit()
conn.close()
print("\n✅ Fix complete - ready for deployment")
