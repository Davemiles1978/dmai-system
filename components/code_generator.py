"""
CodeGenerator — uses available LLM providers to generate working Flask/Python code
for each gap item identified by SelfScanner.
"""
import os, json, ast, logging, time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PROVIDER_ORDER = ["groq", "google_ai_studio", "openai", "deepseek"]

CODE_GENERATION_PROMPT = """You are writing production Python code for DMAI, an autonomous AI system.

System context:
- Framework: Flask (NOT FastAPI)
- Database: SQLite at data/dmai_knowledge.db
- Python: 3.11
- All paths: use DATA_PATH env var (default "data"), always .rstrip("/") before os.path.join
- Auth: Basic auth with admin:dmai_master
- Identity: internal=DMAI, public persona=Alex Riviera

Task: Implement the following as a complete, production-ready Python module:

CAPABILITY: {capability_name}
DESCRIPTION: {description}
FILE: {target_file}

Requirements:
1. Complete working implementation — no stubs, no NotImplementedError, no TODO
2. All imports at top
3. Full error handling with try/except
4. If Flask route: include @app.route, auth check, return jsonify(...)
5. If standalone component: class with start() and run_forever() methods
6. SQLite: parameterised queries only
7. logger = logging.getLogger(__name__)

Write ONLY the Python code. No explanations. Start with imports."""


class CodeGenerator:
    def __init__(self, data_path="data"):
        self.data_path = data_path.rstrip("/")
        self.log_path = os.path.join(self.data_path, "self_generation_log.jsonl")

    def generate(self, gap_item: dict) -> str | None:
        capability_name = gap_item.get("name", gap_item.get("path", "unknown"))
        description = gap_item.get("description", str(gap_item))
        target_file = gap_item.get("component") or f"components/{capability_name.replace('/', '_').replace(' ', '_')}.py"

        prompt = CODE_GENERATION_PROMPT.format(
            capability_name=capability_name,
            description=description,
            target_file=target_file
        )

        for provider in PROVIDER_ORDER:
            code = self._call_provider(provider, prompt)
            if code:
                clean = self._extract_code(code)
                if self._validate(clean):
                    self._log(capability_name, provider, target_file, "success")
                    return clean
                logger.warning(f"CodeGenerator: {provider} produced invalid code for {capability_name}")
            time.sleep(1)

        self._log(capability_name, "all_failed", target_file, "failure")
        return None

    def _call_provider(self, provider: str, prompt: str) -> str | None:
        try:
            import requests
            if provider == "groq":
                key = os.environ.get("GROQ_API_KEY")
                if not key:
                    return None
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

            elif provider == "google_ai_studio":
                key = os.environ.get("GOOGLE_AI_STUDIO_KEY")
                if not key:
                    return None
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

            elif provider == "openai":
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    return None
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

            elif provider == "deepseek":
                key = os.environ.get("DEEPSEEK_API_KEY")
                if not key:
                    return None
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"CodeGenerator._call_provider({provider}): {e}")
        return None

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return response.strip()

    def _validate(self, code: str) -> bool:
        if not code or len(code) < 50:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _log(self, capability: str, provider: str, target_file: str, status: str):
        os.makedirs(self.data_path, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "capability": capability,
            "provider": provider,
            "target_file": target_file,
            "status": status
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
