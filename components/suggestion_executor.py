"""
SuggestionExecutor — DMAI's self-development engine.

Picks up pending suggestions (from David or self-generated) and autonomously:
1. Analyses what code is needed using the best available LLM
2. Writes implementation into components/self_built/<suggestion_id>/
3. Simple changes (1 file, <100 lines): commit directly to main
4. Complex changes: open a GitHub PR for David's review
5. Logs every completion as a capability + training entry + graph node
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from components.db import safe_open_kdb

logger = logging.getLogger(__name__)

DB_PATH    = Path("data/dmai_knowledge.db")
REPO_ROOT  = Path(__file__).resolve().parent.parent
SELF_BUILT = REPO_ROOT / "components" / "self_built"
TRAIN_LOG  = REPO_ROOT / "data" / "training" / "self_built_log.jsonl"

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO      = os.getenv("GITHUB_REPO", "Davemiles1978/dmai-system")


# ── LLM call (Groq → DeepSeek → Cerebras) ─────────────────────────────────
def _call_llm(prompt: str, system: str = "", max_tokens: int = 3000, temp: float = 0.2) -> Optional[str]:
    import requests as _req
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    providers = []
    if GROQ_API_KEY:
        providers.append(("https://api.groq.com/openai/v1/chat/completions",
                          GROQ_API_KEY, "llama-3.3-70b-versatile"))
    if DEEPSEEK_API_KEY:
        providers.append(("https://api.deepseek.com/chat/completions",
                          DEEPSEEK_API_KEY, "deepseek-chat"))
    if CEREBRAS_API_KEY:
        providers.append(("https://api.cerebras.ai/v1/chat/completions",
                          CEREBRAS_API_KEY, "llama3.1-70b"))

    for url, key, model in providers:
        try:
            r = _req.post(url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temp},
                timeout=90)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except Exception as e:
            logger.debug("LLM provider %s failed: %s", url, e)
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM response text."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


# ── DB helpers ─────────────────────────────────────────────────────────────
def _db():
    conn = safe_open_kdb(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_suggestion(sid: str, **kwargs):
    if not kwargs:
        return
    kwargs["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [sid]
    try:
        conn = _db()
        conn.execute(f"UPDATE suggestions SET {sets} WHERE id=?", vals)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("DB update failed for suggestion %s: %s", sid, e)


# ── Git helpers ─────────────────────────────────────────────────────────────
def _git(*args, cwd=None):
    result = subprocess.run(
        ["git"] + list(args),
        cwd=str(cwd or REPO_ROOT),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _configure_git():
    _git("config", "user.email", "milesd040@gmail.com")
    _git("config", "user.name", "DMAI")


def _open_github_pr(branch: str, title: str, body: str) -> Optional[str]:
    """Open a GitHub PR and return the PR URL, or None on failure."""
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — skipping PR creation")
        return None
    import requests as _req
    try:
        r = _req.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/pulls",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                     "Accept": "application/vnd.github+json"},
            json={"title": title, "head": branch, "base": "main", "body": body},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("html_url")
    except Exception as e:
        logger.error("PR creation failed: %s", e)
        return None


# ── Main executor ───────────────────────────────────────────────────────────
class SuggestionExecutor:

    def execute(self, suggestion_id: str):
        """Full pipeline: analyse → code → commit/PR → log."""
        try:
            conn = _db()
            row = conn.execute("SELECT * FROM suggestions WHERE id=?", (suggestion_id,)).fetchone()
            conn.close()
            if not row:
                logger.error("Suggestion %s not found", suggestion_id)
                return
            title       = row["title"]
            description = row["description"]
        except Exception as e:
            logger.error("Failed to load suggestion %s: %s", suggestion_id, e)
            return

        try:
            # ── Step 1: Analyse ────────────────────────────────────────────
            _update_suggestion(suggestion_id, status="analysing")
            plan = self._analyse(title, description, suggestion_id)
            if not plan:
                _update_suggestion(suggestion_id, status="failed",
                                   result="LLM analysis failed — no providers available or all timed out.")
                return

            complexity = plan.get("complexity", "complex")
            _update_suggestion(suggestion_id, status="coding",
                               complexity=complexity, plan=json.dumps(plan))

            # ── Step 2: Generate code ──────────────────────────────────────
            files_written = self._generate_code(plan, suggestion_id, title)
            if not files_written:
                _update_suggestion(suggestion_id, status="failed",
                                   result="Code generation produced no files.")
                return

            # ── Step 3: Commit / PR ────────────────────────────────────────
            _configure_git()
            _git("pull", "--rebase", "origin", "main")

            rel_paths = [str(p.relative_to(REPO_ROOT)) for p in files_written]

            if complexity == "simple":
                for p in rel_paths:
                    _git("add", p)
                _git("commit", "-m", f"feat(self-build): {title[:72]}")
                _git("push", "origin", "main")
                final_status = "completed"
                pr_url = None
                result_msg = (f"Committed {len(files_written)} file(s) directly to main. "
                              f"Files: {', '.join(rel_paths)}")
            else:
                branch = f"auto/suggestion-{suggestion_id[:8]}-{re.sub(r'[^a-z0-9]+', '-', title.lower())[:30]}"
                _git("checkout", "-b", branch)
                for p in rel_paths:
                    _git("add", p)
                _git("commit", "-m", f"feat(self-build): {title[:72]}")
                _git("push", "origin", branch)
                _git("checkout", "main")

                pr_body = (f"## DMAI Self-Build: {title}\n\n"
                           f"**Description:** {description}\n\n"
                           f"**Complexity:** {complexity}\n\n"
                           f"**Files generated:**\n" +
                           "\n".join(f"- `{p}`" for p in rel_paths) +
                           f"\n\n**Approach:**\n{plan.get('approach', 'See generated files.')}\n\n"
                           f"---\n*Auto-generated by DMAI SuggestionExecutor*")
                pr_url = _open_github_pr(branch, f"[DMAI Self-Build] {title}", pr_body)
                final_status = "pr_opened"
                result_msg = (f"PR opened for review. {len(files_written)} file(s) staged. "
                              f"Files: {', '.join(rel_paths)}")
                if not pr_url:
                    result_msg += " (PR creation failed — GITHUB_TOKEN may be missing)"

            _update_suggestion(suggestion_id,
                               status=final_status,
                               result=result_msg,
                               pr_url=pr_url,
                               branch=branch if complexity != "simple" else "main",
                               files_changed=json.dumps(rel_paths),
                               completed_at=_now())

            # ── Step 4: Log as capability + training ───────────────────────
            self._log_completion(suggestion_id, title, rel_paths)

        except Exception as e:
            logger.error("SuggestionExecutor.execute failed for %s: %s", suggestion_id, e)
            _update_suggestion(suggestion_id, status="failed", result=str(e))

    # ── Analysis ────────────────────────────────────────────────────────────
    def _analyse(self, title: str, description: str, sid: str) -> Optional[dict]:
        system = "You are DMAI's internal coding planner. Respond ONLY with valid JSON."
        prompt = f"""Analyse this development suggestion and produce a JSON implementation plan.

Title: {title}
Description: {description}
Suggestion ID: {sid}

Return JSON with this exact shape:
{{
  "complexity": "simple",
  "rationale": "single new utility file under 80 lines",
  "files": [
    {{"path": "components/self_built/{sid}/feature.py", "purpose": "what this file does"}}
  ],
  "approach": "step by step implementation notes",
  "dependencies": ["requests", "json"]
}}

Rules:
- complexity = "simple" if: single new file in components/self_built/, <100 lines total
- complexity = "complex" if: multiple files, new Flask routes, DB schema changes, or >100 lines
- All files MUST be under components/self_built/{sid}/ (never modify existing system files)
- Keep it minimal and production-ready"""

        text = _call_llm(prompt, system=system, max_tokens=1500, temp=0.1)
        if not text:
            return None
        plan = _extract_json(text)
        if not plan:
            # Fallback: build a basic plan
            plan = {
                "complexity": "complex",
                "rationale": "LLM returned unstructured response",
                "files": [{"path": f"components/self_built/{sid}/implementation.py",
                            "purpose": f"Implementation for: {title}"}],
                "approach": description,
                "dependencies": []
            }
        return plan

    # ── Code generation ──────────────────────────────────────────────────────
    def _generate_code(self, plan: dict, sid: str, title: str) -> list:
        files_written = []
        approach = plan.get("approach", "")
        deps = ", ".join(plan.get("dependencies", []))

        for file_spec in plan.get("files", []):
            fpath_str = file_spec.get("path", f"components/self_built/{sid}/main.py")
            purpose   = file_spec.get("purpose", title)

            # Security: force path to be inside self_built
            if "self_built" not in fpath_str:
                fpath_str = f"components/self_built/{sid}/{Path(fpath_str).name}"

            abs_path = REPO_ROOT / fpath_str
            abs_path.parent.mkdir(parents=True, exist_ok=True)

            system = "You are DMAI writing her own source code. Return ONLY raw Python code with no markdown fences."
            prompt = f"""Write complete, production-ready Python code for this file.

File: {fpath_str}
Purpose: {purpose}
Overall approach: {approach}
Required dependencies: {deps}
Context: Part of the DMAI autonomous AI system (Flask app, Python 3.11+, SQLite).

Start the file with a module docstring: \"\"\"<purpose>\"\"\".
Include type hints. Handle exceptions gracefully. No mock data.
Return ONLY the raw Python code."""

            code = _call_llm(prompt, system=system, max_tokens=2500, temp=0.15)
            if not code:
                code = f'"""{purpose}\nAuto-generated by DMAI SuggestionExecutor.\n"""\n# TODO: implement\n'

            # Strip any accidental markdown fences
            code = re.sub(r"^```(?:python)?\n?", "", code.strip())
            code = re.sub(r"\n?```$", "", code.strip())

            abs_path.write_text(code, encoding="utf-8")
            files_written.append(abs_path)
            logger.info("SuggestionExecutor: wrote %s (%d bytes)", fpath_str, len(code))

        # Always write a README.md inside the suggestion folder
        readme_path = REPO_ROOT / "components" / "self_built" / sid / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(
            f"# {title}\n\nAuto-generated by DMAI SuggestionExecutor on {_now()[:10]}.\n\n"
            f"## Approach\n{plan.get('approach', '')}\n\n"
            f"## Files\n" + "\n".join(f"- `{f['path']}`" for f in plan.get("files", [])) + "\n",
            encoding="utf-8"
        )
        files_written.append(readme_path)
        return files_written

    # ── Post-completion logging ──────────────────────────────────────────────
    def _log_completion(self, sid: str, title: str, file_paths: list):
        # Log to capabilities table
        try:
            cap_id = str(uuid.uuid4())
            conn = _db()
            conn.execute(
                "INSERT OR IGNORE INTO capabilities (id, name, type, capability_type, description, "
                "file_path, runtime_mode, language, integrated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (cap_id, title, "module", "self_built",
                 f"Self-built by DMAI suggestion system. Files: {', '.join(file_paths)}",
                 file_paths[0] if file_paths else "", "autonomous", "Python", _now())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("capability log failed: %s", e)

        # Append to training JSONL
        try:
            TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "source": "suggestion_executor",
                "suggestion_id": sid,
                "title": title,
                "files": file_paths,
                "date": _now()[:10],
                "domain": "self_improvement"
            }
            with open(TRAIN_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("training log failed: %s", e)

        # Add graph node
        try:
            from components.graph_writer import GraphWriter
            GraphWriter().add_insight_node(
                concept=title, domain="self_improvement", source="suggestion_executor"
            )
        except Exception as e:
            logger.debug("GraphWriter node failed (non-fatal): %s", e)

    # ── Self-suggestion generation ────────────────────────────────────────────
    def generate_self_suggestions(self):
        """Mine insight gaps and auto-create development suggestions."""
        try:
            conn = _db()

            # Count existing pending self-suggestions
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM suggestions WHERE source='self' AND status='pending'"
            ).fetchone()[0]
            if pending_count >= 5:
                conn.close()
                return

            # Find high-confidence insights with no matching capability name
            rows = conn.execute("""
                SELECT DISTINCT i.source_topic as concept
                FROM insights i
                WHERE i.confidence > 0.7
                  AND NOT EXISTS (
                      SELECT 1 FROM capabilities c
                      WHERE LOWER(c.name) LIKE '%' || LOWER(i.source_topic) || '%'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM suggestions s
                      WHERE LOWER(s.title) LIKE '%' || LOWER(i.source_topic) || '%'
                        AND s.source = 'self'
                  )
                LIMIT 3
            """).fetchall()
            conn.close()

            slots = max(0, 5 - pending_count)
            for row in rows[:slots]:
                concept = row["concept"]
                now = _now()
                sid = str(uuid.uuid4())
                new_conn = _db()
                new_conn.execute(
                    "INSERT INTO suggestions (id, source, title, description, status, created_at, updated_at) "
                    "VALUES (?, 'self', ?, ?, 'pending', ?, ?)",
                    (sid,
                     f"Implement capability: {concept}",
                     f"DMAI identified a knowledge gap — she has studied '{concept}' extensively "
                     f"(confidence >0.7) but has no corresponding implementation. "
                     f"Build a utility module that applies this knowledge.",
                     now, now)
                )
                new_conn.commit()
                new_conn.close()
                logger.info("SuggestionExecutor: self-suggested '%s' (id=%s)", concept, sid)

                # Fire execution in background
                import threading
                t = threading.Thread(target=self.execute, args=(sid,), daemon=True,
                                     name=f"suggestion-exec-{sid[:8]}")
                t.start()

        except Exception as e:
            logger.error("generate_self_suggestions failed: %s", e)
