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
    """Use DMAI's main chat pipeline which has full provider fallback chain."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        # Build message in DMAI's format
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        from dmai_core_complete import _ai_chat
        response = _ai_chat(full_prompt)
        if response and not response.startswith("I've checked my memory"):
            return response
    except Exception as e:
        logger.warning("_ai_chat fallback failed: %s", e)
    
    # Final fallback: direct providers
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    import requests as _req
    import os as _os
    providers = []
    if _os.environ.get("GROQ_API_KEY"):
        providers.append(("https://api.groq.com/openai/v1/chat/completions",
                          _os.environ["GROQ_API_KEY"], "llama-3.3-70b-versatile"))
    if _os.environ.get("DEEPSEEK_API_KEY"):
        providers.append(("https://api.deepseek.com/chat/completions",
                          _os.environ["DEEPSEEK_API_KEY"], "deepseek-chat"))
    if _os.environ.get("CEREBRAS_API_KEY"):
        providers.append(("https://api.cerebras.ai/v1/chat/completions",
                          _os.environ["CEREBRAS_API_KEY"], "llama3.1-70b"))
    if _os.environ.get("OPENAI_API_KEY"):
        providers.append(("https://api.openai.com/v1/chat/completions",
                          _os.environ["OPENAI_API_KEY"], "gpt-4o-mini"))
    if _os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("https://api.anthropic.com/v1/messages",
                          _os.environ["ANTHROPIC_API_KEY"], "claude-3-5-sonnet-20241022"))

    for url, key, model in providers:
        try:
            if "anthropic" in url:
                r = _req.post(url,
                    headers={"x-api-key": key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"},
                    json={"model": model, "max_tokens": max_tokens, "messages": messages},
                    timeout=90)
            else:
                r = _req.post(url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp},
                    timeout=90)
            r.raise_for_status()
            if "anthropic" in url:
                text = r.json()["content"][0]["text"].strip()
            else:
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
    """Get suggestions DB connection (PG primary, SQLite fallback)."""
    import os as _os
    db_url = _os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg2 as _pg
            import psycopg2.extras as _pg_extras
            if db_url.startswith("postgres://"):
                db_url = "postgresql://" + db_url[len("postgres://"):]
            conn = _pg.connect(db_url)
            conn.autocommit = True
            conn.cursor_factory = _pg_extras.RealDictCursor
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return conn
        except Exception as _e:
            logger.warning("_db: PG failed, fallback SQLite: %s", _e)
    conn = safe_open_kdb(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _db_execute(conn, sql, params=()):
    """Execute on PG or SQLite, translating ? to %s for PG."""
    if hasattr(conn, 'cursor_factory'):  # psycopg2 connection
        pg_sql = sql.replace("?", "%s")
        cur = conn.cursor()
        cur.execute(pg_sql, params)
        return cur
    return conn.execute(sql, params)


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
        _db_execute(conn, f"UPDATE suggestions SET {sets} WHERE id=?", vals)
        try:
            conn.commit()
        except Exception:
            pass
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

    def execute(self, suggestion_id: str, attempt: int = 0, previous_failures: list = None):
        """Recursive self-healing pipeline: analyse → code → commit/PR → log.

        On failure, logs what went wrong, designs an alternative approach, and
        retries with exponential backoff. Never gives up — every failure is a
        teaching opportunity committed to persistent memory.
        """
        if previous_failures is None:
            previous_failures = []

        # ── Load suggestion ────────────────────────────────────────────
        try:
            conn = _db()
            row = _db_execute(conn, "SELECT * FROM suggestions WHERE id=?", (suggestion_id,)).fetchone()
            conn.close()
            if not row:
                logger.error("Suggestion %s not found", suggestion_id)
                return
            title       = row["title"]
            description = row["description"]
        except Exception as e:
            logger.error("Failed to load suggestion %s: %s", suggestion_id, e)
            return

        # ── Exponential backoff ────────────────────────────────────────
        if attempt > 0:
            delay = min(2 ** (attempt - 1), 300)  # 1s, 2s, 4s, 8s... max 5min
            logger.info("Suggestion %s retry attempt %d — waiting %ds", suggestion_id, attempt, delay)
            time.sleep(delay)

        try:
            # ── Step 1: Analyse (with failure memory) ─────────────────
            status_label = f"analysing (attempt {attempt+1})" if attempt > 0 else "analysing"
            _update_suggestion(suggestion_id, status=status_label)

            plan = self._analyse(title, description, suggestion_id, previous_failures)
            if not plan:
                failure = {
                    "attempt": attempt + 1,
                    "stage": "analysis",
                    "error": "LLM analysis failed — no providers available or all timed out.",
                    "timestamp": _now()
                }
                previous_failures.append(failure)
                _update_suggestion(suggestion_id,
                                   status="retrying",
                                   result=json.dumps(previous_failures))
                self._log_failure_to_kaizen(suggestion_id, title, failure)
                self.execute(suggestion_id, attempt + 1, previous_failures)
                return

            complexity = plan.get("complexity", "complex")
            _update_suggestion(suggestion_id, status="coding",
                               complexity=complexity, plan=json.dumps(plan))

            # ── Step 2: Generate code ─────────────────────────────────
            files_written = self._generate_code(plan, suggestion_id, title)
            if not files_written:
                failure = {
                    "attempt": attempt + 1,
                    "stage": "code_generation",
                    "error": "Code generation produced no files.",
                    "plan_used": plan,
                    "timestamp": _now()
                }
                previous_failures.append(failure)
                _update_suggestion(suggestion_id,
                                   status="retrying",
                                   result=json.dumps(previous_failures))
                self._log_failure_to_kaizen(suggestion_id, title, failure)
                self.execute(suggestion_id, attempt + 1, previous_failures)
                return

            # ── Step 3: Commit / PR ───────────────────────────────────
            try:
                _configure_git()
                _git("pull", "--rebase", "origin", "main")
            except Exception as git_e:
                # Git failures are often environment issues — stash and retry
                try:
                    _git("stash")
                    _git("pull", "--rebase", "origin", "main")
                    _git("stash", "pop")
                except Exception:
                    pass  # Continue anyway — might still be able to commit

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

            # ── Success! ──────────────────────────────────────────────
            final_result = {
                "attempts": attempt + 1,
                "previous_failures": previous_failures,
                "final_approach": plan,
                "files": rel_paths,
                "completed_at": _now()
            }
            _update_suggestion(suggestion_id,
                               status=final_status,
                               result=json.dumps(final_result),
                               pr_url=pr_url,
                               branch=branch if complexity != "simple" else "main",
                               files_changed=json.dumps(rel_paths),
                               completed_at=_now())

            self._log_completion(suggestion_id, title, rel_paths)
            logger.info("Suggestion %s COMPLETED after %d attempt(s)", suggestion_id, attempt + 1)

        except Exception as e:
            failure = {
                "attempt": attempt + 1,
                "stage": "execution",
                "error": str(e)[:500],
                "timestamp": _now()
            }
            previous_failures.append(failure)
            logger.error("Suggestion %s attempt %d failed: %s", suggestion_id, attempt + 1, e)

            try:
                _update_suggestion(suggestion_id,
                                   status="retrying",
                                   result=json.dumps(previous_failures))
            except Exception:
                pass

            self._log_failure_to_kaizen(suggestion_id, title, failure)
            self.execute(suggestion_id, attempt + 1, previous_failures)

    def _log_failure_to_kaizen(self, sid: str, title: str, failure: dict):
        """Log a failed attempt to Kaizen so DMAI learns across suggestions."""
        try:
            kaizen_entry = {
                "source": f"suggestion:{sid}",
                "title": f"Failed attempt {failure['attempt']} for: {title}",
                "stage": failure["stage"],
                "error": failure["error"],
                "timestamp": failure["timestamp"],
                "category": "self_build"
            }
            kaizen_file = REPO_ROOT / "data" / "kaizen_failures.jsonl"
            kaizen_file.parent.mkdir(parents=True, exist_ok=True)
            with open(kaizen_file, "a") as f:
                f.write(json.dumps(kaizen_entry) + "\n")
        except Exception as e:
            logger.warning("Failed to log to Kaizen: %s", e)

    # ── Analysis ────────────────────────────────────────────────────────────
    def _analyse(self, title: str, description: str, sid: str, previous_failures: list = None) -> Optional[dict]:
        system = "You are DMAI's internal coding planner. Respond ONLY with valid JSON."
        # Build failure context for the LLM
        failure_context = ""
        if previous_failures:
            failure_context = "\n\nPREVIOUS FAILED ATTEMPTS (learn from these — do NOT repeat):\n"
            for f in previous_failures:
                failure_context += f"- Attempt {f['attempt']}: [{f['stage']}] {f['error'][:200]}\n"
            failure_context += "\nDesign a DIFFERENT approach than those above.\n"

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
        """Mine insight gaps and auto-create development suggestions.

        PR V-fast follow-up: the previous implementation ran an O(N·M)
        cross-product SELECT (~25k insights × ~20k capabilities ×
        LIKE-substring), which held the connection busy for 30s+ and
        starved the write mutex. Rewrite: pull small candidate sets
        into Python and do set-difference in-process.
        """
        try:
            conn = _db()

            # Count existing pending self-suggestions
            pending_count = conn.execute(
                "SELECT COUNT(*) FROM suggestions WHERE source='self' AND status='pending'"
            ).fetchone()[0]
            if pending_count >= 5:
                conn.close()
                return

            # Distinct high-confidence insight topics (top-50 by recency /
            # ordering — bounded so the scan is cheap).
            insight_rows = conn.execute("""
                SELECT DISTINCT source_topic
                FROM insights
                WHERE confidence > 0.7
                  AND source_topic IS NOT NULL
                  AND LENGTH(source_topic) > 2
                LIMIT 50
            """).fetchall()
            insight_topics = [(r["source_topic"] or "").strip() for r in insight_rows]
            insight_topics = [t for t in insight_topics if t]

            # Existing capability names (lowercased). Bounded LIMIT to keep
            # memory + wire modest — 20k rows is fine, 200k is not.
            cap_rows = conn.execute(
                "SELECT name FROM capabilities LIMIT 50000"
            ).fetchall()
            cap_names_lc = {(r["name"] or "").lower() for r in cap_rows}

            # Existing self-suggestion titles (lowercased).
            sug_rows = conn.execute(
                "SELECT title FROM suggestions WHERE source='self' LIMIT 10000"
            ).fetchall()
            sug_titles_lc = {(r["title"] or "").lower() for r in sug_rows}
            conn.close()

            # In-process filter: keep topics whose lowercase form isn't a
            # substring of any capability name and isn't already suggested.
            def _already_covered(topic: str) -> bool:
                tlc = topic.lower()
                if any(tlc in n for n in cap_names_lc):
                    return True
                if any(tlc in t for t in sug_titles_lc):
                    return True
                return False

            rows = [
                {"concept": t} for t in insight_topics if not _already_covered(t)
            ][:3]

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
