"""
components/code_writer.py
──────────────────────────────────────────────────────────────────────────────
DMAI CodeWriter — self-generation capability.

Allows DMAI to:
  1. Write new Python components to disk
  2. Patch existing files (add methods / fix bugs)
  3. Validate syntax before saving
  4. Self-complete outstanding Kaizen/TODO items that are code tasks
  5. Log all generated code to data/code_writer/history.jsonl

This is the foundation of DMAI's self-generation stage.

Safety rules:
  - Never overwrites files without a backup
  - Never executes generated code (write-only)
  - All writes are logged with origin (kaizen/self_generated/instructor)
  - File paths are sandboxed to the repo root
"""

import ast
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.code_writer")

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_HISTORY_FILE = _REPO_ROOT / "data" / "code_writer" / "history.jsonl"
_BACKUP_DIR   = _REPO_ROOT / "data" / "code_writer" / "backups"


class CodeWriter:
    """
    DMAI's self-generation engine.
    Writes, patches, and validates Python code on behalf of DMAI.
    """

    def __init__(self, ai_hub=None, si_core=None):
        self.ai_hub  = ai_hub
        self.si_core = si_core
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("CodeWriter initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def write_new_file(
        self,
        file_path: str,
        content: str,
        origin: str = "self_generated",
        description: str = "",
        dry_run: bool = False,
    ) -> Dict:
        """Write a new Python/text file to disk after syntax validation."""
        path = self._safe_path(file_path)
        if path is None:
            return {"ok": False, "error": f"Path '{file_path}' outside repo root — rejected"}

        if path.exists():
            return {"ok": False, "error": f"File already exists: {file_path}. Use patch_file() to modify."}

        # Syntax check if Python
        if file_path.endswith(".py"):
            valid, err = self._validate_python(content)
            if not valid:
                return {"ok": False, "error": f"Syntax error: {err}"}

        if dry_run:
            return {"ok": True, "dry_run": True, "path": str(path), "lines": len(content.splitlines())}

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._log(file_path, "write_new", origin, description, len(content.splitlines()))
        logger.info("CodeWriter wrote new file: %s (%d lines)", file_path, len(content.splitlines()))
        return {"ok": True, "path": str(path), "lines": len(content.splitlines())}

    def patch_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        origin: str = "self_generated",
        description: str = "",
        dry_run: bool = False,
    ) -> Dict:
        """Replace old_string with new_string in an existing file (with backup)."""
        path = self._safe_path(file_path)
        if path is None:
            return {"ok": False, "error": f"Path '{file_path}' outside repo root — rejected"}
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}

        original = path.read_text(encoding="utf-8")
        if old_string not in original:
            return {"ok": False, "error": f"Target string not found in {file_path}"}

        patched = original.replace(old_string, new_string, 1)

        if file_path.endswith(".py"):
            valid, err = self._validate_python(patched)
            if not valid:
                return {"ok": False, "error": f"Patched file has syntax error: {err}"}

        if dry_run:
            return {"ok": True, "dry_run": True, "path": str(path)}

        # Backup first
        backup_path = _BACKUP_DIR / f"{Path(file_path).name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.bak"
        shutil.copy2(path, backup_path)

        path.write_text(patched, encoding="utf-8")
        self._log(file_path, "patch", origin, description, len(patched.splitlines()))
        logger.info("CodeWriter patched: %s", file_path)
        return {"ok": True, "path": str(path), "backup": str(backup_path)}

    def append_method(
        self,
        file_path: str,
        class_name: str,
        method_code: str,
        origin: str = "self_generated",
        description: str = "",
        dry_run: bool = False,
    ) -> Dict:
        """
        Append a new method to a class in an existing Python file.
        Finds the last method in the class and inserts after it.
        """
        path = self._safe_path(file_path)
        if path is None:
            return {"ok": False, "error": "Path outside repo root"}
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}

        content = path.read_text(encoding="utf-8")
        lines   = content.splitlines()

        # Find class body end (last indented line before next class or EOF)
        in_class  = False
        last_line = None
        indent    = "    "
        for i, line in enumerate(lines):
            if line.strip().startswith(f"class {class_name}"):
                in_class = True
                continue
            if in_class:
                if line.startswith("class ") and not line.startswith(indent):
                    break   # next top-level class
                if line.strip():
                    last_line = i

        if last_line is None:
            return {"ok": False, "error": f"Class '{class_name}' not found in {file_path}"}

        # Ensure method is indented
        method_lines = []
        for ml in method_code.strip().splitlines():
            method_lines.append("    " + ml if ml.strip() else "")

        new_lines = (
            lines[:last_line + 1]
            + [""]
            + method_lines
            + lines[last_line + 1:]
        )
        new_content = "\n".join(new_lines) + "\n"

        valid, err = self._validate_python(new_content)
        if not valid:
            return {"ok": False, "error": f"Appended code has syntax error: {err}"}

        if dry_run:
            return {"ok": True, "dry_run": True, "inserted_after_line": last_line + 1}

        backup_path = _BACKUP_DIR / f"{Path(file_path).name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.bak"
        shutil.copy2(path, backup_path)
        path.write_text(new_content, encoding="utf-8")
        self._log(file_path, "append_method", origin, description, len(method_lines))
        logger.info("CodeWriter appended method to %s::%s", file_path, class_name)
        return {"ok": True, "path": str(path), "inserted_after_line": last_line + 1}

    def generate_component(
        self,
        component_name: str,
        description: str,
        requirements: List[str],
        origin: str = "self_generated",
        dry_run: bool = False,
    ) -> Dict:
        """
        Use AI to generate a new DMAI component from a description.
        Writes to components/<component_name>.py
        """
        if not self.ai_hub:
            return {"ok": False, "error": "No AI hub available for code generation"}

        prompt = f"""You are DMAI, an autonomous AI system. Generate a production-ready Python component.

Component name: {component_name}
Description: {description}
Requirements:
{chr(10).join(f'  - {r}' for r in requirements)}

Rules:
- Use only standard library + common packages (requests, sqlite3, pathlib, json, threading)
- Include class docstring
- Include logging via: logger = logging.getLogger("dmai.{component_name.lower()}")
- Do NOT use mock data
- All methods must have docstrings
- Include __all__ at the top
- Output ONLY the Python code, no markdown fences

Generate the complete component:"""

        try:
            import asyncio
            if hasattr(self.ai_hub, "query_all_tutors"):
                result = self.ai_hub.query_all_tutors(prompt)
                code = ""
                if isinstance(result, dict):
                    syn = result.get("synthesis")
                    if isinstance(syn, dict):
                        code = syn.get("unified_answer", "") or ""
                    elif isinstance(syn, str):
                        code = syn

                    if not code:
                        br = result.get("best_response", "")
                        if isinstance(br, str):
                            code = br
                        elif isinstance(br, dict):
                            code = br.get("unified_answer", "") or br.get("text", "") or ""

                    if not code:
                        responses = result.get("responses", {}) or {}
                        if isinstance(responses, dict):
                            for v in responses.values():
                                if isinstance(v, str) and v.strip():
                                    code = v
                                    break
                                elif isinstance(v, dict):
                                    cand = v.get("unified_answer") or v.get("text") or v.get("content") or v.get("response")
                                    if isinstance(cand, str) and cand.strip():
                                        code = cand
                                        break

                    code = code or ""
            elif hasattr(self.ai_hub, "chat"):
                loop = asyncio.new_event_loop()
                try:
                    code = loop.run_until_complete(self.ai_hub.chat(prompt))
                finally:
                    loop.close()
            else:
                return {"ok": False, "error": "AI hub has no usable chat method"}

            # Strip markdown fences if present
            code = str(code).strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            valid, err = self._validate_python(code)
            if not valid:
                logger.warning("Generated code has syntax error: %s", err)
                return {"ok": False, "error": f"Generated code syntax error: {err}", "code": code[:500]}

            file_path = f"components/{component_name}.py"
            result = self.write_new_file(
                file_path, code, origin=origin,
                description=description, dry_run=dry_run
            )
            result["generated_lines"] = len(code.splitlines())
            return result

        except Exception as e:
            logger.error("Code generation failed: %s", e)
            return {"ok": False, "error": str(e)}

    def execute_kaizen_fix(
        self,
        kaizen_id: str,
        file_path: str,
        problem: str,
        suggested_fix: str,
        origin: str = "kaizen_auto_repair",
    ) -> Dict:
        """
        Attempt to auto-repair a Kaizen item by patching the relevant file.
        Returns dict with ok, action, and details.
        """
        logger.info("Kaizen auto-repair: %s → %s", kaizen_id, problem[:60])

        # Check if file exists
        full_path = self._safe_path(file_path) if file_path else None
        if full_path and not full_path.exists():
            return {
                "ok": False,
                "kaizen_id": kaizen_id,
                "error": f"File not found: {file_path}",
            }

        # If we have a concrete suggested_fix with old/new markers, apply it
        if "<<<OLD>>>" in suggested_fix and "<<<NEW>>>" in suggested_fix:
            parts = suggested_fix.split("<<<OLD>>>", 1)
            old_new = parts[1].split("<<<NEW>>>", 1)
            if len(old_new) == 2:
                old_str, new_str = old_new[0].strip(), old_new[1].strip()
                if file_path:
                    return self.patch_file(
                        file_path, old_str, new_str,
                        origin=origin, description=f"Kaizen fix: {problem[:80]}"
                    )

        # Otherwise use AI to generate a fix
        if not self.ai_hub:
            return {"ok": False, "error": "No AI hub and no structured fix provided"}

        content_preview = ""
        if full_path and full_path.exists():
            content_preview = full_path.read_text(encoding="utf-8")[:3000]

        prompt = f"""You are DMAI fixing a bug in your own codebase.

Problem: {problem}
File: {file_path or 'unknown'}
Suggested approach: {suggested_fix}

Current file content (first 3000 chars):
{content_preview}

Output a structured patch in EXACTLY this format:
<<<OLD>>>
<the exact string to replace>
<<<NEW>>>
<the replacement string>

If no patch is needed or you cannot fix it, output: CANNOT_FIX: <reason>"""

        try:
            if hasattr(self.ai_hub, "query_all_tutors"):
                result = self.ai_hub.query_all_tutors(prompt)
                response = ""
                if isinstance(result, dict):
                    response = result.get("synthesis") or result.get("best_response") or ""
                    if not response:
                        for v in result.get("responses", {}).values():
                            if isinstance(v, dict) and v.get("response"):
                                response = v["response"]
                                break
            else:
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    response = loop.run_until_complete(self.ai_hub.chat(prompt))
                finally:
                    loop.close()

            response = str(response).strip()
            if response.startswith("CANNOT_FIX:"):
                return {"ok": False, "error": response}

            if "<<<OLD>>>" in response and "<<<NEW>>>" in response:
                parts = response.split("<<<OLD>>>", 1)
                old_new = parts[1].split("<<<NEW>>>", 1)
                if len(old_new) == 2:
                    old_str = old_new[0].strip()
                    new_str = old_new[1].split("<<<")[0].strip()
                    if file_path and old_str:
                        return self.patch_file(
                            file_path, old_str, new_str,
                            origin=origin, description=f"AI Kaizen fix: {problem[:80]}"
                        )

            return {"ok": False, "error": "AI response did not include a parseable patch"}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_path(self, file_path: str) -> Optional[Path]:
        """Return resolved path only if it's within the repo root."""
        try:
            p = (_REPO_ROOT / file_path).resolve()
            if _REPO_ROOT.resolve() in p.parents or p == _REPO_ROOT.resolve():
                return p
            logger.warning("Rejected path outside repo root: %s", file_path)
            return None
        except Exception:
            return None

    def _validate_python(self, code: str) -> Tuple[bool, str]:
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def _log(self, file_path: str, action: str, origin: str, description: str, lines: int) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "file": file_path,
            "origin": origin,
            "description": description[:200],
            "lines": lines,
        }
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_HISTORY_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Could not log CodeWriter action: %s", e)

    def get_history(self, limit: int = 20) -> List[Dict]:
        if not _HISTORY_FILE.exists():
            return []
        lines = _HISTORY_FILE.read_text().strip().splitlines()
        records = []
        for line in reversed(lines):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
            if len(records) >= limit:
                break
        return records
