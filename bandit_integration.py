"""
DMAI Bandit Security Scanner Integration
=========================================
Integrates Bandit static analysis into the DMAI code generation pipeline.
Scans all generated Python code before it is returned or executed.
Falls back to AST-based scanner if Bandit is not installed.
"""

import ast
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bandit Scanner
# ---------------------------------------------------------------------------

class BanditScanner:
    """
    Runs Bandit static analysis on generated Python code.
    Falls back to AST scanner from security.py if Bandit is unavailable.
    """

    SEVERITY_BLOCK  = {"HIGH"}
    CONFIDENCE_BLOCK = {"HIGH"}

    def __init__(self):
        """Initialise scanner and detect Bandit availability."""
        self._bandit_available = self._check_bandit()
        if not self._bandit_available:
            logger.warning(
                "Bandit not installed. Using AST fallback scanner. "
                "Install with: pip install bandit>=1.7.0"
            )

    def _check_bandit(self) -> bool:
        """Check if bandit CLI is installed and executable."""
        try:
            result = subprocess.run(
                ["bandit", "--version"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def scan(self, code: str, filename: str = "generated.py") -> dict:
        """
        Scan Python code with Bandit or AST fallback.

        Returns:
            {
              "safe": bool,
              "high_severity_count": int,
              "issues": [...],
              "bandit_available": bool,
              "skipped": bool
            }
        """
        if not self._bandit_available:
            return self._ast_fallback_scan(code)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["bandit", "-r", tmp_path, "-f", "json", "-ll"],
                capture_output=True, timeout=30, text=True
            )
            try:
                report = json.loads(result.stdout)
            except Exception:
                report = {"results": [], "metrics": {}}

            issues = []
            high_count = 0
            for item in report.get("results", []):
                sev  = item.get("issue_severity", "LOW")
                conf = item.get("issue_confidence", "LOW")
                issue = {
                    "severity":   sev,
                    "confidence": conf,
                    "test_id":    item.get("test_id", ""),
                    "text":       item.get("issue_text", ""),
                    "line":       item.get("line_number", 0),
                }
                issues.append(issue)
                if sev in self.SEVERITY_BLOCK and conf in self.CONFIDENCE_BLOCK:
                    high_count += 1

            return {
                "safe":               high_count == 0,
                "high_severity_count": high_count,
                "issues":             issues,
                "bandit_available":   True,
                "skipped":            False,
            }
        except subprocess.TimeoutExpired:
            logger.warning("Bandit scan timed out for code snippet")
            return self._ast_fallback_scan(code)
        except Exception as e:
            logger.warning("Bandit scan failed: %s", e)
            return self._ast_fallback_scan(code)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _ast_fallback_scan(self, code: str) -> dict:
        """AST-based fallback scanner when Bandit is unavailable."""
        BANNED = {"exec", "eval", "compile", "__import__", "breakpoint"}
        issues = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = ""
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name in BANNED:
                        issues.append({
                            "severity":   "HIGH",
                            "confidence": "HIGH",
                            "test_id":    "AST001",
                            "text":       f"Banned call detected: {name}()",
                            "line":       getattr(node, "lineno", 0),
                        })
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = []
                    if isinstance(node, ast.Import):
                        mods = [a.name for a in node.names]
                    else:
                        mods = [node.module or ""]
                    for mod in mods:
                        if mod in ("subprocess", "pty"):
                            issues.append({
                                "severity":   "HIGH",
                                "confidence": "MEDIUM",
                                "test_id":    "AST002",
                                "text":       f"Dangerous import: {mod}",
                                "line":       getattr(node, "lineno", 0),
                            })
        except SyntaxError as e:
            issues.append({
                "severity": "HIGH", "confidence": "HIGH",
                "test_id": "AST000",
                "text": f"Syntax error in generated code: {e}",
                "line": getattr(e, "lineno", 0) or 0,
            })

        high = sum(1 for i in issues if i["severity"] == "HIGH" and i["confidence"] == "HIGH")
        return {
            "safe":                high == 0,
            "high_severity_count": high,
            "issues":              issues,
            "bandit_available":    False,
            "skipped":             False,
        }

    def scan_and_filter(self, code: str) -> Tuple[str, bool, List]:
        """
        Scan and strip HIGH-severity lines from generated code.
        Returns (filtered_code, is_safe, issues).
        """
        result = self.scan(code)
        if result["safe"]:
            return code, True, []

        # Strip lines flagged as HIGH severity
        bad_lines = {
            i["line"] for i in result["issues"]
            if i.get("severity") == "HIGH"
        }
        if bad_lines:
            lines   = code.splitlines()
            cleaned = [
                f"# [SECURITY FILTERED line {n+1}]" if (n + 1) in bad_lines else l
                for n, l in enumerate(lines)
            ]
            filtered = "\n".join(cleaned)
            logger.warning(
                "BanditScanner: filtered %d HIGH-severity line(s) from generated code",
                len(bad_lines)
            )
            return filtered, False, result["issues"]

        return code, False, result["issues"]

    def ensure_bandit_in_requirements(self) -> None:
        """Add bandit to requirements.txt if missing."""
        candidates = [
            Path(__file__).parent.parent / "requirements.txt",
            Path("requirements.txt"),
        ]
        for req_path in candidates:
            if req_path.exists():
                content = req_path.read_text()
                if "bandit" not in content.lower():
                    with open(req_path, "a") as f:
                        f.write("\nbandit>=1.7.0\n")
                    logger.info("Added bandit to %s", req_path)
                return


# ---------------------------------------------------------------------------
# Flask decorator
# ---------------------------------------------------------------------------

def scan_route_decorator(f):
    """
    Flask route decorator that scans any 'code' field in request JSON body.
    Returns 400 if HIGH-severity issues found.
    """
    import functools
    from flask import request, jsonify

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        data = request.get_json(silent=True) or {}
        code = data.get("code", "")
        if code:
            scanner = BanditScanner()
            _, safe, issues = scanner.scan_and_filter(code)
            if not safe:
                return jsonify({
                    "error": "Generated code failed security scan",
                    "issues": issues[:5],   # limit response size
                }), 400
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_code_safely(spec: str, generator_fn) -> Tuple[str, dict]:
    """
    Wrap a code generation function with Bandit scanning.
    Returns (code, scan_result). Code may be filtered if issues found.
    """
    raw_code = generator_fn(spec)
    scanner  = BanditScanner()
    filtered, is_safe, issues = scanner.scan_and_filter(raw_code)
    return filtered, {
        "safe":   is_safe,
        "issues": issues,
        "original_length": len(raw_code),
        "filtered_length": len(filtered),
    }


# Singleton for reuse
_scanner = None

def get_scanner() -> BanditScanner:
    """Return module-level BanditScanner singleton."""
    global _scanner
    if _scanner is None:
        _scanner = BanditScanner()
    return _scanner
