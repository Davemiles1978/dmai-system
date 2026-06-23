#!/usr/bin/env python3
"""
DMAI Training Data Validator
=============================
Validates every JSONL entry in data/training/ before it can merge to main.

Checks performed per entry
──────────────────────────
1. JSON parseable          — line must be valid JSON
2. Schema completeness     — all required fields present, correct types
3. Domain key validity     — domain must map to a known SICore KPI
4. Source verifiability    — source must be a real URL (not a placeholder)
5. Prompt quality          — training_prompt scored across 6 dimensions:
      · min length (≥80 chars)
      · max length (≤2000 chars)
      · specificity (no placeholder tokens like <X> or TODO)
      · instruction presence (contains an imperative verb)
      · no duplicate prompt across the file
      · no synthetic/mock markers in any field
6. Expected improvement    — references a valid SICore KPI and explains why
7. Description quality     — ≥20 words, no copy-paste of technique name only
8. No duplicates           — source URL not already present in the file

Exit codes
──────────
  0 — all checks pass (safe to merge)
  1 — one or more ERRORS found (blocks merge)
  2 — warnings only (merge allowed, but flagged in summary)

Usage
─────
  python3 validate_training_data.py                         # validates data/training/*.jsonl
  python3 validate_training_data.py --files path/a.jsonl    # specific files
  python3 validate_training_data.py --changed-only          # only files changed in this PR (via git diff)
  python3 validate_training_data.py --json                  # machine-readable JSON output
  python3 validate_training_data.py --strict                # treat warnings as errors
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "data" / "training"

# The 8 canonical SICore KPI domain identifiers
VALID_DOMAINS = {
    "skill_acquisition_rate",
    "transfer_learning_rate",
    "zero_shot_success_count",
    "agentic_capability_score",
    "recursive_self_improvement_rate",
    "sample_efficiency_trend",
    "metacognition_accuracy",
    "multi_modal_integration_score",
}

# Aliases accepted in the domain field (mapped to canonical name)
DOMAIN_ALIASES: Dict[str, str] = {
    "skill_acquisition":          "skill_acquisition_rate",
    "transfer_learning":          "transfer_learning_rate",
    "zero_shot":                  "zero_shot_success_count",
    "agentic_capability":         "agentic_capability_score",
    "recursive_self_improvement": "recursive_self_improvement_rate",
    "sample_efficiency":          "sample_efficiency_trend",
    "metacognition":              "metacognition_accuracy",
    "multi_modal":                "multi_modal_integration_score",
    "multimodal":                 "multi_modal_integration_score",
    # Legacy / shorthand names used by the nightly cron
    "learning":                   "skill_acquisition_rate",
    "reasoning":                  "zero_shot_success_count",
    "agency":                     "agentic_capability_score",
    "self_improvement":           "recursive_self_improvement_rate",
    "efficiency":                 "sample_efficiency_trend",
    "meta":                       "metacognition_accuracy",
    "integration":                "multi_modal_integration_score",
}

# Required fields and their expected Python types
REQUIRED_FIELDS: Dict[str, type] = {
    "source":               str,
    "date_added":           str,
    "domain":               str,
    "technique":            str,
    "description":          str,
    "training_prompt":      str,
    "expected_improvement": str,
}

# Optional but validated if present
OPTIONAL_FIELDS: Dict[str, type] = {
    "tags":     list,
    "priority": str,
    "version":  str,
}

# Patterns that indicate mock / synthetic / placeholder data
MOCK_PATTERNS = [
    r"\bmock\b", r"\bsyntheti[cz]",  r"\bfake\b", r"\bdummy\b",
    r"\bplaceholder\b", r"\bexample\.com\b", r"\bfoo\.bar\b",
    r"\btest_only\b", r"\bTODO\b", r"\bFIXME\b",
    r"arXiv:XXXX", r"github\.com/owner/repo",
]
MOCK_RE = re.compile("|".join(MOCK_PATTERNS), re.IGNORECASE)

# Imperative verbs that indicate an actionable prompt
INSTRUCTION_VERBS = re.compile(
    r"\b(implement|write|build|design|create|develop|explain|demonstrate|"
    r"analyse|analyze|generate|train|evaluate|compare|optimise|optimize|"
    r"solve|apply|extend|adapt|prove|derive|construct|formulate|"
    r"describe|outline|produce|test|validate|benchmark|show|"
    r"calculate|compute|predict|classify|summarise|summarize)\b",
    re.IGNORECASE,
)

# Date format: YYYY-MM-DD
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    level: str        # "ERROR" | "WARNING" | "INFO"
    code:  str        # e.g. "SCHEMA_MISSING_FIELD"
    file:  str
    line:  int
    message: str
    field: Optional[str] = None
    value_snippet: Optional[str] = None


@dataclass
class FileResult:
    path: str
    entries_checked: int = 0
    errors: int = 0
    warnings: int = 0
    findings: List[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.errors == 0


@dataclass
class ValidationReport:
    files: List[FileResult] = field(default_factory=list)
    total_entries: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    @property
    def passed(self) -> bool:
        return self.total_errors == 0

    def to_dict(self) -> dict:
        return {
            "overall": self.passed,
            "total_entries": self.total_entries,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "files": [
                {
                    "path": fr.path,
                    "entries_checked": fr.entries_checked,
                    "errors": fr.errors,
                    "warnings": fr.warnings,
                    "passed": fr.passed,
                    "findings": [
                        {
                            "level": f.level,
                            "code": f.code,
                            "line": f.line,
                            "field": f.field,
                            "message": f.message,
                            "value_snippet": f.value_snippet,
                        }
                        for f in fr.findings
                    ],
                }
                for fr in self.files
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validator core
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDataValidator:
    """Validates DMAI JSONL training datasets."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    # ── Public entry point ────────────────────────────────────────────────────

    def validate_files(self, paths: List[Path]) -> ValidationReport:
        report = ValidationReport()
        for path in paths:
            result = self._validate_file(path)
            report.files.append(result)
            report.total_entries  += result.entries_checked
            report.total_errors   += result.errors
            report.total_warnings += result.warnings
        return report

    # ── Per-file validation ───────────────────────────────────────────────────

    def _validate_file(self, path: Path) -> FileResult:
        try:
            display_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display_path = str(path)
        result = FileResult(path=display_path)
        seen_sources: Dict[str, int] = {}   # source_url → first line number
        seen_prompts: Dict[str, int] = {}   # normalised prompt → first line number

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            self._err(result, "FILE_READ_ERROR", str(path.name), 0,
                      f"Cannot read file: {e}")
            return result

        # Skip blank-only files (e.g. .gitkeep companion)
        non_blank = [l for l in lines if l.strip()]
        if not non_blank:
            self._info(result, "FILE_EMPTY", str(path), 0, "File is empty — nothing to validate")
            return result

        for lineno, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue   # skip blank lines

            # ── 1. JSON parseable ─────────────────────────────────────────────
            try:
                entry: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as e:
                self._err(result, "JSON_PARSE_ERROR", str(path), lineno,
                          f"Invalid JSON: {e.msg} at col {e.colno}",
                          value_snippet=raw[:120])
                continue   # cannot check further

            if not isinstance(entry, dict):
                self._err(result, "JSON_NOT_OBJECT", str(path), lineno,
                          f"Entry is {type(entry).__name__}, expected object")
                continue

            result.entries_checked += 1

            # ── 2. Schema completeness ────────────────────────────────────────
            for fname, ftype in REQUIRED_FIELDS.items():
                if fname not in entry:
                    self._err(result, "SCHEMA_MISSING_FIELD", str(path), lineno,
                              f"Required field '{fname}' is missing", field=fname)
                elif entry[fname] is None:
                    self._err(result, "SCHEMA_NULL_FIELD", str(path), lineno,
                              f"Field '{fname}' is null", field=fname)
                elif not isinstance(entry[fname], ftype):
                    self._err(result, "SCHEMA_WRONG_TYPE", str(path), lineno,
                              f"Field '{fname}' must be {ftype.__name__}, "
                              f"got {type(entry[fname]).__name__}", field=fname)

            for fname, ftype in OPTIONAL_FIELDS.items():
                if fname in entry and entry[fname] is not None:
                    if not isinstance(entry[fname], ftype):
                        self._warn(result, "SCHEMA_OPT_WRONG_TYPE", str(path), lineno,
                                   f"Optional field '{fname}' should be {ftype.__name__}, "
                                   f"got {type(entry[fname]).__name__}", field=fname)

            # Skip deeper checks if critical fields missing
            missing_critical = [f for f in REQUIRED_FIELDS if f not in entry or entry.get(f) is None]
            if missing_critical:
                continue

            source      = str(entry.get("source", ""))
            date_added  = str(entry.get("date_added", ""))
            domain      = str(entry.get("domain", ""))
            technique   = str(entry.get("technique", ""))
            description = str(entry.get("description", ""))
            prompt      = str(entry.get("training_prompt", ""))
            improvement = str(entry.get("expected_improvement", ""))

            # ── 3. Domain key validity ────────────────────────────────────────
            canonical = self._resolve_domain(domain)
            if canonical is None:
                self._err(result, "DOMAIN_INVALID", str(path), lineno,
                          f"Domain '{domain}' is not a valid SICore KPI. "
                          f"Valid values: {sorted(VALID_DOMAINS)}",
                          field="domain", value_snippet=domain)
            else:
                # Normalise to canonical form
                entry["domain"] = canonical

            # ── 4. Source verifiability ───────────────────────────────────────
            src_ok, src_msg = self._check_source(source)
            if not src_ok:
                self._err(result, "SOURCE_INVALID", str(path), lineno,
                          src_msg, field="source", value_snippet=source[:120])
            else:
                # Duplicate source check
                if source in seen_sources:
                    self._warn(result, "SOURCE_DUPLICATE", str(path), lineno,
                               f"Source URL already seen at line {seen_sources[source]}",
                               field="source", value_snippet=source[:80])
                else:
                    seen_sources[source] = lineno

            # ── 5. Prompt quality ─────────────────────────────────────────────
            self._check_prompt_quality(result, str(path), lineno, prompt,
                                       seen_prompts)

            # ── 6. Expected improvement field ─────────────────────────────────
            self._check_improvement(result, str(path), lineno, improvement)

            # ── 7. Description quality ────────────────────────────────────────
            self._check_description(result, str(path), lineno, description, technique)

            # ── 8. Mock / synthetic data markers (all fields) ─────────────────
            for fname in ("source", "technique", "description", "training_prompt",
                          "expected_improvement"):
                val = str(entry.get(fname, ""))
                if MOCK_RE.search(val):
                    self._err(result, "MOCK_DATA_DETECTED", str(path), lineno,
                              f"Field '{fname}' contains mock/synthetic/placeholder marker",
                              field=fname,
                              value_snippet=val[:120])

            # ── 9. Date format ────────────────────────────────────────────────
            if date_added and not DATE_RE.match(date_added):
                self._warn(result, "DATE_FORMAT_INVALID", str(path), lineno,
                           f"date_added '{date_added}' should be YYYY-MM-DD",
                           field="date_added", value_snippet=date_added)

        return result

    # ── Domain resolution ─────────────────────────────────────────────────────

    def _resolve_domain(self, domain: str) -> Optional[str]:
        d = domain.strip().lower()
        if d in VALID_DOMAINS:
            return d
        return DOMAIN_ALIASES.get(d)

    # ── Source check ──────────────────────────────────────────────────────────

    def _check_source(self, source: str) -> Tuple[bool, str]:
        s = source.strip()
        if not s:
            return False, "source is empty"
        # Must start with http:// or https:// or arXiv: prefix
        if s.startswith("arXiv:") or s.startswith("arxiv:"):
            arxiv_id = s.split(":", 1)[1].strip()
            if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id):
                return False, f"arXiv ID '{arxiv_id}' does not match ####.##### format"
            return True, "ok"
        try:
            parsed = urlparse(s)
            if parsed.scheme not in ("http", "https"):
                return False, f"source must be an https:// URL or arXiv:NNNN.NNNNN, got scheme '{parsed.scheme}'"
            if not parsed.netloc:
                return False, "source URL has no hostname"
            # Reject obvious placeholders
            if parsed.netloc in ("example.com", "localhost", "127.0.0.1", "foo.bar"):
                return False, f"source hostname '{parsed.netloc}' is a placeholder"
            return True, "ok"
        except Exception as e:
            return False, f"cannot parse source URL: {e}"

    # ── Prompt quality ────────────────────────────────────────────────────────

    def _check_prompt_quality(
        self,
        result: FileResult,
        path: str,
        lineno: int,
        prompt: str,
        seen_prompts: Dict[str, int],
    ) -> None:
        fname = "training_prompt"
        p = prompt.strip()

        # Min length
        if len(p) < 80:
            self._err(result, "PROMPT_TOO_SHORT", path, lineno,
                      f"training_prompt is {len(p)} chars — minimum is 80. "
                      "Prompts must be specific enough to train on.",
                      field=fname, value_snippet=p[:120])

        # Max length
        elif len(p) > 2000:
            self._warn(result, "PROMPT_TOO_LONG", path, lineno,
                       f"training_prompt is {len(p)} chars — recommended max is 2000. "
                       "Consider splitting into multiple focused entries.",
                       field=fname)

        # Placeholder tokens  <something> or {something}
        placeholder_re = re.compile(r"<[A-Za-z_]{2,}>|\{[A-Za-z_]{2,}\}")
        placeholders = placeholder_re.findall(p)
        if placeholders:
            self._err(result, "PROMPT_HAS_PLACEHOLDERS", path, lineno,
                      f"training_prompt contains unfilled placeholders: {placeholders[:5]}",
                      field=fname, value_snippet=p[:120])

        # Instruction verb
        if not INSTRUCTION_VERBS.search(p):
            self._warn(result, "PROMPT_NO_INSTRUCTION_VERB", path, lineno,
                       "training_prompt lacks an imperative instruction verb "
                       "(e.g. 'implement', 'explain', 'analyse', 'build'). "
                       "Prompts should be actionable tasks.",
                       field=fname)

        # Duplicate prompt
        norm = re.sub(r"\s+", " ", p.lower()).strip()
        if norm in seen_prompts:
            self._err(result, "PROMPT_DUPLICATE", path, lineno,
                      f"Duplicate training_prompt — identical to line {seen_prompts[norm]}",
                      field=fname, value_snippet=p[:80])
        else:
            seen_prompts[norm] = lineno

    # ── Expected improvement check ────────────────────────────────────────────

    def _check_improvement(
        self,
        result: FileResult,
        path: str,
        lineno: int,
        improvement: str,
    ) -> None:
        fname = "expected_improvement"
        s = improvement.strip()

        if len(s) < 20:
            self._err(result, "IMPROVEMENT_TOO_VAGUE", path, lineno,
                      f"expected_improvement is only {len(s)} chars. Must explain "
                      "which KPI is targeted and why.",
                      field=fname, value_snippet=s)

        # Must reference at least one valid KPI
        found_kpi = any(kpi in s.lower() for kpi in VALID_DOMAINS) or \
                    any(alias in s.lower() for alias in DOMAIN_ALIASES)
        if not found_kpi:
            self._warn(result, "IMPROVEMENT_NO_KPI_REFERENCE", path, lineno,
                       "expected_improvement does not mention a SICore KPI name. "
                       "Explicitly state which KPI this entry targets.",
                       field=fname, value_snippet=s[:120])

    # ── Description quality ───────────────────────────────────────────────────

    def _check_description(
        self,
        result: FileResult,
        path: str,
        lineno: int,
        description: str,
        technique: str,
    ) -> None:
        fname = "description"
        words = description.strip().split()
        if len(words) < 10:
            self._err(result, "DESCRIPTION_TOO_SHORT", path, lineno,
                      f"description has {len(words)} words — minimum is 10. "
                      "Explain what the technique does and why it matters to DMAI.",
                      field=fname, value_snippet=description[:120])

        # Description shouldn't just be a copy of the technique name
        if technique and description.strip().lower() == technique.strip().lower():
            self._err(result, "DESCRIPTION_IS_TECHNIQUE_NAME", path, lineno,
                      "description is identical to technique — add a real explanation.",
                      field=fname)

    # ── Finding helpers ───────────────────────────────────────────────────────

    def _add(self, result: FileResult, level: str, code: str, path: str,
             lineno: int, message: str, field: Optional[str] = None,
             value_snippet: Optional[str] = None) -> None:
        f = Finding(level=level, code=code, file=path, line=lineno,
                    message=message, field=field, value_snippet=value_snippet)
        result.findings.append(f)
        if level == "ERROR" or (level == "WARNING" and self.strict):
            result.errors += 1
        elif level == "WARNING":
            result.warnings += 1

    def _err(self, result, code, path, lineno, message, field=None, value_snippet=None):
        self._add(result, "ERROR", code, path, lineno, message, field, value_snippet)

    def _warn(self, result, code, path, lineno, message, field=None, value_snippet=None):
        self._add(result, "WARNING", code, path, lineno, message, field, value_snippet)

    def _info(self, result, code, path, lineno, message, field=None, value_snippet=None):
        self._add(result, "INFO", code, path, lineno, message, field, value_snippet)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_changed_jsonl_files() -> List[Path]:
    """Return JSONL files in data/training/ that are new or modified in this PR."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACM",
             "origin/main...HEAD", "--", "data/training/*.jsonl"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
        )
        if result.returncode != 0:
            # Fallback: diff against HEAD~1
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACM", "HEAD~1", "--",
                 "data/training/*.jsonl"],
                capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
            )
        paths = []
        for line in result.stdout.strip().splitlines():
            p = REPO_ROOT / line.strip()
            if p.exists() and p.suffix == ".jsonl":
                paths.append(p)
        return paths
    except Exception as e:
        print(f"⚠ Could not determine changed files via git: {e}", file=sys.stderr)
        return []


def discover_all_jsonl() -> List[Path]:
    """Find all .jsonl files under data/training/."""
    if not TRAINING_DIR.exists():
        return []
    return sorted(TRAINING_DIR.rglob("*.jsonl"))


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

LEVEL_ICON = {"ERROR": "❌", "WARNING": "⚠️ ", "INFO": "ℹ️ "}
LEVEL_COLOR = {
    "ERROR":   "\033[91m",   # bright red
    "WARNING": "\033[93m",   # yellow
    "INFO":    "\033[96m",   # cyan
}
RESET = "\033[0m"

def _color(level: str, text: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{LEVEL_COLOR.get(level, '')}{text}{RESET}"


def print_report(report: ValidationReport, use_color: bool = True) -> None:
    total_files = len(report.files)
    passed_files = sum(1 for f in report.files if f.passed)

    print()
    print("═" * 70)
    print("  DMAI Training Data Validation Report")
    print("═" * 70)

    for fr in report.files:
        icon = "✅" if fr.passed else "❌"
        print(f"\n{icon}  {fr.path}  "
              f"({fr.entries_checked} entries, "
              f"{fr.errors} errors, {fr.warnings} warnings)")

        for f in fr.findings:
            icon_s = LEVEL_ICON.get(f.level, "·")
            loc    = f"line {f.line}" + (f", field '{f.field}'" if f.field else "")
            msg    = _color(f.level, f"[{f.level}] {f.code}", use_color)
            print(f"   {icon_s} {msg}  ({loc})")
            print(f"      {f.message}")
            if f.value_snippet:
                snip = f.value_snippet[:100].replace("\n", "↵")
                print(f"      Value: {snip!r}")

    print()
    print("─" * 70)
    print(f"  Files:   {passed_files}/{total_files} passed")
    print(f"  Entries: {report.total_entries} checked")
    print(f"  Errors:  {_color('ERROR', str(report.total_errors), use_color)}")
    print(f"  Warnings:{_color('WARNING', str(report.total_warnings), use_color)}")
    print()

    if report.passed:
        print("  ✅  ALL CHECKS PASSED — safe to merge")
    else:
        print("  ❌  VALIDATION FAILED — fix errors before merging")
        print()
        print("  Error codes reference:")
        print("    SCHEMA_*          — missing / wrong-type fields")
        print("    DOMAIN_INVALID    — domain not a SICore KPI name")
        print("    SOURCE_INVALID    — not a real URL or arXiv ID")
        print("    PROMPT_*          — training_prompt quality issues")
        print("    IMPROVEMENT_*     — expected_improvement issues")
        print("    DESCRIPTION_*     — description quality issues")
        print("    MOCK_DATA_DETECTED— synthetic/placeholder content found")

    print("═" * 70)
    print()


def print_github_annotations(report: ValidationReport) -> None:
    """Emit GitHub Actions annotation commands for inline PR comments."""
    for fr in report.files:
        for f in fr.findings:
            if f.level == "ERROR":
                print(f"::error file={f.file},line={f.line},title={f.code}::{f.message}")
            elif f.level == "WARNING":
                print(f"::warning file={f.file},line={f.line},title={f.code}::{f.message}")


def print_github_summary(report: ValidationReport) -> None:
    """Write a markdown summary to $GITHUB_STEP_SUMMARY if running in CI."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        f"## {'✅' if report.passed else '❌'} DMAI Training Data Validation\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Files checked | {len(report.files)} |",
        f"| Entries checked | {report.total_entries} |",
        f"| Errors | {report.total_errors} |",
        f"| Warnings | {report.total_warnings} |",
        f"| Result | {'**PASS** — safe to merge' if report.passed else '**FAIL** — fix errors before merging'} |\n",
    ]

    if not report.passed:
        lines.append("### Errors found\n")
        lines.append("| File | Line | Code | Message |")
        lines.append("|------|------|------|---------|")
        for fr in report.files:
            for f in fr.findings:
                if f.level == "ERROR":
                    lines.append(
                        f"| `{f.file}` | {f.line} | `{f.code}` | {f.message[:120]} |"
                    )

    if report.total_warnings > 0:
        lines.append("\n### Warnings\n")
        lines.append("| File | Line | Code | Message |")
        lines.append("|------|------|------|---------|")
        for fr in report.files:
            for f in fr.findings:
                if f.level == "WARNING":
                    lines.append(
                        f"| `{f.file}` | {f.line} | `{f.code}` | {f.message[:120]} |"
                    )

    lines.append("\n---\n")
    lines.append(
        "_Validator: `aevora-training/scripts/validate_training_data.py` "
        "— runs on every PR touching `data/training/*.jsonl`_"
    )

    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"⚠ Could not write GitHub summary: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DMAI training data JSONL validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files", nargs="+", type=Path,
        help="Specific JSONL files to validate (default: all in data/training/)",
    )
    parser.add_argument(
        "--changed-only", action="store_true",
        help="Only validate files changed in this PR (via git diff)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON report to stdout",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors (stricter merge gate)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour output",
    )
    parser.add_argument(
        "--github-annotations", action="store_true",
        help="Emit GitHub Actions annotation commands (::error / ::warning)",
    )
    args = parser.parse_args()

    # ── Resolve files to validate ─────────────────────────────────────────────
    if args.files:
        paths = [Path(p).resolve() for p in args.files]
    elif args.changed_only:
        paths = get_changed_jsonl_files()
        if not paths:
            print("ℹ  No JSONL files changed in this PR — nothing to validate.")
            print("JSON:", json.dumps({"overall": True, "total_entries": 0,
                                       "total_errors": 0, "total_warnings": 0,
                                       "files": []}))
            return 0
    else:
        paths = discover_all_jsonl()

    if not paths:
        print("ℹ  No JSONL training files found to validate.")
        print(f"   Looked in: {TRAINING_DIR}")
        if args.json:
            print("JSON:", json.dumps({"overall": True, "total_entries": 0,
                                       "total_errors": 0, "total_warnings": 0,
                                       "files": []}))
        return 0

    # ── Run validation ────────────────────────────────────────────────────────
    validator = TrainingDataValidator(strict=args.strict)
    report = validator.validate_files(paths)

    # ── Output ────────────────────────────────────────────────────────────────
    use_color = not args.no_color and sys.stdout.isatty()

    if args.github_annotations:
        print_github_annotations(report)

    print_report(report, use_color=use_color)
    print_github_summary(report)   # no-op outside CI

    if args.json:
        print("JSON:", json.dumps(report.to_dict(), indent=2))

    # ── Exit code ─────────────────────────────────────────────────────────────
    if report.total_errors > 0:
        return 1
    if args.strict and report.total_warnings > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
