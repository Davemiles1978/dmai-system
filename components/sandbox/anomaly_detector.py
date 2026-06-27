"""
AnomalyDetector — scans sandbox output (and the original code) for suspicious
patterns: credential leaks, network attempts, shell injection, file traversal,
encoding bypasses and oversized output.

Each detected anomaly is returned as a dict::

    {
        "category":     "CREDENTIAL_LEAK",
        "severity":     "CRITICAL",
        "match":        "<matched text, truncated to 100 chars>",
        "line_number":  12,
        "description":  "AWS access key id found in output",
    }
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

# Severity ordering used elsewhere to compute the "highest" severity.
SEVERITY_ORDER: Dict[str, int] = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

_MAX_MATCH = 100


def _compile(pattern: str, flags: int = 0) -> Pattern[str]:
    return re.compile(pattern, flags)


class AnomalyDetector:
    """Stateless scanner — safe to share a single instance across requests."""

    # ── Credential leak patterns ────────────────────────────────────────────
    _AWS_KEY = _compile(r"AKIA[0-9A-Z]{16}")
    _AWS_SECRET = _compile(r"[0-9a-zA-Z/+]{40}")
    _GCP_KEY = _compile(r"AIza[0-9A-Za-z\-_]{35}")
    _PRIVATE_KEY = _compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")
    _JWT = _compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}")
    _PASSWORD = _compile(r"""password\s*=\s*["'][^"']{8,}["']""", re.IGNORECASE)

    # ── Network patterns ────────────────────────────────────────────────────
    _IPV4 = _compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    _URL = _compile(r"https?://[^\s\"'<>]{10,}")
    _NET_TOOL = _compile(r"\b(curl|wget|requests\.get|urllib|http\.client)\b")
    _NET_IMPORT = _compile(r"\bimport\s+(socket|requests|urllib|httpx|aiohttp)\b")

    # ── Shell injection patterns ────────────────────────────────────────────
    _EVAL = _compile(r"\beval\s*\(")
    _EXEC = _compile(r"\bexec\s*\(")
    _DUNDER_IMPORT = _compile(r"__import__\s*\(")
    _OS_SYSTEM = _compile(r"os\.system\s*\(")
    _SUBPROCESS = _compile(r"\bsubprocess\b")
    _IMPORTLIB = _compile(r"\bimportlib\b")
    _COMPILE = _compile(r"\bcompile\s*\(")

    # ── File traversal patterns ─────────────────────────────────────────────
    _TRAVERSAL = _compile(r"\.\./.*\.\.")
    _SENSITIVE_FILE = _compile(r"/etc/(passwd|shadow|sudoers|hosts)")
    _PROC = _compile(r"/proc/[0-9]")
    _SYS = _compile(r"/sys/(kernel|class|bus)")
    _ROOT_HOME = _compile(r"~root")

    # ── Encoding bypass patterns ────────────────────────────────────────────
    _BASE64 = _compile(r"[A-Za-z0-9+/]{200,}={0,2}")
    _HEX = _compile(r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){20,}")
    _UNICODE = _compile(r"\\u[0-9a-fA-F]{4}(?:\\u[0-9a-fA-F]{4}){10,}")

    _LARGE_OUTPUT_THRESHOLD = 10240

    def detect(self, stdout: str, stderr: str, code: str) -> List[Dict]:
        """Return a list of anomaly dicts found in output and/or code."""
        stdout = stdout or ""
        stderr = stderr or ""
        code = code or ""

        output = (stdout + "\n" + stderr)
        anomalies: List[Dict] = []

        anomalies.extend(self._scan_credentials(output, "output"))
        anomalies.extend(self._scan_credentials(code, "code"))
        anomalies.extend(self._scan_network(output, "output"))
        anomalies.extend(self._scan_network(code, "code"))
        anomalies.extend(self._scan_shell_injection(output, in_code=False))
        anomalies.extend(self._scan_shell_injection(code, in_code=True))
        anomalies.extend(self._scan_traversal(output, "output"))
        anomalies.extend(self._scan_traversal(code, "code"))
        anomalies.extend(self._scan_encoding(output, "output"))
        anomalies.extend(self._scan_encoding(code, "code"))

        if len(stdout) > self._LARGE_OUTPUT_THRESHOLD:
            anomalies.append(self._make(
                "LARGE_OUTPUT", "MEDIUM",
                f"stdout length={len(stdout)}", stdout, 0,
                "Output exceeds 10KB — possible data exfiltration attempt",
            ))

        return anomalies

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _line_number(text: str, index: int) -> int:
        return text.count("\n", 0, index) + 1

    @staticmethod
    def _make(category: str, severity: str, match: str,
              source_text: str, index: int, description: str) -> Dict:
        return {
            "category": category,
            "severity": severity,
            "match": (match or "")[:_MAX_MATCH],
            "line_number": AnomalyDetector._line_number(source_text, index),
            "description": description,
        }

    def _emit(self, pattern: Pattern[str], text: str, category: str,
              severity: str, description: str) -> List[Dict]:
        out: List[Dict] = []
        for m in pattern.finditer(text):
            out.append(self._make(category, severity, m.group(0),
                                   text, m.start(), description))
        return out

    # ── category scanners ────────────────────────────────────────────────────
    def _scan_credentials(self, text: str, where: str) -> List[Dict]:
        out: List[Dict] = []
        out += self._emit(self._AWS_KEY, text, "CREDENTIAL_LEAK", "CRITICAL",
                          f"AWS access key id found in {where}")
        out += self._emit(self._GCP_KEY, text, "CREDENTIAL_LEAK", "CRITICAL",
                          f"GCP API key found in {where}")
        out += self._emit(self._PRIVATE_KEY, text, "CREDENTIAL_LEAK", "CRITICAL",
                          f"Private key PEM header found in {where}")
        out += self._emit(self._JWT, text, "CREDENTIAL_LEAK", "CRITICAL",
                          f"JWT token found in {where}")
        out += self._emit(self._PASSWORD, text, "CREDENTIAL_LEAK", "CRITICAL",
                          f"Hardcoded password assignment found in {where}")
        # AWS secret keys only count when near the words 'secret' or 'aws'.
        for m in self._AWS_SECRET.finditer(text):
            window = text[max(0, m.start() - 40): m.end() + 40].lower()
            if "secret" in window or "aws" in window:
                out.append(self._make(
                    "CREDENTIAL_LEAK", "CRITICAL", m.group(0), text, m.start(),
                    f"Possible AWS secret access key found in {where}",
                ))
        return out

    def _scan_network(self, text: str, where: str) -> List[Dict]:
        out: List[Dict] = []
        out += self._emit(self._IPV4, text, "NETWORK_ATTEMPT", "HIGH",
                          f"IPv4 address found in {where}")
        out += self._emit(self._URL, text, "NETWORK_ATTEMPT", "HIGH",
                          f"URL found in {where}")
        out += self._emit(self._NET_TOOL, text, "NETWORK_ATTEMPT", "HIGH",
                          f"Network tool/library invocation found in {where}")
        out += self._emit(self._NET_IMPORT, text, "NETWORK_ATTEMPT", "HIGH",
                          f"Network library import found in {where}")
        return out

    def _scan_shell_injection(self, text: str, in_code: bool) -> List[Dict]:
        severity = "CRITICAL" if in_code else "HIGH"
        where = "code" if in_code else "output"
        patterns: List[Tuple[Pattern[str], str]] = [
            (self._EVAL, "eval() call"),
            (self._EXEC, "exec() call"),
            (self._DUNDER_IMPORT, "__import__() call"),
            (self._OS_SYSTEM, "os.system() call"),
            (self._SUBPROCESS, "subprocess usage"),
            (self._IMPORTLIB, "importlib usage"),
            (self._COMPILE, "compile() call"),
        ]
        out: List[Dict] = []
        for pat, desc in patterns:
            out += self._emit(pat, text, "SHELL_INJECTION", severity,
                              f"{desc} found in {where}")
        return out

    def _scan_traversal(self, text: str, where: str) -> List[Dict]:
        patterns: List[Tuple[Pattern[str], str]] = [
            (self._TRAVERSAL, "path traversal sequence"),
            (self._SENSITIVE_FILE, "access to sensitive system file"),
            (self._PROC, "access to /proc"),
            (self._SYS, "access to /sys"),
            (self._ROOT_HOME, "reference to root home"),
        ]
        out: List[Dict] = []
        for pat, desc in patterns:
            out += self._emit(pat, text, "FILE_TRAVERSAL", "HIGH",
                              f"{desc} found in {where}")
        return out

    def _scan_encoding(self, text: str, where: str) -> List[Dict]:
        out: List[Dict] = []
        out += self._emit(self._BASE64, text, "ENCODING_BYPASS", "MEDIUM",
                          f"Large base64 blob found in {where}")
        out += self._emit(self._HEX, text, "ENCODING_BYPASS", "MEDIUM",
                          f"Long hex escape string found in {where}")
        out += self._emit(self._UNICODE, text, "ENCODING_BYPASS", "MEDIUM",
                          f"Long unicode escape sequence found in {where}")
        return out


def highest_severity(anomalies: List[Dict]) -> str:
    """Return the highest severity present, or 'NONE' for an empty list."""
    best = "NONE"
    for a in anomalies:
        sev = a.get("severity", "NONE")
        if SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(best, 0):
            best = sev
    return best
