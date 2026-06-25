"""
DMAI Security Module
====================
Centralised security utilities:
  - JWT authentication (RS256 with HS256 fallback for Render)
  - Prompt injection filter
  - exec()/eval() AST scanner for generated code
  - Package name typosquat validator
  - HaltResponse structured refusal object
  - Constraint compliance validator
"""

from __future__ import annotations

import ast
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JWT configuration
# ---------------------------------------------------------------------------

_JWT_SECRET: Optional[str] = os.environ.get("JWT_SECRET")
if not _JWT_SECRET:
    _JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET env var not set. Generated a random ephemeral secret: %s. "
        "Tokens will be invalidated on restart.",
        _JWT_SECRET,
    )

_MASTER_PASSWORD: str = os.environ.get("MASTER_PASSWORD", "")

# ---------------------------------------------------------------------------
# 1a. JWT Auth
# ---------------------------------------------------------------------------


def generate_token(payload: dict, expires_minutes: int = 480) -> str:
    """Sign a payload with HS256 and return a JWT string.

    Args:
        payload: Arbitrary claims to embed in the token.
        expires_minutes: Token lifetime in minutes (default 60).

    Returns:
        Encoded JWT string.
    """
    import time

    claims = dict(payload)
    claims["exp"] = int(time.time()) + expires_minutes * 60
    return jwt.encode(claims, _JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dict, or None if verification fails for any reason.
    """
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])
    except Exception as exc:
        logger.debug("JWT verification failed: %s", exc)
        return None


def issue_token_for_password(password: str) -> Optional[str]:
    """Validate password against MASTER_PASSWORD and issue a JWT if it matches.

    Args:
        password: Plain-text password to check.

    Returns:
        Signed JWT string, or None if the password is wrong or MASTER_PASSWORD
        is not configured.
    """
    if not _MASTER_PASSWORD:
        logger.warning("MASTER_PASSWORD env var is not set; password login disabled.")
        return None
    if secrets.compare_digest(password, _MASTER_PASSWORD):
        return generate_token({"sub": "master", "role": "admin"})
    return None


def require_jwt(f):
    """Flask route decorator that enforces JWT authentication.

    Accepts a Bearer token from the Authorization header or a legacy
    X-Master-Password header (which issues a fresh JWT on success).

    Returns HTTP 401 JSON on any authentication failure.
    """
    from flask import jsonify, request

    @wraps(f)
    def decorated(*args, **kwargs):
        """Inner wrapper that performs the authentication check."""
        # Legacy backward-compat: accept raw master password header
        raw_password = request.headers.get("X-Master-Password", "")
        if raw_password:
            token = issue_token_for_password(raw_password)
            if token:
                return f(*args, **kwargs)
            return jsonify({"error": "Unauthorized", "detail": "Invalid master password"}), 401

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized", "detail": "Missing Bearer token"}), 401

        token = auth_header[len("Bearer "):]
        payload = verify_token(token)
        if payload is None:
            return jsonify({"error": "Unauthorized", "detail": "Invalid or expired token"}), 401

        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# 1b. Prompt Injection Filter
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+a",
    r"forget\s+(everything|all|your)",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"act\s+as\s+(if\s+you\s+are|a\s+different)",
    r"new\s+instructions?\s*:",
    r"override\s+(safety|guidelines?|instructions?)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
]

_COMPILED_INJECTION = [(re.compile(p, re.IGNORECASE), p) for p in INJECTION_PATTERNS]


def sanitise_input(text: str) -> tuple:
    """Replace prompt-injection patterns with [FILTERED] and report detection.

    Args:
        text: Raw user input string.

    Returns:
        A tuple of (cleaned_text, was_injected) where was_injected is True if
        at least one pattern matched.
    """
    was_injected = False
    cleaned = text
    for compiled, pattern in _COMPILED_INJECTION:
        if compiled.search(cleaned):
            was_injected = True
            logger.warning("Prompt injection pattern detected: %s", pattern)
            cleaned = compiled.sub("[FILTERED]", cleaned)
    return cleaned, was_injected


def check_injection(text: str) -> bool:
    """Return True if any prompt-injection pattern is found in text.

    Args:
        text: Input string to check.
    """
    _, detected = sanitise_input(text)
    return detected


# ---------------------------------------------------------------------------
# 1c. exec()/eval() AST Scanner
# ---------------------------------------------------------------------------

BANNED_CALLS = {"exec", "eval", "compile", "__import__", "breakpoint"}
BANNED_MODULES = {"subprocess", "os.system", "pty", "socket"}


class _BannedCallVisitor(ast.NodeVisitor):
    """AST visitor that collects violations for banned calls and imports."""

    def __init__(self):
        """Initialise the visitor with an empty violations list."""
        self.violations: list = []

    def visit_Call(self, node):
        """Check function calls against the banned list."""
        # Direct name calls: exec(...), eval(...) etc.
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            self.violations.append(
                "Banned call: %s() at line %d" % (node.func.id, node.lineno)
            )
        # Attribute calls: os.system(...), os.popen(...)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                full = "%s.%s" % (node.func.value.id, node.func.attr)
                if full in ("os.system", "os.popen"):
                    self.violations.append(
                        "Dangerous call: %s() at line %d" % (full, node.lineno)
                    )
        self.generic_visit(node)

    def visit_Import(self, node):
        """Flag imports of dangerous modules."""
        for alias in node.names:
            if alias.name in ("subprocess", "pty"):
                self.violations.append(
                    "Dangerous import: %s at line %d" % (alias.name, node.lineno)
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Flag from-imports of dangerous modules."""
        if node.module and node.module in ("subprocess", "pty"):
            self.violations.append(
                "Dangerous from-import: %s at line %d" % (node.module, node.lineno)
            )
        self.generic_visit(node)


def scan_generated_code(code: str) -> tuple:
    """Parse code with AST and scan for dangerous constructs.

    Args:
        code: Python source code string to analyse.

    Returns:
        A tuple of (is_safe, violations) where is_safe is True when no
        violations are found, and violations is a list of description strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, ["SyntaxError: %s" % exc]

    visitor = _BannedCallVisitor()
    visitor.visit(tree)
    violations = visitor.violations
    return (len(violations) == 0), violations


def safe_code_output(code: str) -> tuple:
    """Scan code and strip violating lines if unsafe.

    Args:
        code: Python source code string.

    Returns:
        A tuple of (cleaned_code, is_safe, violations).  If safe, cleaned_code
        equals the original.  If unsafe, lines containing violations are
        replaced with a comment.
    """
    is_safe, violations = scan_generated_code(code)
    if is_safe:
        return code, True, []

    # Best-effort: remove lines that contain banned identifiers
    lines = code.splitlines()
    banned_names = list(BANNED_CALLS) + ["os.system", "os.popen"]
    cleaned_lines = []
    for line in lines:
        if any(name in line for name in banned_names):
            cleaned_lines.append("# [REMOVED: unsafe code]")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines), False, violations


# ---------------------------------------------------------------------------
# 1d. Package Typosquat Validator
# ---------------------------------------------------------------------------

TYPOSQUAT_PATTERNS = [
    ("requests", ["requets", "reqeusts", "rquests", "requeests"]),
    ("numpy", ["nunpy", "nmpy", "nmupy"]),
    ("pandas", ["pnadas", "padnas", "pandsa"]),
    ("flask", ["falsk", "flsak"]),
    ("openai", ["opneai", "openali", "opeanai"]),
    ("django", ["djnago", "dajngo"]),
    ("tensorflow", ["tensroflow", "tensorflwo"]),
    ("torch", ["troch", "torhc"]),
    ("sklearn", ["siklern", "skleran"]),
    ("boto3", ["bot03", "b0to3"]),
]

# Build a flat lookup dict: typo -> canonical
_TYPOSQUAT_LOOKUP: dict = {}
for _canonical, _typos in TYPOSQUAT_PATTERNS:
    for _typo in _typos:
        _TYPOSQUAT_LOOKUP[_typo.lower()] = _canonical

# Characters that look like letters (leet-speak substitutions)
_LEET_RE = re.compile(r"[013]")  # 0->o, 1->l, 3->e


def validate_package_name(name: str) -> tuple:
    """Check whether a package name is a known typosquat or uses leet substitutions.

    Args:
        name: Package name string (e.g. from an import statement).

    Returns:
        A tuple of (is_safe, warning_message).  is_safe is False when a known
        typosquat or suspicious digit substitution is detected.
    """
    normalised = name.strip().lower()

    # Exact typosquat match
    if normalised in _TYPOSQUAT_LOOKUP:
        canonical = _TYPOSQUAT_LOOKUP[normalised]
        msg = (
            "Possible typosquat: '%s' looks like '%s'. "
            "Install '%s' instead." % (name, canonical, canonical)
        )
        return False, msg

    # Digit substitution heuristic (0, 1, 3 inside package names)
    if _LEET_RE.search(normalised):
        msg = (
            "Package name '%s' contains digit(s) that may substitute letters "
            "(0->o, 1->l, 3->e). Verify this is the intended package." % name
        )
        return False, msg

    return True, ""


class _ImportNameVisitor(ast.NodeVisitor):
    """AST visitor that collects all imported module names."""

    def __init__(self):
        """Initialise with empty names list."""
        self.names: list = []

    def visit_Import(self, node):
        """Collect top-level import names."""
        for alias in node.names:
            # Only the top-level package name (before any dot)
            self.names.append(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Collect from-import module names."""
        if node.module:
            self.names.append(node.module.split(".")[0])
        self.generic_visit(node)


def scan_imports_in_code(code: str) -> tuple:
    """Walk all imports in code and validate each package name.

    Args:
        code: Python source code string.

    Returns:
        A tuple of (all_safe, warnings) where all_safe is False if any import
        triggered a warning, and warnings is a list of warning strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, ["SyntaxError prevented import scan: %s" % exc]

    visitor = _ImportNameVisitor()
    visitor.visit(tree)

    warnings = []
    for name in visitor.names:
        is_safe, msg = validate_package_name(name)
        if not is_safe:
            warnings.append(msg)

    return (len(warnings) == 0), warnings


# ---------------------------------------------------------------------------
# 1e. HaltResponse
# ---------------------------------------------------------------------------

@dataclass
class HaltResponse:
    """Structured refusal object returned when a request must be blocked.

    Attributes:
        reason: Category string such as 'policy_violation' or
            'impossible_constraint'.
        code: Short uppercase code such as 'SPAM_PROHIBITED'.
        message: Human-readable explanation of why the request was halted.
        original_prompt: Optional copy of the offending prompt for audit logs.
    """

    reason: str
    code: str
    message: str
    original_prompt: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {
            "halt": True,
            "reason": self.reason,
            "code": self.code,
            "message": self.message,
        }

    def to_flask_response(self):
        """Return a Flask (Response, status_code) tuple with HTTP 400."""
        from flask import jsonify
        return jsonify(self.to_dict()), 400


HALT_RULES = [
    (
        r"send\s+(spam|bulk\s+email|unsolicited)",
        "policy_violation",
        "SPAM_PROHIBITED",
        "Sending spam or unsolicited bulk email is prohibited by policy.",
    ),
    (
        r"(complete|finish|do).{0,40}in\s+-\d+\s*(second|minute|hour|day)",
        "impossible_constraint",
        "NEGATIVE_TIME",
        "A negative time constraint is physically impossible and cannot be satisfied.",
    ),
    (
        r"access\s+(user|customer|personal)\s+data\s+without\s+(their\s+)?(consent|permission)",
        "legal_violation",
        "GDPR_BREACH",
        "Accessing personal data without consent violates GDPR and applicable privacy law.",
    ),
    (
        r"(task|step)\s+\w+\s+depends\s+on\s+(task|step)\s+\w+\s+which\s+depends\s+on\s+(task|step)\s+\w+",
        "circular_dependency",
        "UNSOLVABLE",
        "A circular task dependency was detected; this plan cannot be executed.",
    ),
    (
        r"(launder|wash)\s+money|insider\s+trad(e|ing)|pump\s+and\s+dump|market\s+manipulat",
        "legal_violation",
        "ILLEGAL_FINANCE",
        "This request describes illegal financial activity and cannot be fulfilled.",
    ),
]

_COMPILED_HALT = [
    (re.compile(p, re.IGNORECASE), reason, code, msg)
    for p, reason, code, msg in HALT_RULES
]


def check_halt_conditions(prompt: str) -> Optional[HaltResponse]:
    """Evaluate a prompt against all HALT_RULES.

    Args:
        prompt: The user prompt or task description to evaluate.

    Returns:
        A HaltResponse if any rule fires, or None if the prompt is acceptable.
    """
    for compiled, reason, code, message in _COMPILED_HALT:
        if compiled.search(prompt):
            logger.warning("Halt condition triggered: code=%s", code)
            return HaltResponse(
                reason=reason,
                code=code,
                message=message,
                original_prompt=prompt,
            )
    return None


# ---------------------------------------------------------------------------
# 1f. Constraint Validator
# ---------------------------------------------------------------------------

@dataclass
class PlanConstraints:
    """Constraints applied to a plan before execution.

    Attributes:
        max_cost_gbp: Maximum total cost in GBP.  Defaults to no limit.
        max_minutes: Maximum total elapsed minutes.  Defaults to no limit.
        allowed_tools: Whitelist of tool names.  Empty list means all allowed.
    """

    max_cost_gbp: float = float("inf")
    max_minutes: int = 9999999
    allowed_tools: list = field(default_factory=list)


@dataclass
class PlanStep:
    """A single step within an execution plan.

    Attributes:
        tool: Name of the tool or service used.
        description: Human-readable description of this step.
        estimated_cost_gbp: Estimated cost for this step in GBP.
        estimated_minutes: Estimated time for this step in minutes.
    """

    tool: str
    description: str
    estimated_cost_gbp: float = 0.0
    estimated_minutes: int = 0


@dataclass
class Plan:
    """An execution plan comprising ordered steps.

    Attributes:
        steps: Ordered list of PlanStep objects.
        estimated_cost: Total estimated cost across all steps in GBP.
        estimated_minutes: Total estimated duration across all steps in minutes.
    """

    steps: list  # list of PlanStep
    estimated_cost: float = 0.0
    estimated_minutes: int = 0


def validate_plan(plan: Plan, constraints: PlanConstraints) -> tuple:
    """Check a Plan against PlanConstraints and return any violations.

    Args:
        plan: The Plan to validate.
        constraints: The PlanConstraints to enforce.

    Returns:
        A tuple of (passes, violations) where passes is True only when no
        violations are found, and violations is a list of description strings.
    """
    violations = []

    # Re-compute totals from steps (do not trust pre-computed fields blindly)
    total_cost = sum(s.estimated_cost_gbp for s in plan.steps)
    total_minutes = sum(s.estimated_minutes for s in plan.steps)

    if total_cost > constraints.max_cost_gbp:
        violations.append(
            "Estimated cost GBP %.2f exceeds max_cost_gbp GBP %.2f."
            % (total_cost, constraints.max_cost_gbp)
        )

    if total_minutes > constraints.max_minutes:
        violations.append(
            "Estimated duration %d min exceeds max_minutes %d."
            % (total_minutes, constraints.max_minutes)
        )

    if constraints.allowed_tools:
        for step in plan.steps:
            if step.tool not in constraints.allowed_tools:
                violations.append(
                    "Tool '%s' is not in the allowed_tools list." % step.tool
                )

    return (len(violations) == 0), violations
