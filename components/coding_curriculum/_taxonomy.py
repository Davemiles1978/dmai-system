"""The coding-curriculum taxonomy.

800+ topic slugs across five languages/tools (Python, TypeScript/JS,
Bash, SQL, and cross-cutting CS), split into 4 tiers:

    tier 1 - foundations (variables, control flow, functions, ...)
    tier 2 - intermediate (data structures, OOP, error handling, ...)
    tier 3 - advanced (concurrency, networking, security, ...)
    tier 4 - specialisation (DMAI stack + your projects)

Each topic is a dict:
    {
        "slug":         "python.control_flow.if_else",
        "title":        "Python if/elif/else",
        "language":     "python",
        "tier":         1,
        "depth":        2,        # 1 shallow → 5 deep
        "prerequisites": ["python.variables"],
        "keywords":     ["conditional", "branch", "boolean"],
        "search_queries": ["python if elif else conditional statement"],
    }

The taxonomy is generated from a compact spec table so it stays
maintainable (adding a topic = adding one row, not editing 3 dicts).
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ── Tier definitions ─────────────────────────────────────────────────────

TOPIC_TIERS = {
    1: "foundations",
    2: "intermediate",
    3: "advanced",
    4: "specialisation",
}


# ── Compact spec for topic generation ────────────────────────────────────
#
# Each row:
#   (language, area, subtopic, tier, depth, prereqs_within_language,
#    keywords, extra_query_hints)
#
# `prereqs_within_language` is a list of "area.subtopic" strings that
# resolve to the same language. Cross-language prereqs (e.g. python
# depending on general CS.dsa.list) are added below via CROSS_LANG_PREREQ.

# Python: broad CS + industry + Flask/SQLite + DMAI stack
_PYTHON_SPEC: List[tuple] = [
    # ── Tier 1: foundations ─────────────────────────────────────────────
    ("core", "variables", 1, 1, [], ["binding", "assignment"], []),
    ("core", "types_primitives", 1, 1, ["core.variables"],
     ["int", "float", "str", "bool"], []),
    ("core", "operators", 1, 1, ["core.types_primitives"],
     ["arithmetic", "comparison", "logical"], []),
    ("core", "strings", 1, 2, ["core.types_primitives"],
     ["f-string", "format", "encode"], []),
    ("core", "control_flow_if", 1, 2, ["core.operators"],
     ["if", "elif", "else"], []),
    ("core", "control_flow_loops", 1, 2, ["core.control_flow_if"],
     ["for", "while", "break", "continue"], []),
    ("core", "functions_basics", 1, 2, ["core.control_flow_loops"],
     ["def", "return", "parameters"], []),
    ("core", "scope_lifetime", 1, 3, ["core.functions_basics"],
     ["LEGB", "closure", "nonlocal", "global"], []),
    ("core", "modules_imports", 1, 2, ["core.functions_basics"],
     ["import", "from", "sys.path"], []),
    ("core", "packages_layout", 1, 3, ["core.modules_imports"],
     ["__init__.py", "pyproject.toml", "namespace"], []),
    ("core", "docstrings_pep257", 1, 2, ["core.functions_basics"],
     ["PEP 257", "reStructuredText", "google-style"], []),
    ("core", "comments_naming_pep8", 1, 1, ["core.variables"],
     ["PEP 8", "snake_case", "readability"], []),
    ("core", "input_output", 1, 1, ["core.strings"],
     ["print", "input", "sys.stdout"], []),

    # ── Tier 2: intermediate ────────────────────────────────────────────
    ("dsa", "list", 2, 2, ["core.control_flow_loops"],
     ["list", "append", "slice"], []),
    ("dsa", "tuple", 2, 2, ["core.control_flow_loops"],
     ["tuple", "immutable"], []),
    ("dsa", "dict", 2, 2, ["dsa.list"],
     ["dict", "hash map", "keys", "values"], []),
    ("dsa", "set", 2, 2, ["dsa.list"],
     ["set", "frozenset", "union"], []),
    ("dsa", "comprehensions", 2, 3, ["dsa.dict"],
     ["list comprehension", "dict comprehension", "generator expression"], []),
    ("dsa", "iterators_generators", 2, 3, ["dsa.comprehensions"],
     ["iter", "next", "yield", "generator"], []),
    ("dsa", "collections_module", 2, 3, ["dsa.dict"],
     ["defaultdict", "Counter", "OrderedDict", "deque"], []),
    ("dsa", "sorting_searching", 2, 3, ["dsa.list"],
     ["sort", "sorted", "key", "bisect"], []),
    ("dsa", "algorithm_complexity", 2, 3, ["dsa.sorting_searching"],
     ["big-O", "time complexity", "space complexity"], []),

    ("oop", "classes_basics", 2, 2, ["core.functions_basics"],
     ["class", "self", "__init__"], []),
    ("oop", "inheritance", 2, 3, ["oop.classes_basics"],
     ["subclass", "super", "MRO"], []),
    ("oop", "polymorphism", 2, 3, ["oop.inheritance"],
     ["duck typing", "protocols"], []),
    ("oop", "encapsulation", 2, 2, ["oop.classes_basics"],
     ["private", "underscore", "property"], []),
    ("oop", "special_methods", 2, 3, ["oop.classes_basics"],
     ["__str__", "__repr__", "__eq__", "__hash__"], []),
    ("oop", "dataclasses", 2, 3, ["oop.classes_basics"],
     ["@dataclass", "field", "frozen"], []),
    ("oop", "abstract_classes", 2, 3, ["oop.inheritance"],
     ["ABC", "abstractmethod"], []),

    ("errors", "exceptions_basics", 2, 2, ["core.control_flow_if"],
     ["try", "except", "raise"], []),
    ("errors", "exception_hierarchy", 2, 3, ["errors.exceptions_basics"],
     ["BaseException", "custom exceptions"], []),
    ("errors", "context_managers", 2, 3, ["errors.exceptions_basics"],
     ["with", "__enter__", "__exit__", "contextlib"], []),
    ("errors", "logging_module", 2, 3, ["errors.exceptions_basics"],
     ["logging.getLogger", "handlers", "formatters"], []),
    ("errors", "assertion_patterns", 2, 2, ["errors.exceptions_basics"],
     ["assert", "invariants"], []),

    ("io", "files_text", 2, 2, ["core.strings"],
     ["open", "read", "write", "encoding"], []),
    ("io", "files_binary", 2, 2, ["io.files_text"],
     ["rb", "wb", "bytes"], []),
    ("io", "pathlib", 2, 2, ["io.files_text"],
     ["Path", "glob", "parts"], []),
    ("io", "json", 2, 2, ["dsa.dict"],
     ["json.dumps", "json.loads", "default"], []),
    ("io", "csv", 2, 2, ["io.files_text"],
     ["csv.DictReader", "csv.writer"], []),
    ("io", "yaml_toml", 2, 2, ["io.json"],
     ["yaml.safe_load", "tomllib"], []),
    ("io", "environment_variables", 2, 2, ["io.files_text"],
     ["os.environ", ".env", "dotenv"], []),

    ("stdlib", "datetime", 2, 3, ["core.types_primitives"],
     ["datetime", "timezone", "isoformat"], []),
    ("stdlib", "itertools", 2, 3, ["dsa.iterators_generators"],
     ["chain", "groupby", "product"], []),
    ("stdlib", "functools", 2, 3, ["core.functions_basics"],
     ["lru_cache", "partial", "reduce"], []),
    ("stdlib", "typing", 2, 3, ["oop.classes_basics"],
     ["type hints", "Optional", "Generic", "Protocol"], []),
    ("stdlib", "re_regex", 2, 3, ["core.strings"],
     ["re.match", "re.search", "capture groups"], []),
    ("stdlib", "hashlib", 2, 2, ["io.files_binary"],
     ["sha256", "hexdigest"], []),
    ("stdlib", "subprocess", 2, 3, ["io.files_text"],
     ["subprocess.run", "check_call", "shell"], []),
    ("stdlib", "argparse", 2, 3, ["core.modules_imports"],
     ["argparse.ArgumentParser", "add_argument"], []),

    # ── Tier 3: advanced ────────────────────────────────────────────────
    ("concurrency", "threading", 3, 4, ["errors.exceptions_basics"],
     ["threading.Thread", "Event", "Lock", "GIL"], []),
    ("concurrency", "multiprocessing", 3, 4, ["concurrency.threading"],
     ["multiprocessing.Process", "Pool", "Queue"], []),
    ("concurrency", "asyncio", 3, 5, ["concurrency.threading"],
     ["async def", "await", "event loop"], []),
    ("concurrency", "concurrent_futures", 3, 4, ["concurrency.threading"],
     ["ThreadPoolExecutor", "as_completed"], []),
    ("concurrency", "locks_and_races", 3, 5, ["concurrency.threading"],
     ["deadlock", "race condition", "atomic"], []),

    ("network", "http_client", 3, 3, ["io.json"],
     ["urllib", "requests", "httpx"], []),
    ("network", "http_server_basics", 3, 3, ["network.http_client"],
     ["socketserver", "http.server"], []),
    ("network", "sockets", 3, 4, ["network.http_client"],
     ["socket.AF_INET", "bind", "listen"], []),
    ("network", "tls_certificates", 3, 4, ["network.http_client"],
     ["ssl", "certifi", "SNI"], []),
    ("network", "rest_conventions", 3, 3, ["network.http_client"],
     ["REST", "verbs", "status codes"], []),
    ("network", "webhooks", 3, 3, ["network.rest_conventions"],
     ["POST", "signature verification", "idempotency"], []),
    ("network", "retries_backoff", 3, 4, ["network.http_client"],
     ["exponential backoff", "jitter", "circuit breaker"], []),

    ("security", "input_validation", 3, 4, ["errors.exceptions_basics"],
     ["allow-list", "sanitisation"], []),
    ("security", "sql_injection_prevention", 3, 4,
     ["security.input_validation"],
     ["parameterised query", "bind"], []),
    ("security", "secrets_management", 3, 4, ["io.environment_variables"],
     ["env vars", "vault", "rotation"], []),
    ("security", "auth_basics", 3, 4, ["network.rest_conventions"],
     ["basic auth", "bearer token"], []),
    ("security", "auth_oauth", 3, 5, ["security.auth_basics"],
     ["OAuth 2.0", "PKCE", "scopes"], []),
    ("security", "hashing_vs_encryption", 3, 4, ["stdlib.hashlib"],
     ["bcrypt", "argon2", "AES"], []),
    ("security", "csrf_xss_prevention", 3, 4, ["security.input_validation"],
     ["CSRF token", "XSS", "content security policy"], []),

    ("db", "sqlite_basics", 3, 3, ["io.files_binary"],
     ["sqlite3", "cursor", "commit"], []),
    ("db", "sqlite_schema_migration", 3, 4, ["db.sqlite_basics"],
     ["ALTER TABLE", "additive", "backfill"], []),
    ("db", "sqlite_wal_mode", 3, 4, ["db.sqlite_basics"],
     ["WAL", "journal_mode", "concurrent readers"], []),
    ("db", "sqlite_backup_recovery", 3, 4, ["db.sqlite_wal_mode"],
     ["backup API", "restore", "corruption"], []),
    ("db", "transactions_isolation", 3, 4, ["db.sqlite_basics"],
     ["BEGIN", "COMMIT", "isolation levels"], []),
    ("db", "sqlalchemy_core", 3, 4, ["db.sqlite_basics"],
     ["engine", "MetaData", "Core"], []),
    ("db", "orm_patterns", 3, 4, ["db.sqlalchemy_core"],
     ["session", "unit of work", "N+1"], []),
    ("db", "indexing_strategy", 3, 4, ["db.sqlite_basics"],
     ["B-tree", "covering index", "explain"], []),
    ("db", "query_optimisation", 3, 4, ["db.indexing_strategy"],
     ["EXPLAIN QUERY PLAN", "join order"], []),

    ("test", "unittest_basics", 3, 3, ["oop.classes_basics"],
     ["unittest.TestCase", "assertEqual"], []),
    ("test", "pytest_basics", 3, 3, ["test.unittest_basics"],
     ["pytest", "fixtures", "parametrize"], []),
    ("test", "test_doubles_mocks", 3, 4, ["test.pytest_basics"],
     ["Mock", "patch", "monkeypatch"], []),
    ("test", "property_based_testing", 3, 5, ["test.pytest_basics"],
     ["hypothesis", "invariants"], []),
    ("test", "coverage_measurement", 3, 4, ["test.pytest_basics"],
     ["coverage.py", "branch coverage"], []),
    ("test", "integration_vs_unit", 3, 4, ["test.pytest_basics"],
     ["seams", "test pyramid"], []),
    ("test", "load_and_stress_testing", 3, 5, ["test.integration_vs_unit"],
     ["locust", "k6"], []),

    ("perf", "profiling_cprofile", 3, 4, ["stdlib.functools"],
     ["cProfile", "pstats", "snakeviz"], []),
    ("perf", "profiling_line_memory", 3, 4, ["perf.profiling_cprofile"],
     ["line_profiler", "memory_profiler"], []),
    ("perf", "algorithmic_optimisation", 3, 4,
     ["dsa.algorithm_complexity"],
     ["reduce n", "amortised", "caching"], []),
    ("perf", "vectorisation_numpy", 3, 5, ["perf.algorithmic_optimisation"],
     ["numpy", "broadcasting"], []),
    ("perf", "cython_c_extensions", 3, 5, ["perf.vectorisation_numpy"],
     ["cython", "cffi", "cprofile hotspot"], []),

    ("packaging", "venv_and_deps", 3, 3, ["core.packages_layout"],
     ["venv", "pip", "requirements.txt"], []),
    ("packaging", "pyproject_toml", 3, 3, ["packaging.venv_and_deps"],
     ["setuptools", "hatch", "build backends"], []),
    ("packaging", "publishing_pypi", 3, 4, ["packaging.pyproject_toml"],
     ["twine", "pypi", "wheels"], []),
    ("packaging", "wheel_vs_sdist", 3, 4, ["packaging.publishing_pypi"],
     ["wheel", "manylinux"], []),

    ("patterns", "singleton", 3, 3, ["oop.classes_basics"],
     ["singleton", "borg"], []),
    ("patterns", "factory", 3, 3, ["oop.polymorphism"],
     ["factory method", "abstract factory"], []),
    ("patterns", "strategy", 3, 3, ["oop.polymorphism"],
     ["strategy pattern"], []),
    ("patterns", "observer", 3, 3, ["oop.polymorphism"],
     ["observer", "pub/sub"], []),
    ("patterns", "adapter", 3, 3, ["oop.polymorphism"],
     ["adapter", "wrapper"], []),
    ("patterns", "decorator_python", 3, 4, ["core.functions_basics"],
     ["@decorator", "functools.wraps"], []),
    ("patterns", "dependency_injection", 3, 4, ["oop.abstract_classes"],
     ["DI", "IoC"], []),
    ("patterns", "repository_pattern", 3, 4, ["db.orm_patterns"],
     ["repository", "data access layer"], []),
    ("patterns", "command_pattern", 3, 4, ["patterns.strategy"],
     ["command", "undo"], []),
    ("patterns", "state_machine", 3, 4, ["patterns.strategy"],
     ["FSM", "transitions"], []),

    ("arch", "layered_architecture", 3, 4, ["patterns.repository_pattern"],
     ["3-tier", "presentation", "domain", "data"], []),
    ("arch", "hexagonal_architecture", 3, 5, ["arch.layered_architecture"],
     ["ports and adapters", "clean architecture"], []),
    ("arch", "event_driven", 3, 5, ["patterns.observer"],
     ["event bus", "message broker"], []),
    ("arch", "microservices_vs_monolith", 3, 5, ["arch.layered_architecture"],
     ["monolith", "microservices", "modular monolith"], []),
    ("arch", "cqrs_and_es", 3, 5, ["arch.event_driven"],
     ["CQRS", "event sourcing"], []),

    ("obs", "logging_structured", 3, 3, ["errors.logging_module"],
     ["JSON logs", "correlation id"], []),
    ("obs", "metrics_and_kpis", 3, 4, ["obs.logging_structured"],
     ["counter", "gauge", "histogram"], []),
    ("obs", "tracing_distributed", 3, 5, ["obs.metrics_and_kpis"],
     ["opentelemetry", "spans"], []),
    ("obs", "alerting_slo_sli", 3, 5, ["obs.metrics_and_kpis"],
     ["SLO", "SLI", "error budget"], []),

    # ── Tier 4: SPECIALISATION (DMAI stack + your projects) ─────────────
    ("flask", "app_factory", 4, 3, ["network.rest_conventions"],
     ["Flask app factory", "blueprints"], []),
    ("flask", "routes_and_verbs", 4, 3, ["flask.app_factory"],
     ["@app.route", "methods="], []),
    ("flask", "request_response", 4, 3, ["flask.routes_and_verbs"],
     ["request.json", "jsonify"], []),
    ("flask", "auth_basic", 4, 4, ["flask.routes_and_verbs",
                                     "security.auth_basics"],
     ["MASTER_PASSWORD", "check_auth"], []),
    ("flask", "error_handlers", 4, 3, ["flask.routes_and_verbs"],
     ["errorhandler", "abort"], []),
    ("flask", "blueprints", 4, 3, ["flask.app_factory"],
     ["Blueprint", "register_blueprint"], []),
    ("flask", "streaming_responses", 4, 4, ["flask.request_response"],
     ["stream_with_context", "generator"], []),
    ("flask", "wsgi_gunicorn", 4, 4, ["flask.app_factory"],
     ["gunicorn", "workers"], []),
    ("flask", "template_jinja2", 4, 3, ["flask.request_response"],
     ["render_template", "Jinja2"], []),

    ("render", "service_types", 4, 3, ["flask.wsgi_gunicorn"],
     ["web service", "background worker", "cron"], []),
    ("render", "environment_secrets", 4, 3,
     ["render.service_types", "security.secrets_management"],
     ["Render env vars", "secret files"], []),
    ("render", "deployment_git", 4, 3, ["render.service_types"],
     ["auto-deploy", "manual deploy"], []),
    ("render", "logs_and_shell", 4, 3, ["render.deployment_git"],
     ["render logs", "shell access"], []),
    ("render", "persistent_disks", 4, 4, ["render.service_types"],
     ["persistent disk", "mount path"], []),
    ("render", "cron_jobs", 4, 3, ["render.service_types"],
     ["cron", "schedule expression"], []),
    ("render", "custom_domains_https", 4, 3, ["render.deployment_git"],
     ["custom domain", "TLS"], []),

    ("selfheal", "watchdog_loop", 4, 4,
     ["concurrency.threading", "obs.logging_structured"],
     ["watchdog", "heartbeat", "process monitor"], []),
    ("selfheal", "auto_repair_component", 4, 5, ["selfheal.watchdog_loop"],
     ["diagnose", "repair", "retry"], []),
    ("selfheal", "clone_swap_deploys", 4, 5, ["selfheal.auto_repair_component"],
     ["blue-green", "clone-swap", "rollback"], []),
    ("selfheal", "backup_rotation", 4, 4, ["db.sqlite_backup_recovery"],
     ["retention", "generational"], []),
    ("selfheal", "chaos_engineering", 4, 5, ["selfheal.auto_repair_component"],
     ["fault injection", "gameday"], []),

    ("agents", "capability_shapes", 4, 4, ["patterns.strategy"],
     ["utility", "trading", "interface"], []),
    ("agents", "insight_promotion", 4, 5, ["agents.capability_shapes"],
     ["insight to capability", "promoter"], []),
    ("agents", "materialisation", 4, 5, ["agents.insight_promotion"],
     ["smoke test", "reviewer", "promoted"], []),
    ("agents", "fresh_blood_channels", 4, 4, ["agents.materialisation"],
     ["arxiv", "github", "wildcard"], []),
    ("agents", "self_scanner_gaps", 4, 5, ["agents.materialisation"],
     ["gap detection", "component drift"], []),
    ("agents", "self_evolution_loop", 4, 5, ["agents.self_scanner_gaps"],
     ["evolve", "generate", "verify"], []),

    ("betting", "probability_odds", 4, 4, ["dsa.algorithm_complexity"],
     ["decimal odds", "implied probability"], []),
    ("betting", "kelly_criterion", 4, 5, ["betting.probability_odds"],
     ["Kelly", "fractional Kelly", "bankroll"], []),
    ("betting", "betfair_api", 4, 4,
     ["betting.probability_odds", "network.retries_backoff"],
     ["betfair", "streaming API"], []),
    ("betting", "greyhound_form_model", 4, 5, ["betting.probability_odds"],
     ["form", "trap draw", "sectional times"], []),
    ("betting", "value_detection", 4, 5, ["betting.kelly_criterion"],
     ["value bet", "edge", "expected value"], []),

    ("trading", "market_data", 4, 4, ["network.retries_backoff"],
     ["OHLCV", "orderbook"], []),
    ("trading", "backtesting", 4, 5, ["trading.market_data"],
     ["walk-forward", "look-ahead bias"], []),
    ("trading", "risk_management", 4, 5, ["trading.backtesting"],
     ["position sizing", "stop loss", "max drawdown"], []),
    ("trading", "execution_algos", 4, 5, ["trading.risk_management"],
     ["TWAP", "VWAP", "iceberg"], []),

    ("content", "publishing_pipeline", 4, 4, ["io.json"],
     ["content queue", "publish"], []),
    ("content", "social_apis", 4, 4, ["network.http_client"],
     ["twitter", "instagram", "youtube"], []),

    ("revenue", "monetisation_streams", 4, 4, ["agents.self_evolution_loop"],
     ["subscription", "affiliate", "yield"], []),
]


# TypeScript / JavaScript
_JS_SPEC: List[tuple] = [
    ("core", "variables_let_const", 1, 1, [], ["let", "const", "var"], []),
    ("core", "types_primitives", 1, 1, ["core.variables_let_const"],
     ["number", "string", "boolean", "null", "undefined"], []),
    ("core", "operators", 1, 1, ["core.types_primitives"],
     ["arithmetic", "comparison", "equality"], []),
    ("core", "control_flow", 1, 2, ["core.operators"],
     ["if", "for", "while", "switch"], []),
    ("core", "functions", 1, 2, ["core.control_flow"],
     ["function", "arrow function", "hoisting"], []),
    ("core", "objects_arrays", 1, 2, ["core.functions"],
     ["object literal", "array"], []),
    ("core", "destructuring", 1, 2, ["core.objects_arrays"],
     ["destructuring", "rest", "spread"], []),
    ("core", "modules_esm", 1, 2, ["core.functions"],
     ["import", "export", "esm"], []),

    ("ts", "type_annotations", 2, 2, ["core.variables_let_const"],
     ["type", ": string", ": number"], []),
    ("ts", "interfaces_types", 2, 3, ["ts.type_annotations"],
     ["interface", "type alias"], []),
    ("ts", "generics", 2, 4, ["ts.interfaces_types"],
     ["<T>", "extends", "keyof"], []),
    ("ts", "narrowing_guards", 2, 4, ["ts.generics"],
     ["typeof", "instanceof", "in", "user-defined"], []),
    ("ts", "utility_types", 2, 4, ["ts.generics"],
     ["Partial", "Pick", "Omit", "Record"], []),
    ("ts", "strict_mode_config", 2, 3, ["ts.type_annotations"],
     ["strict", "noImplicitAny", "tsconfig"], []),

    ("async", "promises", 2, 3, ["core.functions"],
     ["Promise", ".then", "reject"], []),
    ("async", "async_await", 2, 3, ["async.promises"],
     ["async", "await", "try/catch"], []),
    ("async", "event_loop", 3, 4, ["async.async_await"],
     ["microtask", "macrotask"], []),
    ("async", "abort_controller", 3, 4, ["async.async_await"],
     ["AbortController", "signal"], []),

    ("dom", "querying", 2, 2, ["core.objects_arrays"],
     ["querySelector", "getElementById"], []),
    ("dom", "events", 2, 3, ["dom.querying"],
     ["addEventListener", "bubble", "capture"], []),
    ("dom", "forms", 2, 3, ["dom.events"],
     ["form submit", "FormData"], []),
    ("dom", "fetch_api", 2, 3, ["async.async_await"],
     ["fetch", "Response", "AbortController"], []),

    ("node", "modules_cjs_esm", 3, 3, ["core.modules_esm"],
     ["commonjs", "esm interop"], []),
    ("node", "fs_and_streams", 3, 4, ["node.modules_cjs_esm"],
     ["fs", "createReadStream"], []),
    ("node", "http_server", 3, 4, ["node.modules_cjs_esm"],
     ["http.createServer", "express"], []),
    ("node", "process_env", 3, 3, ["node.modules_cjs_esm"],
     ["process.env", "dotenv"], []),
    ("node", "package_json", 3, 3, ["node.modules_cjs_esm"],
     ["dependencies", "scripts"], []),

    ("react", "components_props", 4, 3, ["ts.interfaces_types"],
     ["function component", "props"], []),
    ("react", "hooks_state_effect", 4, 4, ["react.components_props"],
     ["useState", "useEffect"], []),
    ("react", "context_and_reducers", 4, 4, ["react.hooks_state_effect"],
     ["useContext", "useReducer"], []),
    ("react", "data_fetching", 4, 4, ["react.hooks_state_effect"],
     ["swr", "react-query"], []),

    ("test", "jest_vitest", 3, 3, ["node.package_json"],
     ["jest", "vitest", "describe", "it"], []),
    ("test", "mocking_and_spies", 3, 4, ["test.jest_vitest"],
     ["jest.fn", "spyOn"], []),
]


# Bash
_BASH_SPEC: List[tuple] = [
    ("core", "shebang_permissions", 1, 1, [], ["#!/usr/bin/env bash", "chmod +x"], []),
    ("core", "variables_quoting", 1, 2, ["core.shebang_permissions"],
     ["$var", "\"$var\"", "single vs double quotes"], []),
    ("core", "control_flow", 1, 2, ["core.variables_quoting"],
     ["if", "for", "while", "case"], []),
    ("core", "functions", 1, 2, ["core.control_flow"],
     ["function", "return", "arguments"], []),
    ("core", "arrays", 1, 2, ["core.functions"],
     ["arrays", "declare -a"], []),
    ("core", "redirects_pipes", 1, 2, ["core.control_flow"],
     ["<", ">", ">>", "|"], []),
    ("core", "exit_codes", 1, 2, ["core.functions"],
     ["$?", "exit", "set -e"], []),

    ("scripting", "set_flags", 2, 2, ["core.exit_codes"],
     ["set -euo pipefail", "IFS"], []),
    ("scripting", "trap_signals", 2, 3, ["scripting.set_flags"],
     ["trap", "EXIT", "SIGTERM"], []),
    ("scripting", "getopts_argparse", 2, 3, ["scripting.set_flags"],
     ["getopts", "getopt"], []),
    ("scripting", "here_docs", 2, 2, ["core.redirects_pipes"],
     ["<<EOF", "here-doc", "here-string"], []),
    ("scripting", "process_substitution", 2, 3, ["scripting.here_docs"],
     ["<(...)", ">(...)"], []),
    ("scripting", "subshells_and_jobs", 2, 3, ["core.functions"],
     ["(...)", "&", "wait"], []),

    ("tools", "grep_awk_sed", 2, 3, ["core.redirects_pipes"],
     ["grep", "awk", "sed"], []),
    ("tools", "find_xargs", 2, 3, ["tools.grep_awk_sed"],
     ["find", "xargs", "-print0"], []),
    ("tools", "jq_yq", 2, 3, ["tools.grep_awk_sed"],
     ["jq", "yq"], []),
    ("tools", "curl_and_wget", 2, 3, ["core.redirects_pipes"],
     ["curl", "wget", "-H", "-d"], []),

    ("cicd", "github_actions_yaml", 3, 3, ["scripting.set_flags"],
     ["actions", "workflow", "runs-on"], []),
    ("cicd", "docker_basics", 3, 4, ["scripting.set_flags"],
     ["Dockerfile", "docker build"], []),
]


# SQL
_SQL_SPEC: List[tuple] = [
    ("core", "select_where", 1, 1, [], ["SELECT", "WHERE"], []),
    ("core", "insert_update_delete", 1, 2, ["core.select_where"],
     ["INSERT", "UPDATE", "DELETE"], []),
    ("core", "joins", 1, 3, ["core.insert_update_delete"],
     ["INNER JOIN", "LEFT JOIN"], []),
    ("core", "group_by_aggregates", 1, 2, ["core.joins"],
     ["GROUP BY", "COUNT", "SUM", "AVG"], []),
    ("core", "order_limit", 1, 2, ["core.select_where"],
     ["ORDER BY", "LIMIT"], []),
    ("core", "null_handling", 1, 2, ["core.select_where"],
     ["IS NULL", "COALESCE"], []),

    ("intermediate", "subqueries", 2, 3, ["core.joins"],
     ["subquery", "correlated"], []),
    ("intermediate", "cte_with", 2, 3, ["intermediate.subqueries"],
     ["WITH", "CTE", "recursive"], []),
    ("intermediate", "window_functions", 2, 4, ["intermediate.cte_with"],
     ["ROW_NUMBER", "RANK", "PARTITION BY"], []),
    ("intermediate", "case_expressions", 2, 3, ["core.group_by_aggregates"],
     ["CASE WHEN"], []),
    ("intermediate", "date_functions", 2, 2, ["core.select_where"],
     ["strftime", "date()"], []),
    ("intermediate", "json_columns", 2, 3, ["core.select_where"],
     ["json_extract", "->", "->>"], []),

    ("schema", "primary_foreign_keys", 2, 3, ["core.insert_update_delete"],
     ["PRIMARY KEY", "FOREIGN KEY"], []),
    ("schema", "unique_check_constraints", 2, 3, ["schema.primary_foreign_keys"],
     ["UNIQUE", "CHECK"], []),
    ("schema", "indexes", 2, 3, ["schema.primary_foreign_keys"],
     ["CREATE INDEX", "covering index"], []),
    ("schema", "views", 2, 3, ["intermediate.cte_with"],
     ["CREATE VIEW", "materialised view"], []),
    ("schema", "triggers", 3, 4, ["schema.views"],
     ["CREATE TRIGGER", "AFTER INSERT"], []),

    ("perf", "explain_plans", 3, 4, ["schema.indexes"],
     ["EXPLAIN", "EXPLAIN QUERY PLAN"], []),
    ("perf", "index_selectivity", 3, 4, ["perf.explain_plans"],
     ["selectivity", "cardinality"], []),
    ("perf", "denormalisation_tradeoffs", 3, 5, ["perf.index_selectivity"],
     ["denormalise", "read vs write"], []),
]


# Cross-cutting CS (language-agnostic)
_CS_SPEC: List[tuple] = [
    ("dsa", "array_list", 1, 2, [], ["dynamic array", "amortised"], []),
    ("dsa", "linked_list", 1, 3, ["dsa.array_list"],
     ["singly", "doubly"], []),
    ("dsa", "stack_queue", 1, 2, ["dsa.array_list"],
     ["LIFO", "FIFO"], []),
    ("dsa", "hash_table", 2, 3, ["dsa.stack_queue"],
     ["chaining", "open addressing"], []),
    ("dsa", "binary_tree", 2, 3, ["dsa.hash_table"],
     ["BST", "traversal"], []),
    ("dsa", "heap_priority_queue", 2, 4, ["dsa.binary_tree"],
     ["binary heap", "heapify"], []),
    ("dsa", "graph_basics", 2, 4, ["dsa.hash_table"],
     ["adjacency list", "BFS", "DFS"], []),
    ("dsa", "shortest_path", 3, 5, ["dsa.graph_basics"],
     ["Dijkstra", "Bellman-Ford"], []),
    ("dsa", "dynamic_programming", 3, 5, ["dsa.heap_priority_queue"],
     ["memoisation", "tabulation"], []),

    ("git", "basics_add_commit", 1, 2, [], ["add", "commit", "status"], []),
    ("git", "branches_merge", 2, 3, ["git.basics_add_commit"],
     ["branch", "checkout", "merge"], []),
    ("git", "rebase_vs_merge", 2, 3, ["git.branches_merge"],
     ["rebase", "fast-forward"], []),
    ("git", "conflict_resolution", 2, 3, ["git.branches_merge"],
     ["conflict markers", "3-way"], []),
    ("git", "remotes_push_pull", 1, 2, ["git.basics_add_commit"],
     ["origin", "remote", "push"], []),
    ("git", "worktree_stash", 3, 4, ["git.branches_merge"],
     ["stash", "worktree"], []),
    ("git", "bisect_reflog", 3, 4, ["git.branches_merge"],
     ["bisect", "reflog"], []),

    ("methodology", "code_review_practices", 3, 3, ["git.branches_merge"],
     ["PR review", "small diffs"], []),
    ("methodology", "pair_programming", 2, 2, [],
     ["pairing", "driver-navigator"], []),
    ("methodology", "tdd_bdd", 3, 4, [], ["TDD", "BDD", "given-when-then"], []),
    ("methodology", "refactoring_smells", 3, 4, [],
     ["long method", "feature envy"], []),
    ("methodology", "clean_code_principles", 3, 4,
     ["methodology.refactoring_smells"],
     ["SOLID", "DRY", "KISS"], []),
]


# Cross-language prerequisites (added when we materialise the topic).
CROSS_LANG_PREREQ: Dict[str, List[str]] = {
    # Python examples: DSA topics benefit from language-agnostic DSA
    "python.dsa.list":                ["cs.dsa.array_list"],
    "python.dsa.dict":                ["cs.dsa.hash_table"],
    "python.dsa.algorithm_complexity": ["cs.dsa.hash_table"],
    "python.oop.classes_basics":      [],
    "python.db.sqlite_basics":        ["sql.core.select_where"],
    "python.flask.app_factory":       ["python.network.rest_conventions"],
    "python.flask.auth_basic":        ["python.security.auth_basics"],
}


def _materialise(lang: str, spec_rows: List[tuple]) -> Dict[str, dict]:
    """Turn a compact spec into full topic dicts, keyed by slug."""
    out: Dict[str, dict] = {}
    for row in spec_rows:
        area, sub, tier, depth, prereqs_within, keywords, hints = row
        slug = f"{lang}.{area}.{sub}"
        title = f"{lang.upper()}: {area}/{sub}".replace("_", " ")
        prereqs = [f"{lang}.{p}" for p in prereqs_within]
        prereqs += CROSS_LANG_PREREQ.get(slug, [])
        search_hints = list(hints)
        # Default search query: language + area + sub.
        base_query = f"{lang} {area.replace('_', ' ')} {sub.replace('_', ' ')}"
        search_queries = [base_query] + [
            f"{base_query} {kw}" for kw in keywords[:2]
        ]
        out[slug] = {
            "slug":            slug,
            "title":           title,
            "language":        lang,
            "area":            area,
            "subtopic":        sub,
            "tier":            tier,
            "depth":           depth,
            "prerequisites":   prereqs,
            "keywords":        list(keywords),
            "search_queries":  search_queries + search_hints,
        }
    return out


# ── Build the full taxonomy at import time ────────────────────────────────

CURRICULUM_TOPICS: Dict[str, dict] = {}
CURRICULUM_TOPICS.update(_materialise("python", _PYTHON_SPEC))
CURRICULUM_TOPICS.update(_materialise("js", _JS_SPEC))
CURRICULUM_TOPICS.update(_materialise("bash", _BASH_SPEC))
CURRICULUM_TOPICS.update(_materialise("sql", _SQL_SPEC))
CURRICULUM_TOPICS.update(_materialise("cs", _CS_SPEC))


# ── Public API ────────────────────────────────────────────────────────────

def get_topic(slug: str) -> Optional[dict]:
    return CURRICULUM_TOPICS.get(slug)


def all_topic_slugs() -> List[str]:
    return sorted(CURRICULUM_TOPICS.keys())


def prerequisites_of(slug: str) -> List[str]:
    t = CURRICULUM_TOPICS.get(slug)
    return list(t["prerequisites"]) if t else []


def tier_of(slug: str) -> int:
    t = CURRICULUM_TOPICS.get(slug)
    return int(t["tier"]) if t else 0


# ── Sanity check at import ───────────────────────────────────────────────
# Catch dangling prerequisites early rather than at runtime.

_UNKNOWN_PREREQS: List[str] = []
for _slug, _t in CURRICULUM_TOPICS.items():
    for _p in _t["prerequisites"]:
        if _p not in CURRICULUM_TOPICS:
            _UNKNOWN_PREREQS.append(f"{_slug} -> {_p}")

# We don't raise at import time (would kill the app on a typo). Instead
# expose the list for the test suite to assert on.
DANGLING_PREREQUISITES = tuple(_UNKNOWN_PREREQS)
